"""LLM-backed agent implementations.

Each LLM agent honours the same Protocol as its deterministic
counterpart in `agents.py`. Behavior contract:

1. Try LLM with the structured prompt (system + user).
2. Strip markdown fences, locate a JSON object, parse it.
3. Validate against the relevant Pydantic model.
4. On any failure, retry up to `config.max_llm_retries` times, each
   retry prefixed with the previous error so the model can correct.
5. If retries exhausted, fall back to the deterministic agent.
6. Record the path (`llm` vs `fallback`) and retry count in
   `last_run` so the coordinator can include it in metadata.

The fallback path means a missing API key, a provider outage, or a
persistently-malformed LLM output never breaks the pipeline — it just
quietly produces a deterministic conversation. The diversity experiment
can read the metadata to know which conversations were LLM-driven.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import json

from kg_mle.executor.session import ExecutorSession
from kg_mle.generator.agents import (
    Assistant,
    DeterministicAssistant,
    DeterministicPlanner,
    DeterministicUser,
    Planner,
    UserSimulator,
    _grounded_params_by_step,
)
from kg_mle.generator.protocol import (
    AssistantTurn,
    GeneratorConfig,
    Plan,
    UserTurn,
)
from kg_mle.llm.clients import StructuredLLMClient
from kg_mle.registry.models import ToolRegistry
from kg_mle.sampler.constraints import SamplingResult


def _extract_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        # Strip fenced code block (```json ... ```)
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    return json.loads(stripped[start : end + 1])


# ---------- LLM Planner ----------------------------------------------------


_PLANNER_SYSTEM_PROMPT = """You are a scenario planner for a synthetic conversation dataset.
You receive a chain of API endpoints the conversation must use, in order.
Produce a Plan JSON describing the user's underlying intent and per-step parameter plans.

Strict rules:
- Output JSON only — no prose, no markdown fences.
- For each step, list ONLY required parameters of that endpoint.
- For parameters supplied by a previous step (you'll see them marked grounded), set suggested_value to null.
- For free parameters, give a realistic example value and a confidence between 0 and 1.
- Use confidence < 0.6 only when you genuinely cannot infer a default — the assistant will ask the user about those.
- Set ambiguous=true for at most one parameter per conversation, only if the user request is naturally vague.
"""


class LLMPlanner:
    def __init__(
        self,
        *,
        client: StructuredLLMClient,
        registry: ToolRegistry,
        fallback: Planner,
        max_retries: int = 1,
    ) -> None:
        self._client = client
        self._registry = registry
        self._endpoint_index = {e.endpoint_id: e for e in registry.endpoints}
        self._fallback = fallback
        self._max_retries = max_retries
        self.last_run: dict[str, Any] = {}

    def plan(self, sampling_result: SamplingResult, *, seed: int) -> Plan:
        prompt = self._build_prompt(sampling_result)
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            user_prompt = (
                f"{prompt}\n\nPrevious-attempt error: {last_error}" if last_error else prompt
            )
            try:
                content = self._client.complete_json(
                    system=_PLANNER_SYSTEM_PROMPT,
                    user=user_prompt,
                    temperature=0.5,
                )
                payload = _extract_json_object(content)
                plan = Plan.model_validate(payload)
                self._validate_against_chain(plan, sampling_result)
                self.last_run = {"path": "llm", "retries": attempt}
                return plan
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                continue
            except Exception as exc:
                # Network / provider errors. Don't retry — fall back.
                self.last_run = {"path": "fallback", "reason": str(exc), "retries": attempt}
                return self._fallback.plan(sampling_result, seed=seed)
        self.last_run = {"path": "fallback", "reason": "parse_or_validation", "retries": self._max_retries + 1}
        return self._fallback.plan(sampling_result, seed=seed)

    def _build_prompt(self, sampling_result: SamplingResult) -> str:
        grounded_by_step = _grounded_params_by_step(sampling_result)
        steps = []
        for idx, endpoint_id in enumerate(sampling_result.endpoints):
            endpoint = self._endpoint_index[endpoint_id]
            steps.append(
                {
                    "step_index": idx,
                    "endpoint_id": endpoint_id,
                    "description": endpoint.description,
                    "required_parameters": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "canonical_name": p.canonical_name,
                            "description": p.description,
                            "grounded": p.name in grounded_by_step[idx],
                        }
                        for p in endpoint.parameters
                        if p.required
                    ],
                }
            )
        chain_summary = " -> ".join(sampling_result.endpoints)
        return json.dumps(
            {
                "chain": chain_summary,
                "steps": steps,
                "output_schema": {
                    "conversation_intent": "string — one sentence",
                    "user_character": "string — e.g. 'curious', 'efficient'",
                    "plan_confidence": "float 0..1",
                    "step_plans": [
                        {
                            "step_index": "int (match the input)",
                            "endpoint_id": "string (match the input)",
                            "parameter_plans": [
                                {
                                    "parameter_name": "string",
                                    "suggested_value": "any | null (null when grounded)",
                                    "confidence": "float 0..1",
                                    "ambiguous": "bool",
                                    "reason": "string",
                                }
                            ],
                        }
                    ],
                    "ambiguous_step_indices": "list[int]",
                },
            },
            indent=2,
        )

    def _validate_against_chain(self, plan: Plan, sampling_result: SamplingResult) -> None:
        if len(plan.step_plans) != len(sampling_result.endpoints):
            raise ValueError(
                f"Plan has {len(plan.step_plans)} steps but chain has "
                f"{len(sampling_result.endpoints)}."
            )
        for plan_step, chain_endpoint in zip(plan.step_plans, sampling_result.endpoints):
            if plan_step.endpoint_id != chain_endpoint:
                raise ValueError(
                    f"Plan step {plan_step.step_index} endpoint "
                    f"{plan_step.endpoint_id!r} != chain endpoint {chain_endpoint!r}."
                )


# ---------- LLM User Simulator --------------------------------------------


_USER_SYSTEM_PROMPT = """You are simulating a real human user requesting help with API-backed tasks.
Reply naturally, in 1-2 sentences. Output JSON only with the schema given.
"""


class LLMUser:
    def __init__(
        self,
        *,
        client: StructuredLLMClient,
        fallback: UserSimulator,
        max_retries: int = 1,
    ) -> None:
        self._client = client
        self._fallback = fallback
        self._max_retries = max_retries
        self.last_run: dict[str, Any] = {}

    def initial_request(self, plan: Plan, *, seed: int) -> UserTurn:
        prompt = self._build_initial_prompt(plan)
        return self._invoke(prompt, fallback=lambda: self._fallback.initial_request(plan, seed=seed))

    def reply_to_clarification(
        self,
        plan: Plan,
        *,
        target_step: int,
        target_parameter: str,
        seed: int,
    ) -> UserTurn:
        prompt = self._build_reply_prompt(plan, target_step, target_parameter)
        return self._invoke(
            prompt,
            fallback=lambda: self._fallback.reply_to_clarification(
                plan,
                target_step=target_step,
                target_parameter=target_parameter,
                seed=seed,
            ),
        )

    def _invoke(self, prompt: str, *, fallback) -> UserTurn:
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            user_prompt = (
                f"{prompt}\n\nPrevious-attempt error: {last_error}" if last_error else prompt
            )
            try:
                content = self._client.complete_json(
                    system=_USER_SYSTEM_PROMPT,
                    user=user_prompt,
                    temperature=0.7,
                )
                payload = _extract_json_object(content)
                turn = UserTurn.model_validate(payload)
                self.last_run = {"path": "llm", "retries": attempt}
                return turn
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                continue
            except Exception as exc:
                self.last_run = {"path": "fallback", "reason": str(exc), "retries": attempt}
                return fallback()
        self.last_run = {"path": "fallback", "reason": "parse_or_validation", "retries": self._max_retries + 1}
        return fallback()

    @staticmethod
    def _build_initial_prompt(plan: Plan) -> str:
        return json.dumps(
            {
                "intent": plan.conversation_intent,
                "character": plan.user_character,
                "output_schema": {
                    "content": "string — the user's natural-language opening message",
                    "is_initial_request": True,
                    "is_clarification_reply": False,
                },
            },
            indent=2,
        )

    @staticmethod
    def _build_reply_prompt(plan: Plan, target_step: int, target_parameter: str) -> str:
        target_value = None
        if 0 <= target_step < len(plan.step_plans):
            for pp in plan.step_plans[target_step].parameter_plans:
                if pp.parameter_name == target_parameter:
                    target_value = pp.suggested_value
                    break
        return json.dumps(
            {
                "intent": plan.conversation_intent,
                "asked_about_parameter": target_parameter,
                "your_preferred_value": target_value,
                "instruction": "Briefly answer the assistant's clarifying question.",
                "output_schema": {
                    "content": "string",
                    "is_initial_request": False,
                    "is_clarification_reply": True,
                },
            },
            indent=2,
        )


# ---------- LLM Assistant --------------------------------------------------


_ASSISTANT_SYSTEM_PROMPT = """You are an AI assistant responding inside a multi-turn conversation that uses tools.
You receive:
- the plan (intent, step plans, ambiguous parameters),
- the conversation so far,
- the executor's plausible default arguments for the current step,
- few-shot example values for each parameter.

Decide one of three actions for this turn:
- 'clarification' — ask the user for a missing or low-confidence value.
- 'tool_calls' — emit one or more tool_calls with concrete arguments.
- 'final_summary' — wrap up because the chain is complete.

Output JSON matching the AssistantTurn schema strictly. Never invent IDs.
Use only values present in example_values or returned by previous tool calls.
"""


class LLMAssistant:
    def __init__(
        self,
        *,
        client: StructuredLLMClient,
        fallback: Assistant,
        max_retries: int = 1,
    ) -> None:
        self._client = client
        self._fallback = fallback
        self._max_retries = max_retries
        self.last_run: dict[str, Any] = {}

    def compose_turn(
        self,
        plan: Plan,
        transcript: list[dict],
        session: ExecutorSession,
        *,
        steps_completed: int,
        clarifications_taken: int,
        config: GeneratorConfig,
        seed: int,
    ) -> AssistantTurn:
        if steps_completed >= len(plan.step_plans):
            return self._fallback.compose_turn(
                plan,
                transcript,
                session,
                steps_completed=steps_completed,
                clarifications_taken=clarifications_taken,
                config=config,
                seed=seed,
            )

        prompt = self._build_prompt(plan, transcript, session, steps_completed, config)
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            user_prompt = (
                f"{prompt}\n\nPrevious-attempt error: {last_error}" if last_error else prompt
            )
            try:
                content = self._client.complete_json(
                    system=_ASSISTANT_SYSTEM_PROMPT,
                    user=user_prompt,
                    temperature=0.4,
                )
                payload = _extract_json_object(content)
                turn = AssistantTurn.model_validate(payload)
                self.last_run = {"path": "llm", "retries": attempt}
                return turn
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                continue
            except Exception as exc:
                self.last_run = {"path": "fallback", "reason": str(exc), "retries": attempt}
                return self._fallback.compose_turn(
                    plan,
                    transcript,
                    session,
                    steps_completed=steps_completed,
                    clarifications_taken=clarifications_taken,
                    config=config,
                    seed=seed,
                )
        self.last_run = {"path": "fallback", "reason": "parse_or_validation", "retries": self._max_retries + 1}
        return self._fallback.compose_turn(
            plan,
            transcript,
            session,
            steps_completed=steps_completed,
            clarifications_taken=clarifications_taken,
            config=config,
            seed=seed,
        )

    @staticmethod
    def _build_prompt(
        plan: Plan,
        transcript: list[dict],
        session: ExecutorSession,
        steps_completed: int,
        config: GeneratorConfig,
    ) -> str:
        current = plan.step_plans[steps_completed]
        return json.dumps(
            {
                "plan": plan.model_dump(),
                "transcript": transcript,
                "current_step": current.model_dump(),
                "suggested_args": session.suggest_arguments(current.endpoint_id),
                "example_values": session.example_values(current.endpoint_id),
                "thresholds": {
                    "planner_param_low_confidence": config.planner_param_low_confidence,
                    "assistant_clarification_threshold": config.assistant_clarification_threshold,
                    "assistant_deviation_threshold": config.assistant_deviation_threshold,
                },
                "output_schema": "AssistantTurn — see system prompt",
            },
            indent=2,
            default=str,
        )


def make_llm_agents(
    *,
    client: StructuredLLMClient,
    registry: ToolRegistry,
    config: GeneratorConfig | None = None,
) -> tuple[LLMPlanner, LLMUser, LLMAssistant]:
    """Convenience builder. Each LLM agent wraps the deterministic version
    as fallback, so the coordinator's protocol is satisfied even when the
    LLM is unavailable."""
    config = config or GeneratorConfig()
    return (
        LLMPlanner(
            client=client,
            registry=registry,
            fallback=DeterministicPlanner(registry, config=config),
            max_retries=config.max_llm_retries,
        ),
        LLMUser(
            client=client,
            fallback=DeterministicUser(),
            max_retries=config.max_llm_retries,
        ),
        LLMAssistant(
            client=client,
            fallback=DeterministicAssistant(),
            max_retries=config.max_llm_retries,
        ),
    )
