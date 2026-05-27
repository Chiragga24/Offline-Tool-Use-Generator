"""Deterministic agent implementations.

These are the always-on, no-credentials versions of the three agents.
They produce structurally valid conversations that satisfy the rubric's
hard requirements (multi-turn disambiguation, valid tool calls, role
tagging, coherent chaining via the executor) without natural-language
realism.

LLM-backed variants live in `llm_agents.py`. The protocols below define
the contract both implementations honour, so the coordinator is agnostic
about which one it's calling.
"""

from __future__ import annotations

import random
from typing import Protocol

from kg_mle.executor.mocks import CANONICAL_EXAMPLES
from kg_mle.executor.session import ExecutorSession
from kg_mle.generator.protocol import (
    AssistantTurn,
    ClarificationTarget,
    GeneratorConfig,
    ParameterPlan,
    Plan,
    StepPlan,
    ToolCallProposal,
    UserTurn,
)
from kg_mle.registry.models import ToolRegistry
from kg_mle.sampler.constraints import SamplingResult


# ----- Protocols ----------------------------------------------------------


class Planner(Protocol):
    def plan(self, sampling_result: SamplingResult, *, seed: int) -> Plan: ...


class UserSimulator(Protocol):
    def initial_request(self, plan: Plan, *, seed: int) -> UserTurn: ...

    def reply_to_clarification(
        self,
        plan: Plan,
        *,
        target_step: int,
        target_parameter: str,
        seed: int,
    ) -> UserTurn: ...


class Assistant(Protocol):
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
    ) -> AssistantTurn: ...


# ----- Deterministic Planner ----------------------------------------------


class DeterministicPlanner:
    """Builds a Plan from a SamplingResult using only deterministic rules.

    Parameter values come from the canonical example pool when available,
    fall back to a placeholder template otherwise. Ambiguity is injected
    on a configurable fraction of conversations (seeded), making the
    planner-driven disambiguation rate tunable for the corpus.
    """

    def __init__(self, registry: ToolRegistry, *, config: GeneratorConfig | None = None) -> None:
        self._endpoint_index = {endpoint.endpoint_id: endpoint for endpoint in registry.endpoints}
        self._config = config or GeneratorConfig()

    def plan(self, sampling_result: SamplingResult, *, seed: int) -> Plan:
        rng = random.Random(seed)
        grounded_by_step = _grounded_params_by_step(sampling_result)
        step_plans = [
            self._build_step_plan(idx, endpoint_id, grounded_by_step[idx], rng)
            for idx, endpoint_id in enumerate(sampling_result.endpoints)
        ]
        ambiguous_step_indices = self._pick_ambiguous_steps(step_plans, rng)
        for step_index in ambiguous_step_indices:
            self._mark_ambiguous_param(step_plans[step_index])
        return Plan(
            conversation_intent=_compose_intent(sampling_result, self._endpoint_index),
            user_character="default",
            plan_confidence=1.0,
            step_plans=step_plans,
            ambiguous_step_indices=ambiguous_step_indices,
        )

    def _build_step_plan(
        self,
        step_index: int,
        endpoint_id: str,
        grounded_params: set[str],
        rng: random.Random,
    ) -> StepPlan:
        endpoint = self._endpoint_index[endpoint_id]
        parameter_plans: list[ParameterPlan] = []
        for param in endpoint.parameters:
            if not param.required:
                continue
            if param.name in grounded_params:
                parameter_plans.append(
                    ParameterPlan(
                        parameter_name=param.name,
                        suggested_value=None,
                        confidence=1.0,
                        ambiguous=False,
                        reason="Grounded — value supplied at runtime by previous step.",
                    )
                )
                continue
            canonical = param.canonical_name or param.name
            pool = CANONICAL_EXAMPLES.get(canonical)
            if pool:
                parameter_plans.append(
                    ParameterPlan(
                        parameter_name=param.name,
                        suggested_value=rng.choice(pool),
                        confidence=1.0,
                        ambiguous=False,
                        reason=f"Canonical example pool for {canonical!r}.",
                    )
                )
            else:
                # Unknown canonical → planner is less sure. Low confidence
                # invites the assistant to ask (when assistant judgment is
                # also high) and is what hits the confidence-gated
                # initiative branch in the coordinator.
                parameter_plans.append(
                    ParameterPlan(
                        parameter_name=param.name,
                        suggested_value=f"<{param.name}>",
                        confidence=0.5,
                        ambiguous=False,
                        reason="No canonical example pool; placeholder default.",
                    )
                )
        return StepPlan(
            step_index=step_index,
            endpoint_id=endpoint_id,
            parameter_plans=parameter_plans,
        )

    def _pick_ambiguous_steps(self, step_plans: list[StepPlan], rng: random.Random) -> list[int]:
        if not step_plans:
            return []
        if rng.random() > self._config.ambiguity_fraction:
            return []
        # Pick step 0 — initial-request ambiguity is the most natural
        # surface for "user didn't specify everything they wanted."
        return [0]

    def _mark_ambiguous_param(self, step_plan: StepPlan) -> None:
        for param_plan in step_plan.parameter_plans:
            if param_plan.suggested_value is None:
                continue
            param_plan.ambiguous = True
            param_plan.confidence = min(param_plan.confidence, 0.5)
            return


# ----- Deterministic User Simulator ---------------------------------------


class DeterministicUser:
    """Emits user turns as fixed templates. No natural-language realism.

    The initial request is derived from `plan.conversation_intent`.
    Clarification replies look up the relevant parameter's
    `suggested_value` (or canonical pool entry) and emit
    `"For X, use Y."` style text. The coordinator passes in the
    parameter name explicitly so the user simulator never has to parse
    the assistant's content.
    """

    def initial_request(self, plan: Plan, *, seed: int) -> UserTurn:
        return UserTurn(content=plan.conversation_intent, is_initial_request=True)

    def reply_to_clarification(
        self,
        plan: Plan,
        *,
        target_step: int,
        target_parameter: str,
        seed: int,
    ) -> UserTurn:
        rng = random.Random(seed)
        value = self._resolve_value(plan, target_step, target_parameter, rng)
        return UserTurn(
            content=f"For {target_parameter}, use {value}.",
            is_clarification_reply=True,
            clarified_value=str(value),
        )

    @staticmethod
    def _resolve_value(plan: Plan, step_index: int, param_name: str, rng: random.Random) -> str:
        if 0 <= step_index < len(plan.step_plans):
            step = plan.step_plans[step_index]
            for pp in step.parameter_plans:
                if pp.parameter_name == param_name and pp.suggested_value not in (None, f"<{param_name}>"):
                    return str(pp.suggested_value)
        # Fall back to canonical pool, then a generic answer
        pool = CANONICAL_EXAMPLES.get(param_name) or CANONICAL_EXAMPLES.get(param_name.removesuffix("_id"))
        if pool:
            return rng.choice(pool)
        return f"any_{param_name}"


# ----- Deterministic Assistant --------------------------------------------


class DeterministicAssistant:
    """Picks one of three turn kinds based on plan + state.

    Decision order (turns through one conversation):
      1. If a planner-marked ambiguous step is current and we haven't
         clarified it yet → clarification turn.
      2. Else if a low-confidence free parameter exists in the current
         step *and* assistant judges its own confidence to ask as high
         enough → clarification turn (assistant-initiative branch).
      3. Else if more steps remain → tool_calls turn using the executor's
         suggest_arguments.
      4. Else → final_summary turn.

    The deterministic assistant never proposes chain deviations. LLM
    variants may.
    """

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
        n_steps = len(plan.step_plans)
        if steps_completed >= n_steps:
            return self._final_summary(transcript)

        current_step = plan.step_plans[steps_completed]

        # Branch 1: planner-driven clarification on current step
        if (
            steps_completed in plan.ambiguous_step_indices
            and not _already_clarified_step(transcript, steps_completed)
        ):
            ambiguous_param = next(
                (pp for pp in current_step.parameter_plans if pp.ambiguous),
                None,
            )
            if ambiguous_param is not None:
                return AssistantTurn(
                    kind="clarification",
                    content=(
                        f"Before I proceed, could you tell me which "
                        f"{ambiguous_param.parameter_name} to use for "
                        f"{_human(current_step.endpoint_id)}?"
                    ),
                    clarification_target=ClarificationTarget(
                        step_index=steps_completed,
                        parameter_name=ambiguous_param.parameter_name,
                    ),
                    assistant_clarification_confidence=1.0,
                )

        # Branch 2: assistant-initiative clarification when planner
        # confidence is low on a free param.
        low_conf_param = next(
            (
                pp
                for pp in current_step.parameter_plans
                if pp.suggested_value is not None
                and pp.confidence < config.planner_param_low_confidence
                and not _already_clarified_param(transcript, pp.parameter_name)
            ),
            None,
        )
        if low_conf_param is not None:
            assistant_conf = 1.0 - low_conf_param.confidence
            if assistant_conf >= config.assistant_clarification_threshold:
                return AssistantTurn(
                    kind="clarification",
                    content=(
                        f"The default I have for {low_conf_param.parameter_name} "
                        f"isn't confident — what value should I use?"
                    ),
                    clarification_target=ClarificationTarget(
                        step_index=steps_completed,
                        parameter_name=low_conf_param.parameter_name,
                    ),
                    assistant_clarification_confidence=assistant_conf,
                )

        # Branch 3: tool call
        proposal = self._compose_tool_call(current_step, session)
        return AssistantTurn(
            kind="tool_calls",
            tool_calls=[proposal],
            assistant_clarification_confidence=1.0,
        )

    @staticmethod
    def _compose_tool_call(step: StepPlan, session: ExecutorSession) -> ToolCallProposal:
        args = session.suggest_arguments(step.endpoint_id)
        # Overlay planner-supplied values for free parameters; grounded
        # parameters already have the right runtime value from
        # suggest_arguments.
        for param_plan in step.parameter_plans:
            if param_plan.suggested_value is None:
                continue
            if param_plan.suggested_value == f"<{param_plan.parameter_name}>":
                continue  # placeholder; keep suggest_arguments default
            args[param_plan.parameter_name] = param_plan.suggested_value
        return ToolCallProposal(
            endpoint_id=step.endpoint_id,
            arguments=args,
            call_confidence=1.0,
        )

    @staticmethod
    def _final_summary(transcript: list[dict]) -> AssistantTurn:
        completed_endpoints = [
            entry["tool_calls"][0]["endpoint_id"]
            for entry in transcript
            if entry.get("role") == "assistant"
            and entry.get("tool_calls")
        ]
        unique = list(dict.fromkeys(completed_endpoints))
        if unique:
            content = "Done. I completed: " + ", ".join(_human(eid) for eid in unique) + "."
        else:
            content = "I wasn't able to complete any of the requested actions."
        return AssistantTurn(kind="final_summary", content=content)


# ----- Helpers ------------------------------------------------------------


def _grounded_params_by_step(sampling_result: SamplingResult) -> list[set[str]]:
    """For each step index, the set of parameter names supplied by a
    grounded transition into that step."""
    grounded: list[set[str]] = [set() for _ in sampling_result.endpoints]
    for transition in sampling_result.transitions:
        if transition.advance_type != "grounded" or transition.parameter is None:
            continue
        try:
            idx = sampling_result.endpoints.index(transition.target)
        except ValueError:
            continue
        grounded[idx].add(transition.parameter)
    return grounded


def _compose_intent(sampling_result: SamplingResult, endpoint_index: dict) -> str:
    first = endpoint_index[sampling_result.endpoints[0]]
    first_desc = first.description.lower().rstrip(".")
    n = len(sampling_result.endpoints)
    domains = sorted({eid.split("/", 1)[0] for eid in sampling_result.endpoints})
    if n == 1:
        return f"I'd like help to {first_desc}."
    if len(domains) == 1:
        return f"I'd like help to {first_desc}, then handle {n - 1} related step(s)."
    return (
        f"I'd like help to {first_desc}, then follow up with {n - 1} more action(s) "
        f"across {', '.join(domains)}."
    )


def _human(endpoint_id: str) -> str:
    """Human-readable rendering of an endpoint_id ('domain/name' → 'name')."""
    return endpoint_id.split("/", 1)[-1].replace("_", " ")


def _already_clarified_step(transcript: list[dict], step_index: int) -> bool:
    for entry in transcript:
        if entry.get("role") == "assistant" and entry.get("clarification_target_step") == step_index:
            return True
    return False


def _already_clarified_param(transcript: list[dict], parameter_name: str) -> bool:
    for entry in transcript:
        if entry.get("role") == "assistant" and entry.get("clarification_target_parameter") == parameter_name:
            return True
    return False
