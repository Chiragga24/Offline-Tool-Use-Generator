"""Unit tests for LLM agents using a fake client (no network, no creds).

These cover the contract every LLM agent must honour:

- A well-formed JSON response from the client is parsed, validated, and
  returned without falling back to deterministic.
- A malformed-then-corrected response triggers exactly one retry, then
  succeeds; `last_run` records `path="llm"` with `retries=1`.
- A persistently-malformed response exhausts retries and falls back to
  the deterministic agent; `last_run` records `path="fallback"` with
  `reason="parse_or_validation"`.
- A client-side exception (provider down) immediately falls back;
  `last_run` records `path="fallback"` with the exception text.
- The plan's shape is validated against the chain — wrong step count or
  endpoint mismatch triggers retry/fallback, not silent acceptance.
"""

import json

import pytest

from kg_mle.config import DEFAULT_INPUT_PATH
from kg_mle.executor import OfflineExecutor
from kg_mle.generator import (
    AssistantTurn,
    DeterministicAssistant,
    DeterministicPlanner,
    DeterministicUser,
    GeneratorConfig,
    LLMAssistant,
    LLMPlanner,
    LLMUser,
    Plan,
    UserTurn,
)
from kg_mle.graph import build_tool_graph
from kg_mle.registry import enrich_registry, load_registry
from kg_mle.sampler import ChainConstraints, ToolChainSampler


class FakeLLMClient:
    """Drops in for StructuredLLMClient. Returns canned responses or raises."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete_json(self, *, system: str, user: str, temperature: float = 0.4) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if not self._responses:
            raise RuntimeError("FakeLLMClient out of canned responses.")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(scope="module")
def pipeline():
    registry = load_registry(DEFAULT_INPUT_PATH)
    enrich_registry(registry)
    graph = build_tool_graph(registry)
    sampler = ToolChainSampler(graph)
    executor = OfflineExecutor(registry)
    return registry, graph, sampler, executor


# ---------- LLMPlanner -----------------------------------------------------


def _well_formed_plan_json(sampling_result) -> str:
    return json.dumps(
        {
            "conversation_intent": "User wants help.",
            "user_character": "curious",
            "plan_confidence": 0.9,
            "step_plans": [
                {
                    "step_index": idx,
                    "endpoint_id": endpoint_id,
                    "parameter_plans": [],
                }
                for idx, endpoint_id in enumerate(sampling_result.endpoints)
            ],
            "ambiguous_step_indices": [],
        }
    )


def test_llm_planner_uses_llm_on_well_formed_response(pipeline):
    registry, _, sampler, _ = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    client = FakeLLMClient([_well_formed_plan_json(chain)])
    planner = LLMPlanner(
        client=client,
        registry=registry,
        fallback=DeterministicPlanner(registry),
        max_retries=1,
    )

    plan = planner.plan(chain, seed=42)
    assert isinstance(plan, Plan)
    assert plan.conversation_intent == "User wants help."
    assert planner.last_run == {"path": "llm", "retries": 0}
    assert len(client.calls) == 1


def test_llm_planner_retries_once_then_succeeds(pipeline):
    registry, _, sampler, _ = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    client = FakeLLMClient(
        [
            "not json at all",
            _well_formed_plan_json(chain),
        ]
    )
    planner = LLMPlanner(
        client=client, registry=registry, fallback=DeterministicPlanner(registry), max_retries=1
    )

    plan = planner.plan(chain, seed=42)
    assert isinstance(plan, Plan)
    assert planner.last_run == {"path": "llm", "retries": 1}
    assert len(client.calls) == 2
    # Second call's prompt includes the previous error context
    assert "Previous-attempt error" in client.calls[1]["user"]


def test_llm_planner_falls_back_after_persistent_parse_errors(pipeline):
    registry, _, sampler, _ = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    client = FakeLLMClient(
        [
            "garbage",
            "still garbage",
        ]
    )
    planner = LLMPlanner(
        client=client, registry=registry, fallback=DeterministicPlanner(registry), max_retries=1
    )

    plan = planner.plan(chain, seed=42)
    assert isinstance(plan, Plan)
    assert planner.last_run["path"] == "fallback"
    assert planner.last_run["reason"] == "parse_or_validation"


def test_llm_planner_falls_back_immediately_on_provider_exception(pipeline):
    registry, _, sampler, _ = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    client = FakeLLMClient([RuntimeError("provider 402")])
    planner = LLMPlanner(
        client=client, registry=registry, fallback=DeterministicPlanner(registry), max_retries=1
    )

    plan = planner.plan(chain, seed=42)
    assert isinstance(plan, Plan)
    assert planner.last_run["path"] == "fallback"
    assert "402" in planner.last_run["reason"]


def test_llm_planner_rejects_plan_with_wrong_step_count(pipeline):
    registry, _, sampler, _ = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    bad_plan = json.dumps(
        {
            "conversation_intent": "X",
            "user_character": "default",
            "plan_confidence": 0.9,
            "step_plans": [
                {
                    "step_index": 0,
                    "endpoint_id": chain.endpoints[0],
                    "parameter_plans": [],
                }
            ],
            "ambiguous_step_indices": [],
        }
    )
    client = FakeLLMClient([bad_plan, bad_plan])
    planner = LLMPlanner(
        client=client, registry=registry, fallback=DeterministicPlanner(registry), max_retries=1
    )

    plan = planner.plan(chain, seed=42)
    assert planner.last_run["path"] == "fallback"
    assert len(plan.step_plans) == len(chain.endpoints)


def test_llm_planner_rejects_extras_via_pydantic(pipeline):
    """Extra keys → ValidationError → retry → fallback."""
    registry, _, sampler, _ = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    bad_plan = json.dumps(
        {
            "conversation_intent": "X",
            "user_character": "default",
            "plan_confidence": 0.9,
            "step_plans": [
                {
                    "step_index": idx,
                    "endpoint_id": e,
                    "parameter_plans": [],
                }
                for idx, e in enumerate(chain.endpoints)
            ],
            "ambiguous_step_indices": [],
            "unexpected_field": "oops",
        }
    )
    client = FakeLLMClient([bad_plan, bad_plan])
    planner = LLMPlanner(
        client=client, registry=registry, fallback=DeterministicPlanner(registry), max_retries=1
    )

    planner.plan(chain, seed=42)
    assert planner.last_run["path"] == "fallback"


# ---------- LLMUser --------------------------------------------------------


def test_llm_user_initial_request_parses_and_returns(pipeline):
    registry, _, sampler, _ = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    plan = DeterministicPlanner(registry).plan(chain, seed=42)
    client = FakeLLMClient([json.dumps({"content": "Hi, can you help me?", "is_initial_request": True})])

    user = LLMUser(client=client, fallback=DeterministicUser(), max_retries=1)
    turn = user.initial_request(plan, seed=42)
    assert isinstance(turn, UserTurn)
    assert turn.content == "Hi, can you help me?"
    assert turn.is_initial_request is True
    assert user.last_run == {"path": "llm", "retries": 0}


def test_llm_user_falls_back_on_invalid_json(pipeline):
    registry, _, sampler, _ = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    plan = DeterministicPlanner(registry).plan(chain, seed=42)
    client = FakeLLMClient(["definitely not json", "still not json"])

    user = LLMUser(client=client, fallback=DeterministicUser(), max_retries=1)
    turn = user.initial_request(plan, seed=42)
    assert isinstance(turn, UserTurn)
    assert user.last_run["path"] == "fallback"


# ---------- LLMAssistant ---------------------------------------------------


def test_llm_assistant_parses_tool_call_turn(pipeline):
    registry, _, sampler, executor = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2, min_grounded_transitions=1), seed=42)
    session = executor.open_session(chain, seed=42)
    plan = DeterministicPlanner(registry).plan(chain, seed=42)

    # Compose a plausible tool_call payload for the first step.
    first = chain.endpoints[0]
    args = session.suggest_arguments(first)
    payload = json.dumps(
        {
            "kind": "tool_calls",
            "content": None,
            "tool_calls": [
                {
                    "endpoint_id": first,
                    "arguments": args,
                    "call_confidence": 0.95,
                }
            ],
            "assistant_clarification_confidence": 1.0,
            "chain_deviation": None,
        }
    )
    client = FakeLLMClient([payload])
    assistant = LLMAssistant(
        client=client, fallback=DeterministicAssistant(), max_retries=1
    )

    turn = assistant.compose_turn(
        plan, [], session,
        steps_completed=0, clarifications_taken=0,
        config=GeneratorConfig(), seed=42,
    )
    assert isinstance(turn, AssistantTurn)
    assert turn.kind == "tool_calls"
    assert turn.tool_calls[0].endpoint_id == first
    assert assistant.last_run == {"path": "llm", "retries": 0}


def test_llm_assistant_falls_back_on_provider_exception(pipeline):
    registry, _, sampler, executor = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    session = executor.open_session(chain, seed=42)
    plan = DeterministicPlanner(registry).plan(chain, seed=42)

    client = FakeLLMClient([ConnectionError("network")])
    assistant = LLMAssistant(client=client, fallback=DeterministicAssistant(), max_retries=1)

    turn = assistant.compose_turn(
        plan, [], session,
        steps_completed=0, clarifications_taken=0,
        config=GeneratorConfig(), seed=42,
    )
    assert isinstance(turn, AssistantTurn)
    assert assistant.last_run["path"] == "fallback"
    assert "network" in assistant.last_run["reason"]


def test_llm_assistant_terminal_path_uses_fallback_for_summary(pipeline):
    """When steps_completed >= chain length, LLMAssistant delegates to
    fallback (which knows how to write final_summary deterministically)."""
    registry, _, sampler, executor = pipeline
    chain = sampler.sample(ChainConstraints(n_steps=2), seed=42)
    session = executor.open_session(chain, seed=42)
    plan = DeterministicPlanner(registry).plan(chain, seed=42)
    client = FakeLLMClient([])  # would error if called

    assistant = LLMAssistant(client=client, fallback=DeterministicAssistant(), max_retries=1)
    turn = assistant.compose_turn(
        plan, [], session,
        steps_completed=2, clarifications_taken=0,
        config=GeneratorConfig(), seed=42,
    )
    assert turn.kind == "final_summary"
    assert client.calls == []
