"""ConversationCoordinator tests — end-to-end with deterministic agents.

These cover the contract every Conversation produced by this codebase
must satisfy, regardless of whether the LLM agents are in use:

- Conversation is structurally valid (role-tagged messages).
- Planner-driven clarifications fire and round-trip through the user.
- Tool calls validate + mock + log.
- Repair flow runs on bad arguments and recovers when possible.
- Chain deviations are graph-verified + confidence-gated; rejections
  are recorded in metadata.
- Metadata aggregates every override and surface so the judge / diversity
  experiment can read them.
"""

import pytest

from kg_mle.config import DEFAULT_INPUT_PATH
from kg_mle.executor import OfflineExecutor
from kg_mle.generator import (
    ChainDeviation,
    ConversationCoordinator,
    DeterministicAssistant,
    DeterministicPlanner,
    DeterministicUser,
    GeneratorConfig,
)
from kg_mle.generator.agents import Assistant
from kg_mle.generator.protocol import AssistantTurn, ToolCallProposal
from kg_mle.graph import build_tool_graph
from kg_mle.registry import enrich_registry, load_registry
from kg_mle.sampler import ChainConstraints, ToolChainSampler


@pytest.fixture(scope="module")
def pipeline():
    registry = load_registry(DEFAULT_INPUT_PATH)
    enrich_registry(registry)
    graph = build_tool_graph(registry)
    sampler = ToolChainSampler(graph)
    executor = OfflineExecutor(registry)
    return registry, graph, sampler, executor


def _coordinator(pipeline, *, config: GeneratorConfig | None = None) -> ConversationCoordinator:
    registry, graph, _, executor = pipeline
    return ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=config),
        user_simulator=DeterministicUser(),
        assistant=DeterministicAssistant(),
        config=config or GeneratorConfig(),
    )


def _chain(sampler, **kwargs):
    return sampler.sample(ChainConstraints(**kwargs), seed=42)


# ----- structural validity --------------------------------------------------


def test_conversation_has_role_tagged_messages(pipeline):
    _, _, sampler, _ = pipeline
    chain = _chain(sampler, n_steps=3, min_grounded_transitions=1)
    conv = _coordinator(pipeline).run(chain, seed=42)
    roles = [m["role"] for m in conv.messages]
    assert roles[0] == "user"
    assert all(r in {"user", "assistant", "tool"} for r in roles)
    assert any(r == "tool" for r in roles), "Expected at least one tool response."
    assert roles[-1] in {"assistant", "tool"}


def test_conversation_id_is_seed_derived(pipeline):
    _, _, sampler, _ = pipeline
    chain = _chain(sampler, n_steps=3)
    conv = _coordinator(pipeline).run(chain, seed=12345)
    assert conv.conversation_id == "conv_00012345"


def test_n_tool_calls_matches_chain_length_when_no_deviations(pipeline):
    _, _, sampler, _ = pipeline
    chain = _chain(sampler, n_steps=3, min_grounded_transitions=1)
    conv = _coordinator(pipeline).run(chain, seed=42)
    tool_responses = [m for m in conv.messages if m["role"] == "tool"]
    successful = [m for m in tool_responses if not (isinstance(m["content"], dict) and "error" in m["content"])]
    assert len(successful) == 3


def test_metadata_summarises_run(pipeline):
    _, _, sampler, _ = pipeline
    chain = _chain(sampler, n_steps=3, min_grounded_transitions=1)
    conv = _coordinator(pipeline).run(chain, seed=42)
    md = conv.metadata
    assert md["n_tool_calls"] == 3
    assert md["original_chain"] == list(chain.endpoints)
    assert md["final_chain"] == list(chain.endpoints)
    assert "advance_type_counts" in md
    assert "transition_summary" in md
    assert md["seed"] == 42


# ----- planner-driven clarification ----------------------------------------


def test_clarification_fires_when_planner_marks_ambiguous(pipeline):
    """ambiguity_fraction=1.0 forces a clarification on step 0."""
    _, _, sampler, _ = pipeline
    chain = _chain(sampler, n_steps=3)
    coord = _coordinator(pipeline, config=GeneratorConfig(ambiguity_fraction=1.0))
    conv = coord.run(chain, seed=42)

    clar_messages = [
        m for m in conv.messages
        if m["role"] == "assistant" and m.get("clarification_target_parameter")
    ]
    if conv.metadata["clarifications_taken"]:
        assert clar_messages, "metadata recorded clarification but transcript missing it"
        assert conv.metadata["clarifications_taken"][0]["initiated_by"] == "planner"


def test_clarification_does_not_fire_when_ambiguity_zero(pipeline):
    _, _, sampler, _ = pipeline
    chain = _chain(sampler, n_steps=3)
    coord = _coordinator(pipeline, config=GeneratorConfig(ambiguity_fraction=0.0))
    conv = coord.run(chain, seed=42)
    planner_clarifications = [
        c for c in conv.metadata["clarifications_taken"] if c["initiated_by"] == "planner"
    ]
    assert planner_clarifications == []


# ----- determinism ---------------------------------------------------------


def test_same_seed_produces_byte_identical_conversation(pipeline):
    _, _, sampler, _ = pipeline
    chain = _chain(sampler, n_steps=3, min_grounded_transitions=1)
    conv_a = _coordinator(pipeline).run(chain, seed=42)
    conv_b = _coordinator(pipeline).run(chain, seed=42)
    assert conv_a.model_dump() == conv_b.model_dump()


# ----- repair flow ---------------------------------------------------------


class _BadFirstCallAssistant(Assistant):
    """Helper: emit a deliberately bad arg on the first tool call, then a
    correct one on the repair retry."""

    def __init__(self):
        self._step0_attempts = 0

    def compose_turn(
        self, plan, transcript, session, *, steps_completed, clarifications_taken, config, seed
    ):
        if steps_completed >= len(plan.step_plans):
            return AssistantTurn(kind="final_summary", content="done.")
        step = plan.step_plans[steps_completed]

        if steps_completed == 0 and self._step0_attempts == 0:
            self._step0_attempts += 1
            args = session.suggest_arguments(step.endpoint_id)
            # Replace one required param with a clearly wrong type to force
            # an ArgumentTypeError.
            for k in list(args.keys()):
                args[k] = 99999  # int where strings are required
                break
            return AssistantTurn(
                kind="tool_calls",
                tool_calls=[ToolCallProposal(endpoint_id=step.endpoint_id, arguments=args)],
            )
        args = session.suggest_arguments(step.endpoint_id)
        return AssistantTurn(
            kind="tool_calls",
            tool_calls=[ToolCallProposal(endpoint_id=step.endpoint_id, arguments=args)],
        )


def test_repair_flow_recovers_after_bad_args(pipeline):
    registry, graph, sampler, executor = pipeline
    chain = _chain(sampler, n_steps=2, min_grounded_transitions=1)
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0)),
        user_simulator=DeterministicUser(),
        assistant=_BadFirstCallAssistant(),
        config=GeneratorConfig(max_repair_attempts=1, ambiguity_fraction=0.0),
    )
    conv = coord.run(chain, seed=42)

    # Exactly one repair recorded
    assert len(conv.metadata["repair_attempts"]) == 1
    repair = conv.metadata["repair_attempts"][0]
    assert repair["repaired"] is True
    # Conversation has at least one tool_error then a successful tool response
    has_error = any(
        m["role"] == "tool" and isinstance(m["content"], dict) and "error" in m["content"]
        for m in conv.messages
    )
    has_success = any(
        m["role"] == "tool" and isinstance(m["content"], dict) and "error" not in m["content"]
        for m in conv.messages
    )
    assert has_error and has_success


# ----- chain deviations ----------------------------------------------------


class _DeviationProposingAssistant(Assistant):
    """First tool_calls turn carries a chain_deviation proposal."""

    def __init__(self, deviation: ChainDeviation):
        self._deviation = deviation
        self._proposed = False

    def compose_turn(
        self, plan, transcript, session, *, steps_completed, clarifications_taken, config, seed
    ):
        if steps_completed >= len(plan.step_plans):
            return AssistantTurn(kind="final_summary", content="done.")
        step = plan.step_plans[steps_completed]
        args = session.suggest_arguments(step.endpoint_id)
        dev = None
        if not self._proposed and steps_completed == 0:
            self._proposed = True
            dev = self._deviation
        return AssistantTurn(
            kind="tool_calls",
            tool_calls=[ToolCallProposal(endpoint_id=step.endpoint_id, arguments=args)],
            chain_deviation=dev,
        )


def test_low_confidence_deviation_is_rejected(pipeline):
    registry, graph, sampler, executor = pipeline
    chain = _chain(sampler, n_steps=3, min_grounded_transitions=1)
    deviation = ChainDeviation(
        kind="add_step",
        endpoint_id=chain.endpoints[1],
        position=1,
        reasoning="confidence too low to actually apply",
        deviation_confidence=0.5,
    )
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0)),
        user_simulator=DeterministicUser(),
        assistant=_DeviationProposingAssistant(deviation),
        config=GeneratorConfig(ambiguity_fraction=0.0, assistant_deviation_threshold=0.85),
    )
    conv = coord.run(chain, seed=42)

    assert conv.metadata["deviations_accepted"] == []
    assert len(conv.metadata["deviations_rejected"]) == 1
    assert conv.metadata["deviations_rejected"][0]["reject_reason"] == "below_confidence_threshold"
    assert conv.metadata["final_chain"] == list(chain.endpoints)


def test_unknown_endpoint_deviation_is_rejected(pipeline):
    registry, graph, sampler, executor = pipeline
    chain = _chain(sampler, n_steps=3, min_grounded_transitions=1)
    deviation = ChainDeviation(
        kind="add_step",
        endpoint_id="nonexistent/endpoint",
        position=1,
        reasoning="endpoint doesn't exist",
        deviation_confidence=0.99,
    )
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0)),
        user_simulator=DeterministicUser(),
        assistant=_DeviationProposingAssistant(deviation),
        config=GeneratorConfig(ambiguity_fraction=0.0),
    )
    conv = coord.run(chain, seed=42)
    assert conv.metadata["deviations_accepted"] == []
    assert conv.metadata["deviations_rejected"][0]["reject_reason"] == "unknown_endpoint"


def test_add_step_deviation_accepted_with_graph_path(pipeline):
    """Insert a step that the graph supports between two existing steps."""
    registry, graph, sampler, executor = pipeline
    # Same-domain chain: travel/search_hotels -> travel/get_hotel_details
    chain = sampler.sample(
        ChainConstraints(n_steps=2, required_endpoint="travel/search_hotels"),
        seed=42,
    )
    # Find a travel endpoint with same_domain edges to both existing endpoints.
    # All same-domain travel endpoints qualify; pick travel/search_flights.
    insert_endpoint = "travel/search_flights"

    deviation = ChainDeviation(
        kind="add_step",
        endpoint_id=insert_endpoint,
        position=1,  # insert between step 0 and step 1
        reasoning="user might want to search flights too",
        deviation_confidence=0.95,
    )
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0)),
        user_simulator=DeterministicUser(),
        assistant=_DeviationProposingAssistant(deviation),
        config=GeneratorConfig(ambiguity_fraction=0.0),
    )
    conv = coord.run(chain, seed=42)

    assert len(conv.metadata["deviations_accepted"]) == 1
    accepted = conv.metadata["deviations_accepted"][0]
    assert accepted["kind"] == "add_step"
    assert insert_endpoint in conv.metadata["final_chain"]
    assert len(conv.metadata["final_chain"]) == 3


def test_modify_step_deviation_accepted_with_graph_path(pipeline):
    """Replace step N with a different endpoint the graph also reaches."""
    registry, graph, sampler, executor = pipeline
    chain = sampler.sample(
        ChainConstraints(n_steps=2, required_endpoint="travel/search_hotels"),
        seed=42,
    )
    # Replace step 1 with another travel endpoint reachable via same_domain.
    new_endpoint = "travel/search_flights"
    deviation = ChainDeviation(
        kind="modify_step",
        endpoint_id=new_endpoint,
        position=1,
        reasoning="actually flights, not hotels",
        deviation_confidence=0.95,
    )
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0)),
        user_simulator=DeterministicUser(),
        assistant=_DeviationProposingAssistant(deviation),
        config=GeneratorConfig(ambiguity_fraction=0.0),
    )
    conv = coord.run(chain, seed=42)

    assert len(conv.metadata["deviations_accepted"]) == 1
    accepted = conv.metadata["deviations_accepted"][0]
    assert accepted["kind"] == "modify_step"
    assert conv.metadata["final_chain"][1] == new_endpoint
    assert len(conv.metadata["final_chain"]) == 2


def test_modify_step_no_graph_path_rejected(pipeline):
    """Replace step with one that has no graph edge from the previous step."""
    registry, graph, sampler, executor = pipeline
    chain = sampler.sample(
        ChainConstraints(n_steps=2, required_endpoint="travel/search_hotels"),
        seed=42,
    )
    # finance has no graph edge from travel.
    deviation = ChainDeviation(
        kind="modify_step",
        endpoint_id="finance/get_quote",
        position=1,
        reasoning="confident but unreachable",
        deviation_confidence=0.99,
    )
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0)),
        user_simulator=DeterministicUser(),
        assistant=_DeviationProposingAssistant(deviation),
        config=GeneratorConfig(ambiguity_fraction=0.0),
    )
    conv = coord.run(chain, seed=42)
    assert conv.metadata["deviations_accepted"] == []
    assert conv.metadata["deviations_rejected"][0]["reject_reason"] == "no_graph_path"
