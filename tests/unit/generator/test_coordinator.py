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
from kg_mle.generator.agents import Assistant, UserSimulator
from kg_mle.generator.coordinator import _extract_clarified_value
from kg_mle.generator.protocol import AssistantTurn, ToolCallProposal, UserTurn
from kg_mle.graph import build_tool_graph
from kg_mle.registry import enrich_registry, load_registry
from kg_mle.sampler import ChainConstraints, SamplingResult, ToolChainSampler, Transition


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
    assert md["tools_visited"] == list(chain.metadata["tools_visited"])
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


class _HallucinatedArgsAssistant(Assistant):
    def compose_turn(
        self, plan, transcript, session, *, steps_completed, clarifications_taken, config, seed
    ):
        if steps_completed >= len(plan.step_plans):
            return AssistantTurn(kind="final_summary", content="done.")
        step = plan.step_plans[steps_completed]
        args = session.suggest_arguments(step.endpoint_id)
        if step.endpoint_id == "events/create_calendar_event":
            args.update(
                {
                    "title": "League of Legends Tournament",
                    "location": "LoL Park, Seoul",
                    "event_type": "tournament",
                    "hallucinated_key": "not a real parameter",
                }
            )
        return AssistantTurn(
            kind="tool_calls",
            tool_calls=[ToolCallProposal(endpoint_id=step.endpoint_id, arguments=args)],
        )


def test_coordinator_sanitizes_llm_hallucinated_locality_args(pipeline):
    registry, graph, _, executor = pipeline
    chain = SamplingResult(
        endpoints=("gaming/get_tournament_schedule", "events/create_calendar_event"),
        transitions=(
            Transition(
                source="gaming/get_tournament_schedule",
                target="events/create_calendar_event",
                advance_type="grounded",
                parameter="start_time",
                source_field="start_time",
                match_type="canonical",
            ),
        ),
        pattern="sequential",
        seed=99,
        constraints=ChainConstraints(n_steps=2, min_grounded_transitions=1),
        metadata={"tools_visited": ["gaming_catalog", "events_calendar"]},
    )
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0)),
        user_simulator=DeterministicUser(),
        assistant=_HallucinatedArgsAssistant(),
        config=GeneratorConfig(ambiguity_fraction=0.0),
    )

    conv = coord.run(chain, seed=99)

    first_response = next(
        m["content"]
        for m in conv.messages
        if m["role"] == "tool" and m["endpoint"] == "gaming/get_tournament_schedule"
    )
    calendar_call = next(
        m["tool_calls"][0]
        for m in conv.messages
        if m.get("tool_calls")
        and m["tool_calls"][0]["endpoint_id"] == "events/create_calendar_event"
    )
    # Grounded + locality fields are reconciled to the prior tool output.
    assert calendar_call["arguments"]["start_time"] == first_response["start_time"]
    assert calendar_call["arguments"]["location"] == first_response["venue"]
    assert calendar_call["arguments"]["location"] != "LoL Park, Seoul"
    # ...but the LLM's type-valid free-text title (no grounding constraint) is
    # PRESERVED, not reset to a deterministic placeholder. This is the
    # selective-sanitizer guarantee: keep what fits, override only provenance.
    assert calendar_call["arguments"]["title"] == "League of Legends Tournament"
    assert calendar_call["arguments"]["event_type"] == "tournament"
    # Undeclared (hallucinated) keys are dropped so they never reach the dataset.
    assert "hallucinated_key" not in calendar_call["arguments"]


class _FakeStateForContext:
    def __init__(self, issued):
        self._issued = issued
        self.log = []

    def issued_ids(self, key):
        return tuple(self._issued.get(key, ()))


class _FakeSessionForContext:
    def __init__(self, issued):
        self.state = _FakeStateForContext(issued)


def test_contextual_value_does_not_bridge_venue_into_city():
    from kg_mle.generator.coordinator import _contextual_value_for_param

    session = _FakeSessionForContext({"venue": ("Midtown",), "location": ("Midtown",)})
    # `location` still pulls the prior venue/location — coherence preserved.
    assert (
        _contextual_value_for_param("location", None, session=session, plan_value=None, user_clarified=False)
        == "Midtown"
    )
    # `city` must NOT inherit a venue/location string; it keeps its clean plan value.
    assert (
        _contextual_value_for_param("city", None, session=session, plan_value="Las Vegas", user_clarified=False)
        is None
    )


def test_contextual_value_bridges_venue_to_location_without_enrichment():
    from kg_mle.generator.coordinator import _contextual_value_for_param

    # Only `venue` was issued (no `location` key) — the cluster still bridges.
    session = _FakeSessionForContext({"venue": ("Old Town",)})
    assert (
        _contextual_value_for_param("location", None, session=session, plan_value=None, user_clarified=False)
        == "Old Town"
    )


def test_extract_clarified_value_strips_param_prefix_and_filler():
    # Deterministic template.
    assert _extract_clarified_value("For city, use Paris.", "city") == "Paris"
    # LLM phrasing that previously leaked "the game_id" into the value.
    assert _extract_clarified_value("Use the game_id valorant.", "game_id") == "valorant"
    assert _extract_clarified_value("game_id is valorant", "game_id") == "valorant"
    # Bare value.
    assert _extract_clarified_value("valorant", "game_id") == "valorant"
    # Trailing filler that restates the parameter.
    assert (
        _extract_clarified_value("Use Las Vegas Convention Center as the location.", "location")
        == "Las Vegas Convention Center"
    )
    assert _extract_clarified_value("Paris for the city", "city") == "Paris"


class _LLMStyleUser(UserSimulator):
    """Mimics an LLM user: vague prose in `content`, exact value in the
    structured `clarified_value` field. Prose and value deliberately disagree
    so the test proves the coordinator prefers the structured field."""

    def initial_request(self, plan, *, seed):
        return UserTurn(content="Help me with this, please.", is_initial_request=True)

    def reply_to_clarification(self, plan, *, target_step, target_parameter, seed):
        return UserTurn(
            content="Hmm, whatever you think is best honestly.",  # no parseable value
            is_clarification_reply=True,
            clarified_value="SENTINEL_VALUE",
        )


def test_clarification_prefers_structured_value_over_prose(pipeline):
    registry, graph, sampler, executor = pipeline
    chain = _chain(sampler, n_steps=2, min_grounded_transitions=1)
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        # Force a clarification on step 0.
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=1.0)),
        user_simulator=_LLMStyleUser(),
        assistant=DeterministicAssistant(),
        config=GeneratorConfig(ambiguity_fraction=1.0),
    )

    conv = coord.run(chain, seed=42)

    clarified_param = next(
        (
            m["clarification_target_parameter"]
            for m in conv.messages
            if m["role"] == "assistant" and m.get("clarification_target_parameter")
        ),
        None,
    )
    if clarified_param is None:
        pytest.skip("No clarification fired for this chain/seed.")

    # The tool call for the clarified step must use the STRUCTURED value, not
    # the vague prose. If the coordinator had parsed `content`, the arg would be
    # the garbage sentence instead.
    tool_call_args = next(
        m["tool_calls"][0]["arguments"]
        for m in conv.messages
        if m.get("tool_calls") and clarified_param in m["tool_calls"][0]["arguments"]
    )
    assert tool_call_args[clarified_param] == "SENTINEL_VALUE"


class _LaterStepAmbiguousPlanner(DeterministicPlanner):
    """Marks a non-first step ambiguous, like an LLM planner can — the case
    that previously made a stalling assistant truncate the chain."""

    def plan(self, sampling_result, *, seed):
        base = super().plan(sampling_result, seed=seed)
        if len(base.step_plans) > 1 and base.step_plans[1].parameter_plans:
            base.ambiguous_step_indices = [1]
            base.step_plans[1].parameter_plans[0].ambiguous = True
            base.step_plans[1].parameter_plans[0].confidence = 0.3
        return base


class _StallAfterFirstAssistant(Assistant):
    """Real tool call at step 0, then always claims tool_calls with none —
    mimics an LLM that stops making progress after the first step."""

    def compose_turn(
        self, plan, transcript, session, *, steps_completed, clarifications_taken, config, seed
    ):
        if steps_completed == 0:
            step = plan.step_plans[0]
            return AssistantTurn(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        endpoint_id=step.endpoint_id,
                        arguments=session.suggest_arguments(step.endpoint_id),
                    )
                ],
            )
        return AssistantTurn(kind="tool_calls", tool_calls=[])


def test_coordinator_completes_chain_when_llm_stalls(pipeline):
    """A stalling LLM (empty tool_calls) with a later ambiguous step must still
    yield a complete, grounded conversation ending in a final summary — not a
    truncated trace that silently reports success."""
    registry, graph, sampler, executor = pipeline
    chain = _chain(sampler, n_steps=3, min_grounded_transitions=1)
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=_LaterStepAmbiguousPlanner(registry, config=GeneratorConfig()),
        user_simulator=DeterministicUser(),
        assistant=_StallAfterFirstAssistant(),
        config=GeneratorConfig(),
    )

    conv = coord.run(chain, seed=42)

    successful = [
        m
        for m in conv.messages
        if m["role"] == "tool"
        and not (isinstance(m["content"], dict) and "error" in m["content"])
    ]
    assert len(successful) == len(chain.endpoints), (
        f"chain truncated: {len(successful)}/{len(chain.endpoints)} steps completed"
    )
    # Ends with a final assistant summary, not a dangling tool response.
    last = conv.messages[-1]
    assert last["role"] == "assistant" and last.get("content") and not last.get("tool_calls")
    guarantee = conv.metadata["completion_guarantee"]
    assert guarantee["triggered"] is True
    assert guarantee["reason"] == "empty_tool_calls"
    assert guarantee["llm_reprompt_attempts"] >= 2
    assert guarantee["llm_reprompt_succeeded"] is False
    assert guarantee["deterministic_turns"]


class _RecoverAfterDirectiveAssistant(Assistant):
    """Stalls once, then follows the internal retry directive with an LLM-authored
    tool call. The directive must not be written into the public transcript."""

    def compose_turn(
        self, plan, transcript, session, *, steps_completed, clarifications_taken, config, seed
    ):
        if steps_completed == 0:
            step = plan.step_plans[0]
            return AssistantTurn(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        endpoint_id=step.endpoint_id,
                        arguments=session.suggest_arguments(step.endpoint_id),
                    )
                ],
            )
        if transcript and "Coordinator internal retry directive" in str(transcript[-1].get("content")):
            step = plan.step_plans[steps_completed]
            return AssistantTurn(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        endpoint_id=step.endpoint_id,
                        arguments=session.suggest_arguments(step.endpoint_id),
                    )
                ],
            )
        return AssistantTurn(kind="tool_calls", tool_calls=[])


def test_stall_reprompt_recovers_before_deterministic_fallback(pipeline):
    registry, graph, sampler, executor = pipeline
    chain = _chain(sampler, n_steps=2, min_grounded_transitions=1)
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0)),
        user_simulator=DeterministicUser(),
        assistant=_RecoverAfterDirectiveAssistant(),
        config=GeneratorConfig(ambiguity_fraction=0.0, max_stall_reprompts=2),
    )

    conv = coord.run(chain, seed=42)

    successful = [
        m
        for m in conv.messages
        if m["role"] == "tool"
        and not (isinstance(m["content"], dict) and "error" in m["content"])
    ]
    assert len(successful) == 2
    guarantee = conv.metadata["completion_guarantee"]
    assert guarantee["triggered"] is True
    assert guarantee["reason"] == "empty_tool_calls"
    assert guarantee["llm_reprompt_attempts"] == 1
    assert guarantee["llm_reprompt_succeeded"] is True
    # Only the deterministic final summary should be recorded, not the recovered tool call.
    assert all(turn.get("endpoint_id") != chain.endpoints[1] for turn in guarantee["deterministic_turns"])
    assert not any(
        "Coordinator internal retry directive" in str(message.get("content"))
        for message in conv.messages
    )


class _PrematureFinalAssistant(Assistant):
    def compose_turn(
        self, plan, transcript, session, *, steps_completed, clarifications_taken, config, seed
    ):
        if steps_completed == 0:
            step = plan.step_plans[0]
            return AssistantTurn(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        endpoint_id=step.endpoint_id,
                        arguments=session.suggest_arguments(step.endpoint_id),
                    )
                ],
            )
        return AssistantTurn(kind="final_summary", content="done too early.")


def test_coordinator_rejects_premature_final_summary(pipeline):
    registry, graph, sampler, executor = pipeline
    chain = _chain(sampler, n_steps=2, min_grounded_transitions=1)
    coord = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0)),
        user_simulator=DeterministicUser(),
        assistant=_PrematureFinalAssistant(),
        config=GeneratorConfig(ambiguity_fraction=0.0),
    )

    conv = coord.run(chain, seed=42)

    successful = [
        m
        for m in conv.messages
        if m["role"] == "tool" and not (isinstance(m["content"], dict) and "error" in m["content"])
    ]
    assert len(successful) == 2
    assert conv.messages[-1]["role"] == "assistant"
    guarantee = conv.metadata["completion_guarantee"]
    assert guarantee["triggered"] is True
    assert guarantee["reason"] == "premature_final_summary"
    assert guarantee["deterministic_turns"]


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
