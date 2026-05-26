"""Tests for ExecutorSession — the integration surface the generator will use.

Each test maps to one of the design decisions we made before implementing:

  1. Deterministic mocks (no LLM polish by default): same seed produces
     byte-identical responses.
  2. suggest_arguments returns a plausible default; the generator's
     overrides go through the same validator.
  3. Failures raise typed exceptions AND appear in the conversation log
     as `role: "tool"` error entries.
  4. Grounding is strict: hallucinated IDs are rejected.
"""

import pytest

from kg_mle.config import DEFAULT_INPUT_PATH
from kg_mle.executor import (
    ExecutorSession,
    MissingRequiredArgumentError,
    OfflineExecutor,
    UngroundedArgumentError,
    ArgumentTypeError,
)
from kg_mle.graph import build_tool_graph
from kg_mle.registry import enrich_registry, load_registry
from kg_mle.sampler import ChainConstraints, SamplingResult, ToolChainSampler


@pytest.fixture(scope="module")
def registry_and_sampler():
    registry = load_registry(DEFAULT_INPUT_PATH)
    enrich_registry(registry)
    graph = build_tool_graph(registry)
    sampler = ToolChainSampler(graph)
    return registry, sampler


def _two_step_grounded_chain(sampler: ToolChainSampler) -> SamplingResult:
    """Find a 2-step chain with a grounded transition. The fixture has many;
    we just take the first the sampler offers with that constraint."""
    return sampler.sample(
        ChainConstraints(n_steps=2, min_grounded_transitions=1),
        seed=42,
    )


def _open_session(registry, sampler, seed=42) -> tuple[ExecutorSession, SamplingResult]:
    result = _two_step_grounded_chain(sampler)
    executor = OfflineExecutor(registry)
    session = executor.open_session(result, seed=seed)
    return session, result


# ---------- 1. Determinism ------------------------------------------------


def test_mock_responses_are_deterministic_for_same_seed(registry_and_sampler):
    registry, sampler = registry_and_sampler
    session_a, chain = _open_session(registry, sampler, seed=99)
    session_b, chain_b = _open_session(registry, sampler, seed=99)
    assert chain.endpoints == chain_b.endpoints

    response_a = session_a.call(chain.endpoints[0], session_a.suggest_arguments(chain.endpoints[0]))
    response_b = session_b.call(chain.endpoints[0], session_b.suggest_arguments(chain.endpoints[0]))

    assert response_a == response_b


def test_mock_responses_differ_across_seeds(registry_and_sampler):
    registry, sampler = registry_and_sampler
    session_a, chain = _open_session(registry, sampler, seed=99)
    session_b, _ = _open_session(registry, sampler, seed=100)

    response_a = session_a.call(chain.endpoints[0], session_a.suggest_arguments(chain.endpoints[0]))
    response_b = session_b.call(chain.endpoints[0], session_b.suggest_arguments(chain.endpoints[0]))

    assert response_a != response_b, "Different seeds should produce different mocks."


# ---------- 2. suggest_arguments + example_values -------------------------


def test_suggest_arguments_returns_plausible_defaults_for_first_step(registry_and_sampler):
    registry, sampler = registry_and_sampler
    session, chain = _open_session(registry, sampler)
    first = chain.endpoints[0]

    args = session.suggest_arguments(first)
    endpoint = session.endpoint_index[first]
    required = {p.name for p in endpoint.parameters if p.required}

    assert required <= set(args), f"Missing required: {required - set(args)}"
    # All suggested values must satisfy the validator (round-trip via call).
    response = session.call(first, args)
    assert isinstance(response, dict)


def test_suggest_arguments_uses_issued_ids_for_grounded_parameters(registry_and_sampler):
    """Step 2's suggested args should reference IDs that step 1 actually issued."""
    registry, sampler = registry_and_sampler
    session, chain = _open_session(registry, sampler)
    if len(chain.transitions) < 1 or chain.transitions[0].advance_type != "grounded":
        pytest.skip("Selected chain has no grounded transition to exercise.")

    session.call(chain.endpoints[0], session.suggest_arguments(chain.endpoints[0]))
    suggested = session.suggest_arguments(chain.endpoints[1])

    transition = chain.transitions[0]
    if transition.parameter and transition.parameter in suggested:
        issued = session.state.issued_ids(transition.source_field or transition.parameter)
        assert suggested[transition.parameter] in issued


def test_example_values_returns_issued_ids_for_grounded_params(registry_and_sampler):
    registry, sampler = registry_and_sampler
    session, chain = _open_session(registry, sampler)
    if len(chain.transitions) < 1 or chain.transitions[0].advance_type != "grounded":
        pytest.skip("Selected chain has no grounded transition to exercise.")

    session.call(chain.endpoints[0], session.suggest_arguments(chain.endpoints[0]))
    examples = session.example_values(chain.endpoints[1])
    transition = chain.transitions[0]
    if transition.parameter and transition.parameter in examples:
        issued = session.state.issued_ids(transition.source_field or transition.parameter)
        assert all(value in issued for value in examples[transition.parameter])
        assert examples[transition.parameter]  # not empty


def test_example_values_returns_canonical_pool_for_free_params(registry_and_sampler):
    registry, sampler = registry_and_sampler
    session, _ = _open_session(registry, sampler)
    # finance/search_symbol takes a query parameter — canonical pool exists for "query".
    endpoint_id = "finance/search_symbol"
    if endpoint_id not in session.endpoint_index:
        pytest.skip("finance/search_symbol not in fixture.")
    examples = session.example_values(endpoint_id)
    assert examples.get("query"), "Expected example pool for canonical 'query' parameter."


def test_generator_override_goes_through_same_validator(registry_and_sampler):
    """Overriding suggested args is allowed but every override is re-validated."""
    registry, sampler = registry_and_sampler
    session, chain = _open_session(registry, sampler)
    args = session.suggest_arguments(chain.endpoints[0])
    endpoint = session.endpoint_index[chain.endpoints[0]]
    # Replace a required string param with the wrong type.
    string_params = [p for p in endpoint.parameters if p.required and p.type == "string"]
    if not string_params:
        pytest.skip("First endpoint has no required string parameter to corrupt.")
    args[string_params[0].name] = 12345  # wrong type
    with pytest.raises(ArgumentTypeError):
        session.call(chain.endpoints[0], args)


# ---------- 3. Failures appear in the conversation log --------------------


def test_failure_records_tool_error_entry_in_log(registry_and_sampler):
    registry, sampler = registry_and_sampler
    session, chain = _open_session(registry, sampler)
    endpoint = session.endpoint_index[chain.endpoints[0]]
    required = [p for p in endpoint.parameters if p.required]
    if not required:
        pytest.skip("First endpoint has no required parameter to omit.")

    with pytest.raises(MissingRequiredArgumentError):
        session.call(chain.endpoints[0], {})

    log = session.state.as_log()
    # There should be at least one assistant tool_call and one tool error entry.
    tool_errors = [entry for entry in log if entry["role"] == "tool" and "error" in entry["content"]]
    assert tool_errors, "Failure should appear as a role: tool error entry."
    error_payload = tool_errors[0]["content"]["error"]
    assert error_payload["kind"] == "missing_required_argument"
    assert error_payload["parameter"] == required[0].name


def test_repair_flow_visible_in_log(registry_and_sampler):
    """Reviewer-facing trace: bad call, error entry, recovered call, response entry."""
    registry, sampler = registry_and_sampler
    session, chain = _open_session(registry, sampler)
    endpoint = session.endpoint_index[chain.endpoints[0]]
    required = [p for p in endpoint.parameters if p.required]
    if not required:
        pytest.skip("First endpoint has no required parameter to omit.")

    with pytest.raises(MissingRequiredArgumentError):
        session.call(chain.endpoints[0], {})
    session.call(chain.endpoints[0], session.suggest_arguments(chain.endpoints[0]))

    log = session.state.as_log()
    kinds = [
        "error" if entry["role"] == "tool" and "error" in entry["content"]
        else entry["role"]
        for entry in log
    ]
    # Expect: assistant (bad call) -> error -> assistant (good call) -> tool (response).
    assert kinds == ["assistant", "error", "assistant", "tool"], f"Unexpected log: {kinds}"


# ---------- 4. Strict grounding -------------------------------------------


def test_hallucinated_grounded_id_is_rejected(registry_and_sampler):
    """The whole point of the offline executor: hallucinated IDs fail."""
    registry, sampler = registry_and_sampler
    session, chain = _open_session(registry, sampler)
    if len(chain.transitions) < 1 or chain.transitions[0].advance_type != "grounded":
        pytest.skip("Selected chain has no grounded transition to exercise.")
    transition = chain.transitions[0]
    if not transition.parameter:
        pytest.skip("Grounded transition has no parameter recorded.")

    session.call(chain.endpoints[0], session.suggest_arguments(chain.endpoints[0]))
    bad_args = session.suggest_arguments(chain.endpoints[1])
    bad_args[transition.parameter] = "fake_hallucinated_id"

    with pytest.raises(UngroundedArgumentError) as exc_info:
        session.call(chain.endpoints[1], bad_args)
    assert exc_info.value.details["parameter"] == transition.parameter
    assert "fake_hallucinated_id" in exc_info.value.details["value"]
    assert exc_info.value.details["expected_one_of"]


def test_grounded_id_from_previous_step_validates(registry_and_sampler):
    registry, sampler = registry_and_sampler
    session, chain = _open_session(registry, sampler)
    if len(chain.transitions) < 1 or chain.transitions[0].advance_type != "grounded":
        pytest.skip("Selected chain has no grounded transition to exercise.")

    response_one = session.call(chain.endpoints[0], session.suggest_arguments(chain.endpoints[0]))
    response_two = session.call(chain.endpoints[1], session.suggest_arguments(chain.endpoints[1]))

    transition = chain.transitions[0]
    if transition.source_field and transition.source_field in response_one:
        issued_value = response_one[transition.source_field]
        # The previous-step's issued ID was carried into the second call's args
        # via suggest_arguments.
        assert session.state.issued_ids(transition.source_field) and (
            issued_value in session.state.issued_ids(transition.source_field)
        )
    assert isinstance(response_two, dict)
