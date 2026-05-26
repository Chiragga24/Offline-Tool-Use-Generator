import pytest

from kg_mle.config import DEFAULT_INPUT_PATH
from kg_mle.graph import build_tool_graph
from kg_mle.registry import enrich_registry, load_registry
from kg_mle.sampler import (
    ChainConstraints,
    SamplingResult,
    ToolChainSampler,
    UnsatisfiableConstraintsError,
)


@pytest.fixture(scope="module")
def sampler() -> ToolChainSampler:
    """Build the sampler once per test module against the real fixture graph.

    Module-scoped to keep test runtime reasonable — graph build is cheap but
    not free, and the sampler itself is stateless across `sample()` calls.
    """
    registry = load_registry(DEFAULT_INPUT_PATH)
    enrich_registry(registry)
    graph = build_tool_graph(registry)
    return ToolChainSampler(graph)


def test_sample_returns_exact_n_steps(sampler: ToolChainSampler):
    result = sampler.sample(ChainConstraints(n_steps=3), seed=42)

    assert isinstance(result, SamplingResult)
    assert len(result.endpoints) == 3
    assert len(result.transitions) == 2
    # Transitions chain consecutive endpoints.
    for i, transition in enumerate(result.transitions):
        assert transition.source == result.endpoints[i]
        assert transition.target == result.endpoints[i + 1]


def test_sample_with_n_steps_range_picks_within_bounds(sampler: ToolChainSampler):
    for seed in range(5):
        result = sampler.sample(ChainConstraints(n_steps=(2, 5)), seed=seed)
        assert 2 <= len(result.endpoints) <= 5
        assert result.metadata["n_steps"] == len(result.endpoints)


def test_sample_is_deterministic_for_same_seed_and_constraints(sampler: ToolChainSampler):
    constraints = ChainConstraints(n_steps=4)
    a = sampler.sample(constraints, seed=123)
    b = sampler.sample(constraints, seed=123)

    assert a.endpoints == b.endpoints
    assert a.transitions == b.transitions


def test_sample_varies_across_seeds(sampler: ToolChainSampler):
    constraints = ChainConstraints(n_steps=4)
    results = {sampler.sample(constraints, seed=seed).endpoints for seed in range(10)}
    assert len(results) > 1, "10 distinct seeds should produce more than one chain shape."


def test_sample_enforces_min_distinct_domains(sampler: ToolChainSampler):
    constraints = ChainConstraints(n_steps=3, min_distinct_domains=2)
    result = sampler.sample(constraints, seed=7)

    domains = {endpoint.split("/")[0] for endpoint in result.endpoints}
    assert len(domains) >= 2


def test_sample_enforces_required_domains(sampler: ToolChainSampler):
    # (gaming, events) is reachable in 3 steps via the
    # gaming/search_games -> events/create_calendar_event grounding edge
    # plus a same-domain hop inside gaming.
    constraints = ChainConstraints(
        n_steps=3,
        required_domains=("gaming", "events"),
        min_distinct_domains=2,
    )
    result = sampler.sample(constraints, seed=11)

    visited = {endpoint.split("/")[0] for endpoint in result.endpoints}
    assert "gaming" in visited
    assert "events" in visited


def test_sample_enforces_min_grounded_transitions(sampler: ToolChainSampler):
    constraints = ChainConstraints(n_steps=3, min_grounded_transitions=2)
    result = sampler.sample(constraints, seed=3)

    grounded = sum(1 for transition in result.transitions if transition.advance_type == "grounded")
    assert grounded >= 2


def test_sample_excludes_semantic_edges_by_default(sampler: ToolChainSampler):
    for seed in range(20):
        result = sampler.sample(ChainConstraints(n_steps=4), seed=seed)
        assert all(transition.advance_type != "semantic" for transition in result.transitions), (
            f"Seed {seed} produced a semantic transition with allow_semantic_edges=False."
        )


def test_sample_starts_at_required_endpoint(sampler: ToolChainSampler):
    constraints = ChainConstraints(n_steps=3, required_endpoint="travel/search_hotels")
    result = sampler.sample(constraints, seed=42)

    assert result.endpoints[0] == "travel/search_hotels"
    assert "travel/search_hotels" in result.endpoints


def test_sample_does_not_revisit_endpoints(sampler: ToolChainSampler):
    constraints = ChainConstraints(n_steps=5, min_distinct_domains=2)
    result = sampler.sample(constraints, seed=99)
    assert len(set(result.endpoints)) == len(result.endpoints)


def test_sample_respects_forbid_endpoint_ids(sampler: ToolChainSampler):
    forbidden = ("travel/search_hotels", "events/create_calendar_event")
    constraints = ChainConstraints(n_steps=4, forbid_endpoint_ids=forbidden)
    result = sampler.sample(constraints, seed=5)

    for endpoint in result.endpoints:
        assert endpoint not in forbidden


def test_sample_metadata_aggregates_corpus_stats(sampler: ToolChainSampler):
    """Metadata is the planner's input — it must carry domains, tools, advance counts."""
    constraints = ChainConstraints(n_steps=3, min_distinct_domains=2)
    result = sampler.sample(constraints, seed=17)

    assert "domains_visited" in result.metadata
    assert "tools_visited" in result.metadata
    assert "advance_type_counts" in result.metadata
    assert "grounded_transitions" in result.metadata
    assert "backtracks" in result.metadata
    assert "start_endpoint" in result.metadata
    assert result.metadata["start_endpoint"] == result.endpoints[0]
    assert result.grounded_transition_count == result.metadata["grounded_transitions"]


def test_sample_raises_when_constraints_unsatisfiable(sampler: ToolChainSampler):
    # 9 domains in the fixture; requiring 10 distinct domains across 10 steps
    # is unsatisfiable.
    constraints = ChainConstraints(n_steps=10, min_distinct_domains=10)
    with pytest.raises(UnsatisfiableConstraintsError):
        sampler.sample(constraints, seed=1)


def test_sample_raises_when_min_grounded_exceeds_possible(sampler: ToolChainSampler):
    # A 2-step chain has only 1 transition; min_grounded_transitions=2 is impossible.
    constraints = ChainConstraints(n_steps=2, min_grounded_transitions=2)
    with pytest.raises(UnsatisfiableConstraintsError):
        sampler.sample(constraints, seed=1)


def test_sample_raises_for_unknown_required_endpoint(sampler: ToolChainSampler):
    constraints = ChainConstraints(n_steps=3, required_endpoint="nonexistent/endpoint")
    with pytest.raises(UnsatisfiableConstraintsError):
        sampler.sample(constraints, seed=1)


def test_sample_uses_graph_not_hardcoded_list(sampler: ToolChainSampler):
    """The assignment's hard requirement: the generator must use the graph sampler.

    This test substitutes the sampler's internal adjacency with an empty mapping
    and confirms the sampler can no longer produce chains — proving it really
    is graph-driven.
    """
    saved = sampler._adjacency
    sampler._adjacency = {}
    try:
        with pytest.raises(UnsatisfiableConstraintsError):
            sampler.sample(ChainConstraints(n_steps=2), seed=1)
    finally:
        sampler._adjacency = saved


def test_sample_records_match_type_for_grounded_transitions(sampler: ToolChainSampler):
    constraints = ChainConstraints(n_steps=3, min_grounded_transitions=1)
    result = sampler.sample(constraints, seed=13)

    grounded_transitions = [t for t in result.transitions if t.advance_type == "grounded"]
    assert grounded_transitions, "Test requires at least one grounded transition."
    for transition in grounded_transitions:
        assert transition.match_type in {"exact_name", "canonical", "alias"}
        assert transition.parameter is not None
        assert transition.source_field is not None
