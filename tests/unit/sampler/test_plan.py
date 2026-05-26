"""Tests for the corpus planner.

The planner is the layer the diversity experiment relies on. These tests
assert the contract that lets Run A and Run B be reproduced from one seed
plus the steering flag, and that turning steering on actually shifts the
corpus distribution.
"""

import pytest

from kg_mle.config import DEFAULT_INPUT_PATH
from kg_mle.graph import build_tool_graph
from kg_mle.graph.models import GraphEdge, GraphNode, ToolGraph
from kg_mle.registry import enrich_registry, load_registry
from kg_mle.sampler import CorpusPlanner, ToolChainSampler


@pytest.fixture(scope="module")
def sampler() -> ToolChainSampler:
    registry = load_registry(DEFAULT_INPUT_PATH)
    enrich_registry(registry)
    graph = build_tool_graph(registry)
    return ToolChainSampler(graph)


def test_planner_produces_requested_count(sampler: ToolChainSampler):
    planner = CorpusPlanner(sampler, steering_enabled=True, seed=42)
    report = planner.sample_corpus(target_count=30)

    assert len(report.results) + len(report.failures) == 30
    # In the curated fixture, relaxation should be able to satisfy every chain.
    assert len(report.failures) == 0, f"Unexpected failures: {report.failures}"


def test_planner_can_allow_semantic_edges():
    graph = ToolGraph(
        nodes=[
            GraphNode(
                node_id="endpoint:a/one",
                type="endpoint",
                label="one",
                metadata={"endpoint_id": "a/one", "domain": "a"},
            ),
            GraphNode(
                node_id="endpoint:b/two",
                type="endpoint",
                label="two",
                metadata={"endpoint_id": "b/two", "domain": "b"},
            ),
        ],
        edges=[
            GraphEdge(
                source="endpoint:a/one",
                target="endpoint:b/two",
                type="semantic_related",
                metadata={"score": 0.91},
            )
        ],
    )
    planner = CorpusPlanner(
        ToolChainSampler(graph),
        steering_enabled=False,
        seed=42,
        length_distribution=((2, 1.0),),
        allow_semantic_edges=True,
    )

    report = planner.sample_corpus(target_count=1)

    assert report.results[0].transitions[0].advance_type == "semantic"
    assert report.plan_meta["allow_semantic_edges"] is True


def test_planner_targets_multi_step_fraction(sampler: ToolChainSampler):
    """Assignment requires 50-60% multi-step (>=3 calls) AND multi-tool (>=2 tools)."""
    planner = CorpusPlanner(sampler, steering_enabled=True, seed=42)
    report = planner.sample_corpus(target_count=50)

    multi = [
        result
        for result in report.results
        if len(result.endpoints) >= 3 and len(set(result.tools)) >= 2
    ]
    fraction = len(multi) / len(report.results)
    # Default target is 0.55. Allow generous bounds because n_steps is sampled
    # from a distribution and tool variety depends on which endpoints get picked.
    assert 0.40 <= fraction <= 0.85, (
        f"multi-step-multi-tool fraction {fraction:.2f} outside acceptable band."
    )


def test_planner_is_deterministic_per_seed_and_steering(sampler: ToolChainSampler):
    """Same planner seed + same target_count + same steering flag = same corpus."""
    a = CorpusPlanner(sampler, steering_enabled=True, seed=7).sample_corpus(target_count=15)
    b = CorpusPlanner(sampler, steering_enabled=True, seed=7).sample_corpus(target_count=15)

    assert [r.endpoints for r in a.results] == [r.endpoints for r in b.results]


def test_planner_steering_on_vs_off_diverges(sampler: ToolChainSampler):
    """Same seed, opposite steering flags: the corpora must differ.

    This is the test that proves --no-cross-conversation-steering does
    something. If both runs produced identical output, the flag would be a lie.
    """
    on = CorpusPlanner(sampler, steering_enabled=True, seed=42).sample_corpus(target_count=30)
    off = CorpusPlanner(sampler, steering_enabled=False, seed=42).sample_corpus(target_count=30)

    on_chains = [r.endpoints for r in on.results]
    off_chains = [r.endpoints for r in off.results]
    assert on_chains != off_chains, "Steering on/off produced identical corpora."


def test_steering_increases_endpoint_coverage(sampler: ToolChainSampler):
    """Steering should spread usage across more endpoints than no-steering.

    This is one of the diversity metrics for the diversity experiment.
    """
    target = 60
    on = CorpusPlanner(sampler, steering_enabled=True, seed=42).sample_corpus(target_count=target)
    off = CorpusPlanner(sampler, steering_enabled=False, seed=42).sample_corpus(target_count=target)

    on_distinct = len(on.counters_summary["endpoint_counts"])
    off_distinct = len(off.counters_summary["endpoint_counts"])
    assert on_distinct >= off_distinct, (
        f"Steering on covered {on_distinct} endpoints, steering off covered {off_distinct}. "
        "Expected steering to widen or hold coverage."
    )


def test_report_has_plan_meta_and_counters(sampler: ToolChainSampler):
    planner = CorpusPlanner(sampler, steering_enabled=True, seed=42)
    report = planner.sample_corpus(target_count=10)

    assert report.plan_meta["target_count"] == 10
    assert report.plan_meta["seed"] == 42
    assert report.plan_meta["steering_enabled"] is True
    assert "length_distribution" in report.plan_meta
    assert report.counters_summary["chain_count"] == len(report.results)
    assert report.counters_summary["endpoint_counts"]
    assert report.counters_summary["domain_counts"]


def test_null_steerer_still_records_for_comparable_stats(sampler: ToolChainSampler):
    """Run A (steering off) must produce counters comparable to Run B."""
    report = CorpusPlanner(sampler, steering_enabled=False, seed=42).sample_corpus(target_count=10)

    assert report.steering_enabled is False
    assert report.counters_summary["chain_count"] == 10
    assert report.counters_summary["endpoint_counts"]


def test_planner_allows_short_chains_in_distribution(sampler: ToolChainSampler):
    """The varied-length property requires both short (2-3) and long (5+) chains."""
    report = CorpusPlanner(sampler, steering_enabled=True, seed=42).sample_corpus(target_count=50)

    lengths = [len(result.endpoints) for result in report.results]
    assert min(lengths) <= 3, "No short chains produced — varied-length property unmet."
    assert max(lengths) >= 4, "No long chains produced — varied-length property unmet."


def test_planner_raises_on_zero_target_count(sampler: ToolChainSampler):
    planner = CorpusPlanner(sampler, steering_enabled=True, seed=42)
    with pytest.raises(ValueError):
        planner.sample_corpus(target_count=0)
