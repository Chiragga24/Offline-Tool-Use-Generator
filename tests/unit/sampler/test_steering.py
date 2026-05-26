from kg_mle.sampler.constraints import ChainConstraints, SamplingResult, Transition
from kg_mle.sampler.steering import CorpusSteerer, NullSteerer


def _result(
    endpoints: tuple[str, ...],
    transitions: tuple[Transition, ...] = (),
    tools_visited: tuple[str, ...] = (),
) -> SamplingResult:
    return SamplingResult(
        endpoints=endpoints,
        transitions=transitions,
        pattern="sequential",
        seed=0,
        constraints=ChainConstraints(n_steps=len(endpoints)),
        metadata={"tools_visited": list(tools_visited)},
    )


def test_steerer_records_counts_per_chain():
    steerer = CorpusSteerer(target_count=10, endpoint_count=4)
    steerer.record(
        _result(
            endpoints=("travel/a", "events/b"),
            transitions=(Transition(source="travel/a", target="events/b", advance_type="grounded"),),
            tools_visited=("travel_tool", "events_tool"),
        )
    )

    assert steerer.counters.domains["travel"] == 1
    assert steerer.counters.domains["events"] == 1
    assert steerer.counters.endpoints["travel/a"] == 1
    assert steerer.counters.endpoints["events/b"] == 1
    assert steerer.counters.endpoint_pairs[("travel/a", "events/b")] == 1
    assert steerer.counters.tools["travel_tool"] == 1
    assert steerer.counters.chain_count == 1


def test_steerer_flags_overused_endpoint_for_forbid_list():
    steerer = CorpusSteerer(target_count=200, endpoint_count=10, endpoint_overuse_factor=1.5)
    # target/endpoint baseline = 20.0; threshold = ceil(20*1.5) = 30
    assert steerer.endpoint_overuse_threshold == 30

    for _ in range(30):
        steerer.record(_result(endpoints=("travel/a",)))
    steerer.record(_result(endpoints=("events/b",)))

    forbidden = steerer.forbid_endpoints()
    assert "travel/a" in forbidden
    assert "events/b" not in forbidden


def test_steerer_flags_overused_endpoint_pair_for_diversity_diagnostics():
    steerer = CorpusSteerer(target_count=100, endpoint_count=10, pair_overuse_factor=1.2)
    for _ in range(12):
        steerer.record(
            _result(
                endpoints=("travel/a", "events/b"),
                transitions=(
                    Transition(source="travel/a", target="events/b", advance_type="grounded"),
                ),
            )
        )

    assert ("travel/a", "events/b") in steerer.forbid_endpoint_pairs()


def test_counters_summary_exposes_cross_conversation_context_fields():
    steerer = CorpusSteerer(target_count=10, endpoint_count=4)
    steerer.record(
        _result(
            endpoints=("finance/a", "weather/b", "events/c"),
            transitions=(
                Transition(source="finance/a", target="weather/b", advance_type="same_domain"),
                Transition(source="weather/b", target="events/c", advance_type="grounded"),
            ),
            tools_visited=("finance_tool", "weather_tool", "events_tool"),
        )
    )
    summary = steerer.counters.as_summary()

    assert summary["chain_count"] == 1
    assert summary["domain_counts"]["finance"] == 1
    assert summary["tool_counts"]["weather_tool"] == 1
    assert summary["endpoint_counts"]["events/c"] == 1
    assert summary["endpoint_pair_counts"]["weather/b->events/c"] == 1
    assert summary["chain_length_counts"]["3"] == 1
    assert summary["domain_pattern_counts"]["finance->weather->events"] == 1


def test_steerer_threshold_has_hard_floor_for_small_corpora():
    """Small target_count should not let the threshold drop below 3, otherwise
    the forbid list fills up so fast the planner runs out of options."""
    steerer = CorpusSteerer(target_count=10, endpoint_count=45)
    assert steerer.endpoint_overuse_threshold >= 3


def test_steerer_least_used_domains_returns_zero_count_first():
    steerer = CorpusSteerer(target_count=4, endpoint_count=4)
    steerer.record(_result(endpoints=("travel/a",)))
    steerer.record(_result(endpoints=("travel/b",)))
    steerer.record(_result(endpoints=("events/c",)))

    picked = steerer.least_used_domains(
        ("travel", "events", "finance", "weather"),
        k=2,
    )
    # finance and weather have zero count; ties broken alphabetically.
    assert picked == ("finance", "weather")


def test_steerer_least_used_domains_with_ties_breaks_by_name():
    steerer = CorpusSteerer(target_count=4, endpoint_count=4)
    steerer.record(_result(endpoints=("travel/a",)))
    steerer.record(_result(endpoints=("events/b",)))

    picked = steerer.least_used_domains(("travel", "events", "finance"), k=1)
    assert picked == ("finance",)


def test_steerer_domain_pattern_collapses_consecutive_same_domain():
    """A chain that stays inside one domain should yield a 1-element pattern."""
    steerer = CorpusSteerer(target_count=2, endpoint_count=2)
    steerer.record(_result(endpoints=("travel/a", "travel/b", "travel/c")))

    assert steerer.counters.domain_patterns[("travel",)] == 1


def test_steerer_domain_pattern_captures_cross_domain_sequence():
    steerer = CorpusSteerer(target_count=2, endpoint_count=2)
    steerer.record(
        _result(endpoints=("entertainment/a", "events/b", "events/c"))
    )

    assert steerer.counters.domain_patterns[("entertainment", "events")] == 1


def test_null_steerer_records_counters_but_never_returns_penalties():
    steerer = NullSteerer()
    for _ in range(5):
        steerer.record(_result(endpoints=("travel/a",)))

    assert steerer.counters.endpoints["travel/a"] == 5
    assert steerer.forbid_endpoints() == ()
    assert steerer.forbid_endpoint_pairs() == ()


def test_null_steerer_least_used_returns_alphabetical_for_reproducibility():
    steerer = NullSteerer()
    picked = steerer.least_used_domains(("travel", "events", "finance"), k=2)
    assert picked == ("events", "finance")


def test_counters_summary_is_json_serialisable():
    import json

    steerer = CorpusSteerer(target_count=4, endpoint_count=4)
    steerer.record(
        _result(
            endpoints=("travel/a", "events/b"),
            transitions=(Transition(source="travel/a", target="events/b", advance_type="grounded"),),
            tools_visited=("t1",),
        )
    )
    summary = steerer.counters.as_summary()
    json.dumps(summary)  # must not raise
    assert summary["chain_count"] == 1
    assert summary["endpoint_pair_counts"]["travel/a->events/b"] == 1
