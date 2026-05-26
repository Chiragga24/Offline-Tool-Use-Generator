from kg_mle.diversity import compute_diversity_metrics


def test_compute_diversity_metrics_counts_domains_pairs_and_patterns():
    conversations = [
        {
            "metadata": {
                "final_chain": ["finance/search_symbol", "finance/get_quote", "events/create_event"],
                "transition_summary": [
                    {"source": "finance/search_symbol", "target": "finance/get_quote"},
                    {"source": "finance/get_quote", "target": "events/create_event"},
                ],
                "tools_visited": ["finance_tool", "events_tool"],
            }
        },
        {
            "metadata": {
                "final_chain": ["weather/get_forecast", "events/create_event"],
                "transition_summary": [
                    {"source": "weather/get_forecast", "target": "events/create_event"}
                ],
                "tools_visited": ["weather_tool", "events_tool"],
            }
        },
    ]

    metrics = compute_diversity_metrics(
        conversations,
        total_endpoints=10,
        total_tools=5,
        endpoint_to_tool={
            "finance/search_symbol": "finance_tool",
            "finance/get_quote": "finance_tool",
            "events/create_event": "events_tool",
            "weather/get_forecast": "weather_tool",
        },
    )

    assert metrics["conversation_count"] == 2
    assert metrics["endpoint_coverage_ratio"] == 0.4
    assert metrics["tool_coverage_ratio"] == 0.6
    assert metrics["distinct_endpoint_pair_ratio"] == 1.0
    assert metrics["domain_pattern_diversity"] == 2
    assert metrics["chain_length_distribution"] == {"2": 1, "3": 1}
    assert metrics["endpoint_pair_counts"]["finance/get_quote->events/create_event"] == 1
