"""Diversity metrics for generated conversation corpora."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def compute_diversity_metrics(
    conversations: list[dict[str, Any]],
    *,
    total_endpoints: int,
    total_tools: int,
    endpoint_to_tool: dict[str, str] | None = None,
) -> dict[str, Any]:
    domains: Counter[str] = Counter()
    endpoints: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    endpoint_pairs: Counter[tuple[str, str]] = Counter()
    chain_lengths: Counter[int] = Counter()
    domain_patterns: Counter[tuple[str, ...]] = Counter()

    for conversation in conversations:
        metadata = conversation.get("metadata", {}) if isinstance(conversation, dict) else {}
        chain = list(metadata.get("final_chain") or metadata.get("original_chain") or [])
        chain_lengths[len(chain)] += 1
        domain_sequence: list[str] = []
        for endpoint_id in chain:
            endpoint = str(endpoint_id)
            domain = endpoint.split("/", 1)[0]
            endpoints[endpoint] += 1
            domains[domain] += 1
            if endpoint_to_tool and endpoint in endpoint_to_tool:
                tools[endpoint_to_tool[endpoint]] += 1
            if not domain_sequence or domain_sequence[-1] != domain:
                domain_sequence.append(domain)
        if domain_sequence:
            domain_patterns[tuple(domain_sequence)] += 1

        transition_summary = metadata.get("transition_summary") or []
        for transition in transition_summary:
            source = transition.get("source")
            target = transition.get("target")
            if source and target:
                endpoint_pairs[(str(source), str(target))] += 1

        if not endpoint_to_tool:
            for tool in metadata.get("tools_visited", ()) or ():
                tools[str(tool)] += 1

    total_transitions = sum(endpoint_pairs.values())
    return {
        "conversation_count": len(conversations),
        "domain_entropy": round(_entropy(domains), 4),
        "endpoint_coverage_ratio": round(_ratio(len(endpoints), total_endpoints), 4),
        "tool_coverage_ratio": round(_ratio(len(tools), total_tools), 4),
        "distinct_endpoint_pair_ratio": round(_ratio(len(endpoint_pairs), total_transitions), 4),
        "domain_pattern_diversity": len(domain_patterns),
        "top_endpoint_share": round(
            _ratio(max(endpoints.values(), default=0), sum(endpoints.values())), 4
        ),
        "chain_length_distribution": {str(k): v for k, v in sorted(chain_lengths.items())},
        "domain_counts": dict(domains),
        "endpoint_counts": dict(endpoints),
        "endpoint_pair_counts": {
            f"{source}->{target}": count
            for (source, target), count in sorted(endpoint_pairs.items())
        },
        "domain_pattern_counts": {
            "->".join(pattern): count for pattern, count in sorted(domain_patterns.items())
        },
    }


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return min(1.0, numerator / denominator)
