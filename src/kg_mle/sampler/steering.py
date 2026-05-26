"""Cross-conversation steering: counters and constraint shaping.

The steerer is the single component that turns "I just sampled chain X" into
"don't sample anything too similar to X next time." Isolating it here means
the planner's main loop is identical whether steering is on or off — turning
steering off just swaps in a `NullSteerer` that records counters for stats
but never returns penalties.

The mechanism is a hybrid:

- Hard exclusion (`forbid_endpoint_ids`): endpoints whose count exceeds a
  per-corpus threshold are added to the next chain's forbid list. This is
  the most decisive lever and what the sampler natively understands.
- Soft preference (`bias_required_domains`): when the planner picks one or
  two `required_domains` per chain, the steerer returns the least-used
  domains so coverage naturally rebalances.

Counters are always recorded — even when steering is "off" via the
`NullSteerer` — because the diversity experiment needs comparable corpus
stats for both Run A (steering off) and Run B (steering on).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from kg_mle.sampler.constraints import SamplingResult


@dataclass
class CorpusCounters:
    """All counters the steerer maintains. Plain dataclass so the planner
    can serialize this directly into the dataset's run-level metadata."""

    domains: Counter[str] = field(default_factory=Counter)
    tools: Counter[str] = field(default_factory=Counter)
    endpoints: Counter[str] = field(default_factory=Counter)
    endpoint_pairs: Counter[tuple[str, str]] = field(default_factory=Counter)
    chain_lengths: Counter[int] = field(default_factory=Counter)
    domain_patterns: Counter[tuple[str, ...]] = field(default_factory=Counter)
    chain_count: int = 0

    def record(self, result: SamplingResult) -> None:
        self.chain_count += 1
        self.chain_lengths[len(result.endpoints)] += 1

        domain_sequence: list[str] = []
        for endpoint_id in result.endpoints:
            domain = endpoint_id.split("/", 1)[0]
            self.endpoints[endpoint_id] += 1
            self.domains[domain] += 1
            if not domain_sequence or domain_sequence[-1] != domain:
                domain_sequence.append(domain)
        self.domain_patterns[tuple(domain_sequence)] += 1

        for transition in result.transitions:
            self.endpoint_pairs[(transition.source, transition.target)] += 1

        for tool in result.metadata.get("tools_visited", ()):
            if tool:
                self.tools[str(tool)] += 1

    def as_summary(self) -> dict:
        """Compact, JSON-serialisable summary for the run-level metadata."""
        return {
            "chain_count": self.chain_count,
            "domain_counts": dict(self.domains),
            "tool_counts": dict(self.tools),
            "endpoint_counts": dict(self.endpoints),
            "chain_length_counts": {str(k): v for k, v in self.chain_lengths.items()},
            "domain_pattern_counts": {
                "->".join(pattern): count for pattern, count in self.domain_patterns.items()
            },
            "endpoint_pair_counts": {
                f"{src}->{tgt}": count for (src, tgt), count in self.endpoint_pairs.items()
            },
        }


class CorpusSteerer:
    """Steering counters + penalty shaping for the planner."""

    def __init__(
        self,
        *,
        target_count: int,
        endpoint_count: int,
        endpoint_overuse_factor: float = 1.6,
        pair_overuse_factor: float = 2.0,
    ) -> None:
        """`endpoint_overuse_factor` is the multiplier over the uniform-usage
        baseline at which an endpoint becomes a candidate for hard exclusion.

        With target_count=100 and endpoint_count=45, uniform usage is
        ~2.2 chains/endpoint. Factor 1.6 means an endpoint exceeding ~3.5
        chains gets flagged. Tunable per corpus size.
        """
        self.counters = CorpusCounters()
        # Baseline is uniform usage per endpoint. We floor it at 1.0 so small
        # corpora (target_count < endpoint_count) still allow at least a few
        # repeats before flagging. The hard floor of 3 prevents the forbid
        # list from filling up so fast that the planner runs out of options.
        baseline = max(1.0, target_count / max(1, endpoint_count))
        self._endpoint_overuse_threshold = max(3, math.ceil(baseline * endpoint_overuse_factor))
        self._pair_overuse_threshold = max(3, math.ceil(baseline * pair_overuse_factor))

    @property
    def endpoint_overuse_threshold(self) -> int:
        return self._endpoint_overuse_threshold

    def record(self, result: SamplingResult) -> None:
        self.counters.record(result)

    def forbid_endpoints(self) -> tuple[str, ...]:
        """Endpoints whose usage exceeds the per-corpus threshold."""
        overused = [
            endpoint_id
            for endpoint_id, count in self.counters.endpoints.items()
            if count >= self._endpoint_overuse_threshold
        ]
        return tuple(sorted(overused))

    def forbid_endpoint_pairs(self) -> tuple[tuple[str, str], ...]:
        """Endpoint transitions whose usage is excessive.

        The walker cannot forbid pairs directly — it forbids endpoints. We
        surface these for the planner's stats and for future use if pair-level
        forbidding becomes a planner constraint.
        """
        return tuple(
            sorted(
                pair
                for pair, count in self.counters.endpoint_pairs.items()
                if count >= self._pair_overuse_threshold
            )
        )

    def least_used_domains(
        self,
        available_domains: Iterable[str],
        *,
        k: int = 1,
    ) -> tuple[str, ...]:
        """Return up to `k` least-used domains from `available_domains`.

        Ties are broken by domain name so the result is reproducible across
        runs with the same counter state. Domains never seen are sorted first
        (zero count) — they get explicit preference, which is what drives
        coverage rebalancing.
        """
        choices = sorted(
            available_domains,
            key=lambda domain: (self.counters.domains.get(domain, 0), domain),
        )
        return tuple(choices[:k])


class NullSteerer:
    """No-op steerer used when `--no-cross-conversation-steering` is set.

    Records counters (so stats are comparable to a steered run) but never
    returns forbid-lists or biased domain choices. Selection by the planner
    falls back to the planner's own deterministic round-robin.
    """

    def __init__(self) -> None:
        self.counters = CorpusCounters()
        self._endpoint_overuse_threshold = 0

    @property
    def endpoint_overuse_threshold(self) -> int:
        return self._endpoint_overuse_threshold

    def record(self, result: SamplingResult) -> None:
        self.counters.record(result)

    def forbid_endpoints(self) -> tuple[str, ...]:
        return ()

    def forbid_endpoint_pairs(self) -> tuple[tuple[str, str], ...]:
        return ()

    def least_used_domains(
        self,
        available_domains: Iterable[str],
        *,
        k: int = 1,
    ) -> tuple[str, ...]:
        sorted_domains = sorted(available_domains)
        return tuple(sorted_domains[:k])
