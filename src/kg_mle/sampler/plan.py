"""Corpus-level planner.

The planner is the layer between the assignment's *dataset properties*
(50–60% multi-step, varied lengths, balanced domain coverage, coherent
chaining) and the walker's *per-chain constraint API*. For each chain in
the corpus, the planner builds a `ChainConstraints` instance that targets
those properties, calls the walker, records the result, and optionally
feeds the result back through the steerer to shape the next chain.

`steering_enabled=False` swaps the active steerer for `NullSteerer`. The
planner's main loop is otherwise identical, which means:

- Run A (`--no-cross-conversation-steering`) and Run B (default) differ
  only in whether the counters influence constraint building.
- Both runs produce comparable corpus stats (NullSteerer still records
  counters, just doesn't read them).
- Determinism is preserved: same planner seed + same target_count +
  same steering flag = same corpus.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from kg_mle.sampler.constraints import (
    ChainConstraints,
    SamplingResult,
    UnsatisfiableConstraintsError,
)
from kg_mle.sampler.steering import CorpusSteerer, NullSteerer
from kg_mle.sampler.walker import ToolChainSampler


# Length distribution targeting the "varied conversation lengths" property.
# Skewed toward 3-4 steps because those satisfy the "50-60% multi-step"
# requirement while keeping conversation length reasonable.
_DEFAULT_LENGTH_DISTRIBUTION: tuple[tuple[int, float], ...] = (
    (2, 0.20),
    (3, 0.35),
    (4, 0.30),
    (5, 0.15),
)


@dataclass(frozen=True)
class PlanEntry:
    """One chain in the corpus plan."""

    plan_index: int
    constraints: ChainConstraints
    seed: int


@dataclass
class CorpusReport:
    """What the planner produced. Fed into the dataset's run-level metadata."""

    results: list[SamplingResult] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    steering_enabled: bool = True
    counters_summary: dict[str, Any] = field(default_factory=dict)
    plan_meta: dict[str, Any] = field(default_factory=dict)


class CorpusPlanner:
    """Plans and samples a corpus targeting the assignment's dataset properties."""

    def __init__(
        self,
        sampler: ToolChainSampler,
        *,
        steering_enabled: bool,
        seed: int,
        length_distribution: tuple[tuple[int, float], ...] = _DEFAULT_LENGTH_DISTRIBUTION,
        multi_step_fraction: float = 0.55,
        max_relaxation_attempts: int = 8,
        allow_semantic_edges: bool = False,
    ) -> None:
        self._sampler = sampler
        self._seed = seed
        self._steering_enabled = steering_enabled
        self._length_distribution = length_distribution
        self._multi_step_fraction = multi_step_fraction
        self._max_relaxation_attempts = max_relaxation_attempts
        self._allow_semantic_edges = allow_semantic_edges
        self._available_domains = tuple(sorted(sampler._endpoints_by_domain.keys()))

    def sample_corpus(self, target_count: int) -> CorpusReport:
        if target_count < 1:
            raise ValueError(f"target_count must be >= 1, got {target_count}.")

        steerer = (
            CorpusSteerer(
                target_count=target_count,
                endpoint_count=len(self._sampler._endpoint_ids),
            )
            if self._steering_enabled
            else NullSteerer()
        )

        rng = random.Random(self._seed)
        report = CorpusReport(steering_enabled=self._steering_enabled)
        report.plan_meta = {
            "target_count": target_count,
            "seed": self._seed,
            "steering_enabled": self._steering_enabled,
            "length_distribution": list(self._length_distribution),
            "multi_step_fraction": self._multi_step_fraction,
            "endpoint_overuse_threshold": steerer.endpoint_overuse_threshold,
            "allow_semantic_edges": self._allow_semantic_edges,
        }

        for plan_index in range(target_count):
            entry = self._build_plan_entry(
                plan_index=plan_index,
                target_count=target_count,
                rng=rng,
                steerer=steerer,
            )
            result = self._sample_with_relaxation(entry, rng=rng, steerer=steerer, report=report)
            if result is not None:
                steerer.record(result)
                report.results.append(result)

        report.counters_summary = steerer.counters.as_summary()
        return report

    def _build_plan_entry(
        self,
        *,
        plan_index: int,
        target_count: int,
        rng: random.Random,
        steerer: CorpusSteerer | NullSteerer,
    ) -> PlanEntry:
        n_steps = _sample_n_steps(self._length_distribution, rng)
        require_multi = (plan_index / target_count) < self._multi_step_fraction or n_steps >= 3
        min_distinct_tools = 2 if require_multi and n_steps >= 3 else 1
        min_distinct_domains = 2 if n_steps >= 4 else 1
        # Coherent-chaining target: most transitions grounded, one same_domain
        # hop tolerated. For n_steps==2 the only transition must be grounded.
        if n_steps == 2:
            min_grounded = 1
        elif n_steps == 3:
            min_grounded = 1
        else:
            min_grounded = n_steps - 2

        required_domains: tuple[str, ...] = ()
        if min_distinct_domains >= 2:
            picked = steerer.least_used_domains(self._available_domains, k=1)
            if picked:
                required_domains = picked

        constraints = ChainConstraints(
            n_steps=n_steps,
            min_distinct_tools=min_distinct_tools,
            min_distinct_domains=min_distinct_domains,
            required_domains=required_domains,
            min_grounded_transitions=min_grounded,
            allow_semantic_edges=self._allow_semantic_edges,
            forbid_endpoint_ids=steerer.forbid_endpoints(),
        )

        # Per-chain seed: stable across runs (derived from planner seed +
        # plan_index), independent of any randomness consumed earlier in the
        # corpus. This means turning steering on/off does not perturb the
        # chain-level RNG, which keeps Run A and Run B directly comparable
        # at the per-chain level for shared (steering-agnostic) outcomes.
        chain_seed = (self._seed * 1_000_003 + plan_index) % (2**31 - 1)
        return PlanEntry(plan_index=plan_index, constraints=constraints, seed=chain_seed)

    def _sample_with_relaxation(
        self,
        entry: PlanEntry,
        *,
        rng: random.Random,
        steerer: CorpusSteerer | NullSteerer,
        report: CorpusReport,
    ) -> SamplingResult | None:
        constraints = entry.constraints
        attempts: list[dict[str, Any]] = []
        for relaxation_step in range(self._max_relaxation_attempts):
            try:
                result = self._sampler.sample(constraints, seed=entry.seed + relaxation_step)
                if relaxation_step > 0:
                    result.metadata["relaxation_history"] = attempts
                return result
            except UnsatisfiableConstraintsError as exc:
                attempts.append(
                    {
                        "step": relaxation_step,
                        "constraints": _dump_constraints(constraints),
                        "diagnostics": str(exc.args[0]),
                    }
                )
                constraints = _relax_one_step(constraints)

        report.failures.append(
            {
                "plan_index": entry.plan_index,
                "seed": entry.seed,
                "constraints": _dump_constraints(entry.constraints),
                "relaxation_history": attempts,
            }
        )
        return None


def _sample_n_steps(
    distribution: tuple[tuple[int, float], ...],
    rng: random.Random,
) -> int:
    values, weights = zip(*distribution)
    return rng.choices(values, weights=weights, k=1)[0]


def _relax_one_step(constraints: ChainConstraints) -> ChainConstraints:
    """Loosen the most-restrictive constraint, one knob per relaxation step.

    Order:
      1. min_grounded_transitions (cheapest signal to give up — falls back to
         same_domain transitions)
      2. required_domains (the chain may still hit them by coincidence)
      3. min_distinct_domains
      4. forbid_endpoint_ids (give up on steering for this one chain rather
         than fail outright — the steerer still records what was sampled)
      5. min_distinct_tools
      6. n_steps (only if still unsatisfiable; preserves chain length late)
    """
    if constraints.min_grounded_transitions > 0:
        return _replace(constraints, min_grounded_transitions=constraints.min_grounded_transitions - 1)
    if constraints.required_domains:
        return _replace(constraints, required_domains=())
    if constraints.min_distinct_domains > 1:
        return _replace(constraints, min_distinct_domains=constraints.min_distinct_domains - 1)
    if constraints.forbid_endpoint_ids:
        return _replace(constraints, forbid_endpoint_ids=())
    if constraints.min_distinct_tools > 1:
        return _replace(constraints, min_distinct_tools=constraints.min_distinct_tools - 1)
    if isinstance(constraints.n_steps, int) and constraints.n_steps > 2:
        return _replace(constraints, n_steps=constraints.n_steps - 1)
    return constraints


def _replace(constraints: ChainConstraints, **changes: Any) -> ChainConstraints:
    """Frozen-dataclass-safe replace helper."""
    from dataclasses import asdict, replace

    return replace(constraints, **changes)


def _dump_constraints(constraints: ChainConstraints) -> dict[str, Any]:
    return {
        "n_steps": constraints.n_steps,
        "min_distinct_tools": constraints.min_distinct_tools,
        "min_distinct_domains": constraints.min_distinct_domains,
        "required_domains": list(constraints.required_domains),
        "min_grounded_transitions": constraints.min_grounded_transitions,
        "allow_semantic_edges": constraints.allow_semantic_edges,
        "forbid_endpoint_ids": list(constraints.forbid_endpoint_ids),
    }
