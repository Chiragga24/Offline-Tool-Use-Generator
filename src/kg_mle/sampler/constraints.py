"""Constraint and result dataclasses for tool-chain sampling.

`ChainConstraints` is the user-facing knob set the sampler accepts. It captures
the assignment's two explicit constrained-sampling examples — exact chain
length and a required domain — plus the dimensions needed to target the
required dataset properties (multi-tool ratio, varied lengths, balanced
coverage, coherent chaining).

`SamplingResult` is the immutable output. It carries enough metadata for the
planner to aggregate corpus-level stats and for downstream components
(executor, judge) to know how each transition was justified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ChainPattern = Literal["sequential", "parallel"]
AdvanceType = Literal["grounded", "same_domain", "semantic"]


@dataclass(frozen=True)
class ChainConstraints:
    """Inputs to one chain sample.

    n_steps is either an exact int or an inclusive (min, max) range. Range
    form drives the "varied conversation lengths" dataset requirement: the
    planner can pass (2, 5) and let the walker pick deterministically per
    seed.

    min_grounded_transitions is the load-bearing knob for "coherent
    chaining": it forces at least N of the (n_steps - 1) transitions to be
    backed by an `output_satisfies_input` edge. Set to n_steps - 1 to
    require fully grounded chains; set to 0 to allow same_domain throughout.
    """

    n_steps: int | tuple[int, int]
    min_distinct_tools: int = 1
    min_distinct_domains: int = 1
    required_domains: tuple[str, ...] = ()
    required_endpoint: str | None = None
    min_grounded_transitions: int = 0
    pattern: ChainPattern = "sequential"
    allow_semantic_edges: bool = False
    forbid_endpoint_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Transition:
    """One step in a sampled chain."""

    source: str
    target: str
    advance_type: AdvanceType
    parameter: str | None = None
    source_field: str | None = None
    match_type: str | None = None


@dataclass(frozen=True)
class SamplingResult:
    """Output of one chain sample.

    `metadata` is intentionally a free-form dict so the planner can extend
    what it aggregates without changing the dataclass shape.
    """

    endpoints: tuple[str, ...]
    transitions: tuple[Transition, ...]
    pattern: ChainPattern
    seed: int
    constraints: ChainConstraints
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("domains_visited", ()))

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("tools_visited", ()))

    @property
    def grounded_transition_count(self) -> int:
        return sum(1 for transition in self.transitions if transition.advance_type == "grounded")


class UnsatisfiableConstraintsError(RuntimeError):
    """Raised when the walker exhausts the search space without finding a chain
    that satisfies the given constraints. Carries the partial-attempt metadata
    so callers can decide whether to relax constraints or surface the failure."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}
