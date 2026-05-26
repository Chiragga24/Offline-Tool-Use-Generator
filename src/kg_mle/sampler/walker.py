"""Graph-walking tool-chain sampler.

The walker uses the `ToolGraph` directly (not a hardcoded list, per the
assignment's hard requirement). It does a deterministic depth-first search
with backtracking, preferring `output_satisfies_input` transitions to keep
chains groundable, falling back to `same_domain` when needed, and only
considering `semantic_related` when the constraints opt in.

Design notes:

- Determinism: a single `random.Random(seed)` instance threads through every
  shuffle. Same seed + same constraints + same graph = same chain.
- Tie-breaking inside each edge-type tier is by `(target endpoint_id)` so
  ordering is stable even before the seeded shuffle. The shuffle is what
  varies across seeds; tie-breaking is what keeps single-seed runs
  reproducible across graph permutations.
- Constraint satisfaction is enforced at the terminal node (length matches
  n_steps) by checking distinct counts and grounded transition counts.
- The walker rejects revisits (no endpoint appears twice in the same chain)
  to avoid trivial loops.
- `parallel` pattern is recognized in the dataclass but emits a sequential
  chain for now; structural parallelism is a planner concern (multiple
  independent chains that fan in to a synthesizer step) and will be added
  alongside the planner.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from kg_mle.graph.models import GraphEdge, ToolGraph
from kg_mle.sampler.constraints import (
    AdvanceType,
    ChainConstraints,
    SamplingResult,
    Transition,
    UnsatisfiableConstraintsError,
)


_EDGE_TYPE_TO_ADVANCE: dict[str, AdvanceType] = {
    "output_satisfies_input": "grounded",
    "same_domain": "same_domain",
    "semantic_related": "semantic",
}

_TIER_PRIORITY: dict[AdvanceType, int] = {
    "grounded": 0,
    "same_domain": 1,
    "semantic": 2,
}


class ToolChainSampler:
    """Graph-walking chain sampler."""

    def __init__(self, graph: ToolGraph) -> None:
        self._graph = graph
        self._adjacency = _build_adjacency(graph)
        self._endpoint_node_ids = _index_endpoint_nodes(graph)
        self._endpoint_ids = tuple(sorted(self._endpoint_node_ids.keys()))
        self._endpoints_by_domain: dict[str, list[str]] = defaultdict(list)
        for endpoint_id in self._endpoint_ids:
            metadata = self._endpoint_node_ids[endpoint_id]
            self._endpoints_by_domain[str(metadata.get("domain", ""))].append(endpoint_id)

    def sample(self, constraints: ChainConstraints, *, seed: int) -> SamplingResult:
        """Sample one chain satisfying `constraints` deterministically from `seed`.

        Raises `UnsatisfiableConstraintsError` if no chain in the search space
        satisfies the constraints.
        """
        rng = random.Random(seed)
        n_steps = _resolve_n_steps(constraints.n_steps, rng)
        _validate_constraints(constraints, n_steps)

        max_grounded = n_steps - 1
        if constraints.min_grounded_transitions > max_grounded:
            raise UnsatisfiableConstraintsError(
                f"min_grounded_transitions={constraints.min_grounded_transitions} "
                f"exceeds maximum possible transitions {max_grounded} for n_steps={n_steps}."
            )

        backtrack_counter = {"value": 0}
        starts = self._candidate_starts(constraints, rng)
        for start in starts:
            chain: list[str] = [start]
            transitions: list[Transition] = []
            if self._search(chain, transitions, n_steps, constraints, rng, backtrack_counter):
                return self._build_result(
                    chain=tuple(chain),
                    transitions=tuple(transitions),
                    constraints=constraints,
                    seed=seed,
                    n_steps=n_steps,
                    backtracks=backtrack_counter["value"],
                )

        raise UnsatisfiableConstraintsError(
            "No chain in the graph satisfies the given constraints.",
            diagnostics={
                "n_steps": n_steps,
                "candidate_starts_tried": len(starts),
                "backtracks": backtrack_counter["value"],
                "constraints": constraints,
            },
        )

    def _candidate_starts(self, constraints: ChainConstraints, rng: random.Random) -> list[str]:
        """Order start candidates deterministically, prefer ones that help satisfy required_domains.

        If `required_endpoint` is set, that's the only start we try (chains
        starting elsewhere can still hit it, but enumerating those is more
        expensive — defer until needed).
        """
        if constraints.required_endpoint:
            if constraints.required_endpoint not in self._endpoint_node_ids:
                raise UnsatisfiableConstraintsError(
                    f"required_endpoint {constraints.required_endpoint!r} is not in the graph."
                )
            return [constraints.required_endpoint]

        candidates = [
            endpoint_id
            for endpoint_id in self._endpoint_ids
            if endpoint_id not in constraints.forbid_endpoint_ids
        ]

        if constraints.required_domains:
            priority = set(constraints.required_domains)

            def domain_priority(endpoint_id: str) -> int:
                metadata = self._endpoint_node_ids[endpoint_id]
                return 0 if metadata.get("domain") in priority else 1

            candidates.sort(key=lambda eid: (domain_priority(eid), eid))
        else:
            candidates.sort()

        rng_local = random.Random(rng.random())
        priority_group = [c for c in candidates if not constraints.required_domains or self._endpoint_node_ids[c].get("domain") in constraints.required_domains]
        rest_group = [c for c in candidates if c not in set(priority_group)]
        rng_local.shuffle(priority_group)
        rng_local.shuffle(rest_group)
        return priority_group + rest_group

    def _search(
        self,
        chain: list[str],
        transitions: list[Transition],
        target_len: int,
        constraints: ChainConstraints,
        rng: random.Random,
        backtrack_counter: dict[str, int],
    ) -> bool:
        if len(chain) == target_len:
            return self._terminal_check(chain, transitions, constraints)

        remaining_steps = target_len - len(chain) - 1
        grounded_so_far = sum(1 for transition in transitions if transition.advance_type == "grounded")
        needed_grounded = max(0, constraints.min_grounded_transitions - grounded_so_far)
        if needed_grounded > remaining_steps + 1:
            # Cannot reach grounded threshold even if every remaining edge is grounded.
            return False

        current = chain[-1]
        outgoing = self._adjacency.get(current, ())
        candidates = self._filter_and_order(
            outgoing,
            chain=chain,
            constraints=constraints,
            rng=rng,
            need_grounded_now=needed_grounded > remaining_steps,
        )

        for candidate in candidates:
            target = candidate.transition.target
            if target in chain:
                continue
            if target in constraints.forbid_endpoint_ids:
                continue
            chain.append(target)
            transitions.append(candidate.transition)
            if self._search(chain, transitions, target_len, constraints, rng, backtrack_counter):
                return True
            chain.pop()
            transitions.pop()
            backtrack_counter["value"] += 1
        return False

    def _filter_and_order(
        self,
        outgoing: tuple["_Candidate", ...],
        *,
        chain: list[str],
        constraints: ChainConstraints,
        rng: random.Random,
        need_grounded_now: bool,
    ) -> list["_Candidate"]:
        allowed: list[_Candidate] = []
        for candidate in outgoing:
            advance = candidate.transition.advance_type
            if advance == "semantic" and not constraints.allow_semantic_edges:
                continue
            if need_grounded_now and advance != "grounded":
                continue
            allowed.append(candidate)

        # Deterministic shuffle within tiers, tiers preserved by priority.
        by_tier: dict[int, list[_Candidate]] = defaultdict(list)
        for candidate in allowed:
            by_tier[_TIER_PRIORITY[candidate.transition.advance_type]].append(candidate)
        ordered: list[_Candidate] = []
        for tier in sorted(by_tier.keys()):
            tier_candidates = sorted(by_tier[tier], key=lambda c: c.transition.target)
            rng.shuffle(tier_candidates)
            ordered.extend(tier_candidates)
        return ordered

    def _terminal_check(
        self,
        chain: list[str],
        transitions: list[Transition],
        constraints: ChainConstraints,
    ) -> bool:
        distinct_tools = {self._endpoint_node_ids[eid].get("tool_name") for eid in chain}
        distinct_domains = {self._endpoint_node_ids[eid].get("domain") for eid in chain}
        if len(distinct_tools) < constraints.min_distinct_tools:
            return False
        if len(distinct_domains) < constraints.min_distinct_domains:
            return False
        if constraints.required_domains:
            visited_domains = {self._endpoint_node_ids[eid].get("domain") for eid in chain}
            if not set(constraints.required_domains).issubset(visited_domains):
                return False
        grounded = sum(1 for transition in transitions if transition.advance_type == "grounded")
        if grounded < constraints.min_grounded_transitions:
            return False
        if constraints.required_endpoint and constraints.required_endpoint not in chain:
            return False
        return True

    def _build_result(
        self,
        *,
        chain: tuple[str, ...],
        transitions: tuple[Transition, ...],
        constraints: ChainConstraints,
        seed: int,
        n_steps: int,
        backtracks: int,
    ) -> SamplingResult:
        domains_visited = tuple(
            dict.fromkeys(str(self._endpoint_node_ids[eid].get("domain", "")) for eid in chain)
        )
        tools_visited = tuple(
            dict.fromkeys(str(self._endpoint_node_ids[eid].get("tool_name", "")) for eid in chain)
        )
        advance_counts = {advance: 0 for advance in _EDGE_TYPE_TO_ADVANCE.values()}
        for transition in transitions:
            advance_counts[transition.advance_type] += 1

        metadata: dict[str, Any] = {
            "domains_visited": list(domains_visited),
            "tools_visited": list(tools_visited),
            "advance_type_counts": advance_counts,
            "grounded_transitions": advance_counts["grounded"],
            "n_steps": n_steps,
            "backtracks": backtracks,
            "start_endpoint": chain[0],
        }
        return SamplingResult(
            endpoints=chain,
            transitions=transitions,
            pattern=constraints.pattern,
            seed=seed,
            constraints=constraints,
            metadata=metadata,
        )


class _Candidate:
    """Internal candidate edge for the walker."""

    __slots__ = ("transition",)

    def __init__(self, transition: Transition) -> None:
        self.transition = transition


def _build_adjacency(graph: ToolGraph) -> dict[str, tuple[_Candidate, ...]]:
    """Index outgoing transitions per endpoint_id.

    Only endpoint-to-endpoint edges become candidates: `output_satisfies_input`,
    `same_domain`, `semantic_related`. Structural edges (`contains_tool`,
    `exposes_endpoint`, `requires_parameter`, `returns_field`) are ignored —
    they describe the graph's anatomy, not chain advancement.
    """
    adjacency: dict[str, list[_Candidate]] = defaultdict(list)
    for edge in graph.edges:
        advance = _EDGE_TYPE_TO_ADVANCE.get(edge.type)
        if advance is None:
            continue
        source_eid = _strip_endpoint_prefix(edge.source)
        target_eid = _strip_endpoint_prefix(edge.target)
        if source_eid is None or target_eid is None:
            continue
        transition = _transition_from_edge(edge, source_eid, target_eid, advance)
        adjacency[source_eid].append(_Candidate(transition))
    return {key: tuple(value) for key, value in adjacency.items()}


def _index_endpoint_nodes(graph: ToolGraph) -> dict[str, dict[str, Any]]:
    """Map endpoint_id -> node metadata (domain, tool_name, etc.)."""
    nodes: dict[str, dict[str, Any]] = {}
    tools_by_node: dict[str, str] = {}
    for node in graph.nodes:
        if node.type == "tool":
            tools_by_node[node.node_id] = node.label

    for node in graph.nodes:
        if node.type != "endpoint":
            continue
        endpoint_id = str(node.metadata.get("endpoint_id") or _strip_endpoint_prefix(node.node_id))
        if not endpoint_id:
            continue
        tool_name = _resolve_tool_name(graph, node.node_id, tools_by_node)
        nodes[endpoint_id] = {
            "domain": node.metadata.get("domain"),
            "tool_name": tool_name,
            "name": node.label,
            "method": node.metadata.get("method"),
            "path": node.metadata.get("path"),
        }
    return nodes


def _resolve_tool_name(
    graph: ToolGraph,
    endpoint_node_id: str,
    tools_by_node: dict[str, str],
) -> str | None:
    for edge in graph.edges:
        if edge.type == "exposes_endpoint" and edge.target == endpoint_node_id:
            return tools_by_node.get(edge.source)
    return None


def _transition_from_edge(
    edge: GraphEdge, source_eid: str, target_eid: str, advance: AdvanceType
) -> Transition:
    metadata = edge.metadata or {}
    return Transition(
        source=source_eid,
        target=target_eid,
        advance_type=advance,
        parameter=metadata.get("parameter"),
        source_field=metadata.get("source_field"),
        match_type=metadata.get("match_type"),
    )


def _strip_endpoint_prefix(node_id: str) -> str | None:
    if node_id.startswith("endpoint:"):
        return node_id[len("endpoint:") :]
    return None


def _resolve_n_steps(n_steps: int | tuple[int, int], rng: random.Random) -> int:
    if isinstance(n_steps, int):
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        return n_steps
    low, high = n_steps
    if low < 1 or high < low:
        raise ValueError(f"n_steps range invalid: {n_steps}.")
    return rng.randint(low, high)


def _validate_constraints(constraints: ChainConstraints, n_steps: int) -> None:
    if constraints.min_distinct_tools > n_steps:
        raise UnsatisfiableConstraintsError(
            f"min_distinct_tools={constraints.min_distinct_tools} exceeds n_steps={n_steps}."
        )
    if constraints.min_distinct_domains > n_steps:
        raise UnsatisfiableConstraintsError(
            f"min_distinct_domains={constraints.min_distinct_domains} exceeds n_steps={n_steps}."
        )
    if len(constraints.required_domains) > n_steps:
        raise UnsatisfiableConstraintsError(
            f"required_domains count={len(constraints.required_domains)} exceeds n_steps={n_steps}."
        )
