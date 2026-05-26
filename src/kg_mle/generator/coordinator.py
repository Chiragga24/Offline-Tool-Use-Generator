"""ConversationCoordinator — the single entry point for one conversation.

The coordinator owns the transcript, drives the three agents in
strict sequence, validates every agent output via Pydantic, and applies
the configured gates (planner-driven disambiguation, confidence-gated
assistant initiative, confidence-gated chain deviations). It also
owns the repair flow: when the executor rejects a tool call, the
coordinator gives the assistant one more chance to compose corrected
arguments, then records the trail.

Conversation termination is chain-bound by default: the conversation
ends after all chain endpoints have been called *and* a final summary
turn has been written. The assistant may extend the chain via accepted
`add_step` or `modify_step` deviations; both are graph-verified before
acceptance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from kg_mle.executor.session import ExecutorSession, OfflineExecutor
from kg_mle.executor.validator import ExecutorError
from kg_mle.generator.agents import Assistant, Planner, UserSimulator
from kg_mle.generator.protocol import (
    AssistantTurn,
    ChainDeviation,
    Conversation,
    GeneratorConfig,
    Plan,
)
from kg_mle.graph.models import GraphEdge, ToolGraph
from kg_mle.registry.models import ToolRegistry
from kg_mle.sampler.constraints import (
    AdvanceType,
    SamplingResult,
    Transition,
)


_EDGE_TO_ADVANCE: dict[str, AdvanceType] = {
    "output_satisfies_input": "grounded",
    "same_domain": "same_domain",
    "semantic_related": "semantic",
}


@dataclass
class CoordinatorRunMeta:
    """Per-conversation metadata the coordinator tracks for the dataset.

    Everything here ends up in `Conversation.metadata` and is what the
    judge / repair / diversity analysis reads."""

    clarifications_taken: list[dict[str, Any]]
    repair_attempts: list[dict[str, Any]]
    deviations_accepted: list[dict[str, Any]]
    deviations_rejected: list[dict[str, Any]]
    final_chain: list[str]
    advance_type_counts: dict[str, int]


class ConversationCoordinator:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        graph: ToolGraph,
        executor: OfflineExecutor,
        planner: Planner,
        user_simulator: UserSimulator,
        assistant: Assistant,
        config: GeneratorConfig | None = None,
    ) -> None:
        self._registry = registry
        self._graph = graph
        self._executor = executor
        self._planner = planner
        self._user = user_simulator
        self._assistant = assistant
        self._config = config or GeneratorConfig()
        self._endpoint_ids: set[str] = {e.endpoint_id for e in registry.endpoints}
        self._adjacency = _build_endpoint_adjacency(graph)

    def run(
        self,
        sampling_result: SamplingResult,
        *,
        seed: int,
        conversation_id: str | None = None,
    ) -> Conversation:
        plan = self._planner.plan(sampling_result, seed=seed)
        session = self._executor.open_session(sampling_result, seed=seed)
        # The current chain may evolve mid-conversation via accepted deviations.
        current_chain: list[str] = list(sampling_result.endpoints)
        current_transitions: list[Transition] = list(sampling_result.transitions)

        transcript: list[dict[str, Any]] = []
        meta = CoordinatorRunMeta(
            clarifications_taken=[],
            repair_attempts=[],
            deviations_accepted=[],
            deviations_rejected=[],
            final_chain=current_chain,
            advance_type_counts=defaultdict(int),
        )

        # Turn 1: initial user request
        user_turn = self._user.initial_request(plan, seed=seed)
        transcript.append({"role": "user", "content": user_turn.content})

        # Main loop
        steps_completed = 0
        clarifications_taken = 0
        turn_counter = 0
        max_turns = self._max_turns_for(current_chain)

        while turn_counter < max_turns:
            turn_counter += 1
            assistant_turn = self._assistant.compose_turn(
                plan,
                transcript,
                session,
                steps_completed=steps_completed,
                clarifications_taken=clarifications_taken,
                config=self._config,
                seed=seed + turn_counter,
            )

            # Deviation evaluation always runs first, regardless of turn kind,
            # because an accepted deviation reshapes the chain before the
            # current turn is interpreted.
            if assistant_turn.chain_deviation is not None:
                outcome = self._apply_deviation_if_accepted(
                    assistant_turn.chain_deviation,
                    current_chain=current_chain,
                    current_transitions=current_transitions,
                    session=session,
                    steps_completed=steps_completed,
                )
                if outcome["accepted"]:
                    meta.deviations_accepted.append(outcome)
                    current_chain = outcome["new_chain"]
                    current_transitions = outcome["new_transitions"]
                    max_turns = self._max_turns_for(current_chain)
                else:
                    meta.deviations_rejected.append(outcome)

            if assistant_turn.kind == "clarification":
                handled = self._handle_clarification(
                    plan=plan,
                    assistant_turn=assistant_turn,
                    transcript=transcript,
                    seed=seed + turn_counter,
                    steps_completed=steps_completed,
                )
                if handled is not None:
                    meta.clarifications_taken.append(handled)
                    clarifications_taken += 1
                continue

            if assistant_turn.kind == "tool_calls":
                if not assistant_turn.tool_calls:
                    # Coordinator-level safety: kind=tool_calls implies ≥1 call.
                    continue
                advanced = self._handle_tool_calls(
                    assistant_turn=assistant_turn,
                    transcript=transcript,
                    session=session,
                    current_transitions=current_transitions,
                    steps_completed=steps_completed,
                    meta=meta,
                    plan=plan,
                    seed=seed + turn_counter,
                )
                steps_completed += advanced
                continue

            if assistant_turn.kind == "final_summary":
                transcript.append({"role": "assistant", "content": assistant_turn.content})
                break

        meta.final_chain = current_chain
        return Conversation(
            conversation_id=conversation_id or f"conv_{seed:08d}",
            messages=transcript,
            plan=plan,
            metadata=_metadata_dict(meta, sampling_result, current_chain, current_transitions),
        )

    # ----- deviation handling --------------------------------------------

    def _apply_deviation_if_accepted(
        self,
        deviation: ChainDeviation,
        *,
        current_chain: list[str],
        current_transitions: list[Transition],
        session: ExecutorSession,
        steps_completed: int,
    ) -> dict[str, Any]:
        """Confidence-gated + graph-verified. Returns an outcome dict either
        way (recorded in metadata).
        """
        proposal = {
            "kind": deviation.kind,
            "endpoint_id": deviation.endpoint_id,
            "position": deviation.position,
            "reasoning": deviation.reasoning,
            "deviation_confidence": deviation.deviation_confidence,
            "accepted": False,
        }

        if deviation.deviation_confidence < self._config.assistant_deviation_threshold:
            proposal["reject_reason"] = "below_confidence_threshold"
            return proposal

        if deviation.endpoint_id not in self._endpoint_ids:
            proposal["reject_reason"] = "unknown_endpoint"
            return proposal

        if deviation.position < steps_completed + 1:
            # Cannot retroactively modify already-executed steps.
            proposal["reject_reason"] = "position_in_past"
            return proposal

        new_chain, new_transitions = self._construct_modified_chain(
            current_chain=current_chain,
            current_transitions=current_transitions,
            deviation=deviation,
        )
        if new_chain is None:
            proposal["reject_reason"] = "no_graph_path"
            return proposal

        # Update the session's sampling_result so subsequent grounding
        # checks use the new transitions.
        session.sampling_result = SamplingResult(
            endpoints=tuple(new_chain),
            transitions=tuple(new_transitions),
            pattern=session.sampling_result.pattern,
            seed=session.sampling_result.seed,
            constraints=session.sampling_result.constraints,
            metadata=session.sampling_result.metadata,
        )

        proposal["accepted"] = True
        proposal["new_chain"] = new_chain
        proposal["new_transitions"] = new_transitions
        return proposal

    def _construct_modified_chain(
        self,
        *,
        current_chain: list[str],
        current_transitions: list[Transition],
        deviation: ChainDeviation,
    ) -> tuple[list[str] | None, list[Transition] | None]:
        if deviation.kind == "add_step":
            if not (0 < deviation.position <= len(current_chain)):
                return None, None
            prev = current_chain[deviation.position - 1]
            next_eid = (
                current_chain[deviation.position]
                if deviation.position < len(current_chain)
                else None
            )
            in_trans = self._best_edge_between(prev, deviation.endpoint_id)
            if in_trans is None:
                return None, None
            if next_eid is not None:
                out_trans = self._best_edge_between(deviation.endpoint_id, next_eid)
                if out_trans is None:
                    return None, None
            else:
                out_trans = None
            new_chain = (
                current_chain[: deviation.position]
                + [deviation.endpoint_id]
                + current_chain[deviation.position :]
            )
            new_transitions = list(current_transitions)
            # Replace transition that used to bridge (prev, next) with two.
            insert_at = deviation.position - 1
            if out_trans is not None:
                if insert_at < len(new_transitions):
                    new_transitions[insert_at] = in_trans
                    new_transitions.insert(insert_at + 1, out_trans)
                else:
                    new_transitions.append(in_trans)
            else:
                # Appending past chain end.
                new_transitions.append(in_trans)
            return new_chain, new_transitions

        if deviation.kind == "modify_step":
            if not (0 <= deviation.position < len(current_chain)):
                return None, None
            prev = current_chain[deviation.position - 1] if deviation.position > 0 else None
            next_eid = (
                current_chain[deviation.position + 1]
                if deviation.position + 1 < len(current_chain)
                else None
            )
            in_trans = (
                self._best_edge_between(prev, deviation.endpoint_id) if prev is not None else None
            )
            if prev is not None and in_trans is None:
                return None, None
            out_trans = (
                self._best_edge_between(deviation.endpoint_id, next_eid)
                if next_eid is not None
                else None
            )
            if next_eid is not None and out_trans is None:
                return None, None
            new_chain = list(current_chain)
            new_chain[deviation.position] = deviation.endpoint_id
            new_transitions = list(current_transitions)
            if in_trans is not None and deviation.position - 1 < len(new_transitions):
                new_transitions[deviation.position - 1] = in_trans
            if out_trans is not None and deviation.position < len(new_transitions):
                new_transitions[deviation.position] = out_trans
            return new_chain, new_transitions

        return None, None

    def _best_edge_between(self, source: str, target: str) -> Transition | None:
        """Pick the strongest endpoint-to-endpoint edge between two nodes.

        Preference: grounded > same_domain > semantic_related. None if
        no endpoint-to-endpoint edge exists in either direction."""
        edges = self._adjacency.get(source, [])
        candidates: list[Transition] = []
        for edge in edges:
            target_eid = _strip_endpoint(edge.target)
            if target_eid != target:
                continue
            advance = _EDGE_TO_ADVANCE.get(edge.type)
            if advance is None:
                continue
            md = edge.metadata or {}
            candidates.append(
                Transition(
                    source=source,
                    target=target,
                    advance_type=advance,
                    parameter=md.get("parameter"),
                    source_field=md.get("source_field"),
                    match_type=md.get("match_type"),
                )
            )
        if not candidates:
            return None
        priority = {"grounded": 0, "same_domain": 1, "semantic": 2}
        candidates.sort(key=lambda t: priority[t.advance_type])
        return candidates[0]

    # ----- turn handlers -------------------------------------------------

    def _handle_clarification(
        self,
        *,
        plan: Plan,
        assistant_turn: AssistantTurn,
        transcript: list[dict[str, Any]],
        seed: int,
        steps_completed: int,
    ) -> dict[str, Any] | None:
        """Record the assistant's question and obtain a user reply."""
        target = assistant_turn.clarification_target
        if target is None:
            # The agent emitted clarification without specifying what.
            # Skip without recording — keeps the conversation moving.
            return None

        transcript.append(
            {
                "role": "assistant",
                "content": assistant_turn.content,
                "clarification_target_step": target.step_index,
                "clarification_target_parameter": target.parameter_name,
                "assistant_clarification_confidence": assistant_turn.assistant_clarification_confidence,
            }
        )

        user_reply = self._user.reply_to_clarification(
            plan,
            target_step=target.step_index,
            target_parameter=target.parameter_name,
            seed=seed,
        )
        transcript.append({"role": "user", "content": user_reply.content})

        # Persist the supplied value back into the plan so the next
        # tool_call uses it.
        if 0 <= target.step_index < len(plan.step_plans):
            for pp in plan.step_plans[target.step_index].parameter_plans:
                if pp.parameter_name == target.parameter_name:
                    # Extract a value from the user reply if it follows the
                    # deterministic template "For X, use Y."
                    value = _extract_clarified_value(user_reply.content, target.parameter_name)
                    if value:
                        pp.suggested_value = value
                    pp.ambiguous = False
                    pp.confidence = max(pp.confidence, 0.9)
                    break

        return {
            "step_index": target.step_index,
            "parameter_name": target.parameter_name,
            "initiated_by": (
                "planner" if steps_completed in plan.ambiguous_step_indices else "assistant"
            ),
            "assistant_clarification_confidence": assistant_turn.assistant_clarification_confidence,
        }

    def _handle_tool_calls(
        self,
        *,
        assistant_turn: AssistantTurn,
        transcript: list[dict[str, Any]],
        session: ExecutorSession,
        current_transitions: list[Transition],
        steps_completed: int,
        meta: CoordinatorRunMeta,
        plan: Plan,
        seed: int,
    ) -> int:
        """Execute one or more tool calls; record success/failure/repair.

        Returns the number of chain steps advanced (typically 1).
        """
        advanced = 0
        for proposal in assistant_turn.tool_calls:
            transcript.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "endpoint_id": proposal.endpoint_id,
                            "arguments": proposal.arguments,
                            "call_confidence": proposal.call_confidence,
                        }
                    ],
                }
            )
            try:
                response = session.call(proposal.endpoint_id, proposal.arguments)
            except ExecutorError as exc:
                transcript.append(
                    {
                        "role": "tool",
                        "endpoint": proposal.endpoint_id,
                        "content": {"error": exc.to_log_entry()},
                    }
                )
                # One repair attempt: re-ask the assistant to compose corrected args.
                repaired = self._attempt_repair(
                    plan=plan,
                    transcript=transcript,
                    session=session,
                    steps_completed=steps_completed,
                    seed=seed,
                    failing_endpoint=proposal.endpoint_id,
                )
                meta.repair_attempts.append(
                    {
                        "endpoint": proposal.endpoint_id,
                        "original_error": exc.to_log_entry(),
                        "repaired": repaired,
                    }
                )
                if repaired:
                    advanced += 1
                # If repair failed, we don't advance the chain; the loop's
                # max_turns budget eventually closes the conversation.
                continue

            # success
            transcript.append(
                {"role": "tool", "endpoint": proposal.endpoint_id, "content": response}
            )
            advance_type = _advance_type_for_step(steps_completed + advanced, current_transitions)
            if advance_type is not None:
                meta.advance_type_counts[advance_type] += 1
            advanced += 1
        return advanced

    def _attempt_repair(
        self,
        *,
        plan: Plan,
        transcript: list[dict[str, Any]],
        session: ExecutorSession,
        steps_completed: int,
        seed: int,
        failing_endpoint: str,
    ) -> bool:
        """Re-prompt the assistant. Returns True if a corrected call succeeded."""
        for attempt in range(self._config.max_repair_attempts):
            repair_turn = self._assistant.compose_turn(
                plan,
                transcript,
                session,
                steps_completed=steps_completed,
                clarifications_taken=0,
                config=self._config,
                seed=seed + 100 + attempt,
            )
            if repair_turn.kind != "tool_calls" or not repair_turn.tool_calls:
                return False
            proposal = repair_turn.tool_calls[0]
            transcript.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "endpoint_id": proposal.endpoint_id,
                            "arguments": proposal.arguments,
                            "call_confidence": proposal.call_confidence,
                        }
                    ],
                    "repair_attempt": attempt + 1,
                }
            )
            try:
                response = session.call(proposal.endpoint_id, proposal.arguments)
            except ExecutorError as exc:
                transcript.append(
                    {
                        "role": "tool",
                        "endpoint": proposal.endpoint_id,
                        "content": {"error": exc.to_log_entry()},
                    }
                )
                continue
            transcript.append(
                {"role": "tool", "endpoint": proposal.endpoint_id, "content": response}
            )
            return True
        return False

    @staticmethod
    def _max_turns_for(chain: list[str]) -> int:
        """Generous upper bound: each step can produce a clarification (2
        messages) + tool_call (2 messages); plus 2 bookends; plus headroom
        for repairs and deviations."""
        return max(20, 8 + len(chain) * 6)


# ----- helpers --------------------------------------------------------------


def _build_endpoint_adjacency(graph: ToolGraph) -> dict[str, list[GraphEdge]]:
    """Index outgoing endpoint-to-endpoint edges per source endpoint_id."""
    adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in graph.edges:
        if edge.type not in _EDGE_TO_ADVANCE:
            continue
        source_eid = _strip_endpoint(edge.source)
        if source_eid is None:
            continue
        adjacency[source_eid].append(edge)
    return dict(adjacency)


def _strip_endpoint(node_id: str) -> str | None:
    if node_id.startswith("endpoint:"):
        return node_id[len("endpoint:") :]
    return None


def _advance_type_for_step(
    step_index: int,
    transitions: list[Transition],
) -> str | None:
    """The advance_type of the transition that led INTO this step."""
    if step_index == 0:
        return None
    if step_index - 1 >= len(transitions):
        return None
    return transitions[step_index - 1].advance_type


def _extract_clarified_value(content: str, parameter_name: str) -> str | None:
    """Parse the deterministic user's reply template 'For X, use Y.'"""
    marker = f"use "
    idx = content.find(marker)
    if idx == -1:
        return None
    value = content[idx + len(marker) :].rstrip(".")
    return value or None


def _metadata_dict(
    meta: CoordinatorRunMeta,
    sampling_result: SamplingResult,
    final_chain: list[str],
    final_transitions: list[Transition],
) -> dict[str, Any]:
    return {
        "seed": sampling_result.seed,
        "original_chain": list(sampling_result.endpoints),
        "final_chain": final_chain,
        "n_tool_calls": len(final_chain),
        "domains": sorted({eid.split("/", 1)[0] for eid in final_chain}),
        "tools_visited": list(sampling_result.metadata.get("tools_visited", [])),
        "advance_type_counts": dict(meta.advance_type_counts),
        "clarifications_taken": meta.clarifications_taken,
        "repair_attempts": meta.repair_attempts,
        "deviations_accepted": [
            {k: v for k, v in dev.items() if k not in ("new_chain", "new_transitions")}
            for dev in meta.deviations_accepted
        ],
        "deviations_rejected": meta.deviations_rejected,
        "transition_summary": [
            {
                "source": t.source,
                "target": t.target,
                "advance_type": t.advance_type,
                "parameter": t.parameter,
            }
            for t in final_transitions
        ],
    }
