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
from kg_mle.executor.validator import ExecutorError, grounded_parameters_for_endpoint
from kg_mle.generator.agents import Assistant, DeterministicAssistant, Planner, UserSimulator
from kg_mle.generator.protocol import (
    AssistantTurn,
    ChainDeviation,
    Conversation,
    GeneratorConfig,
    Plan,
    ToolCallProposal,
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
    completion_guarantee: dict[str, Any]


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
            completion_guarantee={
                "triggered": False,
                "reason": None,
                "unfinished_from_step": None,
                "llm_reprompt_attempts": 0,
                "llm_reprompt_succeeded": False,
                "deterministic_turns": [],
            },
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
                    _mark_completion_guarantee(
                        meta,
                        reason="empty_tool_calls",
                        steps_completed=steps_completed,
                    )
                    retry_turn = self._reprompt_stalled_assistant(
                        plan=plan,
                        transcript=transcript,
                        session=session,
                        steps_completed=steps_completed,
                        clarifications_taken=clarifications_taken,
                        seed=seed + turn_counter,
                        reason="empty_tool_calls",
                        meta=meta,
                    )
                    if retry_turn is not None:
                        assistant_turn = retry_turn
                    if not assistant_turn.tool_calls:
                        # The agent claimed a tool call but supplied none. Fall back
                        # to the deterministic assistant to make real progress.
                        assistant_turn = DeterministicAssistant().compose_turn(
                            plan,
                            transcript,
                            session,
                            steps_completed=steps_completed,
                            clarifications_taken=clarifications_taken,
                            config=self._config,
                            seed=seed + turn_counter,
                        )
                        _record_deterministic_turn(meta, assistant_turn, steps_completed)
                    # The fallback/reprompt may decide the current step needs a
                    # clarification; handle it rather than dropping it (dropping
                    # it would burn turns until max_turns and truncate the chain).
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
                    if assistant_turn.kind != "tool_calls" or not assistant_turn.tool_calls:
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
                if advanced == 0 and steps_completed < len(current_chain):
                    _mark_completion_guarantee(
                        meta,
                        reason="tool_call_no_progress",
                        steps_completed=steps_completed,
                    )
                continue

            if assistant_turn.kind == "final_summary":
                if steps_completed < len(plan.step_plans):
                    _mark_completion_guarantee(
                        meta,
                        reason="premature_final_summary",
                        steps_completed=steps_completed,
                    )
                    retry_turn = self._reprompt_stalled_assistant(
                        plan=plan,
                        transcript=transcript,
                        session=session,
                        steps_completed=steps_completed,
                        clarifications_taken=clarifications_taken,
                        seed=seed + turn_counter,
                        reason="premature_final_summary",
                        meta=meta,
                    )
                    if retry_turn is not None:
                        assistant_turn = retry_turn
                    if assistant_turn.kind == "final_summary":
                        assistant_turn = DeterministicAssistant().compose_turn(
                            plan,
                            transcript,
                            session,
                            steps_completed=steps_completed,
                            clarifications_taken=clarifications_taken,
                            config=self._config,
                            seed=seed + turn_counter,
                        )
                        _record_deterministic_turn(meta, assistant_turn, steps_completed)
                    if assistant_turn.kind != "final_summary":
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
                transcript.append({"role": "assistant", "content": assistant_turn.content})
                break

        if steps_completed < len(current_chain):
            _mark_completion_guarantee(
                meta,
                reason="max_turns_unfinished",
                steps_completed=steps_completed,
            )
            retry_turn = self._reprompt_stalled_assistant(
                plan=plan,
                transcript=transcript,
                session=session,
                steps_completed=steps_completed,
                clarifications_taken=clarifications_taken,
                seed=seed + 80001,
                reason="max_turns_unfinished",
                meta=meta,
            )
            if retry_turn is not None:
                if retry_turn.kind == "clarification":
                    handled = self._handle_clarification(
                        plan=plan,
                        assistant_turn=retry_turn,
                        transcript=transcript,
                        seed=seed + 80001,
                        steps_completed=steps_completed,
                    )
                    if handled is not None:
                        meta.clarifications_taken.append(handled)
                        clarifications_taken += 1
                elif retry_turn.kind == "tool_calls" and retry_turn.tool_calls:
                    advanced = self._handle_tool_calls(
                        assistant_turn=retry_turn,
                        transcript=transcript,
                        session=session,
                        current_transitions=current_transitions,
                        steps_completed=steps_completed,
                        meta=meta,
                        plan=plan,
                        seed=seed + 80001,
                    )
                    steps_completed += advanced

        # Completion guarantee. If the agent (typically an LLM) stalled and the
        # main loop exited with the chain unfinished, drive the remaining steps
        # deterministically. This keeps every conversation complete and grounded
        # regardless of provider behaviour — the conversation-level version of
        # the per-turn deterministic fallback. Deterministic agents always reach
        # final_summary in-loop, so this is a no-op for the offline default.
        det = DeterministicAssistant()
        safety = 0
        while steps_completed < len(current_chain) and safety < 3 * len(current_chain) + 6:
            safety += 1
            turn = det.compose_turn(
                plan,
                transcript,
                session,
                steps_completed=steps_completed,
                clarifications_taken=clarifications_taken,
                config=self._config,
                seed=seed + 90001 + safety,
            )
            _mark_completion_guarantee(
                meta,
                reason=meta.completion_guarantee.get("reason") or "deterministic_completion",
                steps_completed=steps_completed,
            )
            _record_deterministic_turn(meta, turn, steps_completed)
            if turn.kind == "clarification":
                handled = self._handle_clarification(
                    plan=plan,
                    assistant_turn=turn,
                    transcript=transcript,
                    seed=seed + 90001 + safety,
                    steps_completed=steps_completed,
                )
                if handled is not None:
                    meta.clarifications_taken.append(handled)
                    clarifications_taken += 1
                continue
            if turn.kind == "tool_calls" and turn.tool_calls:
                advanced = self._handle_tool_calls(
                    assistant_turn=turn,
                    transcript=transcript,
                    session=session,
                    current_transitions=current_transitions,
                    steps_completed=steps_completed,
                    meta=meta,
                    plan=plan,
                    seed=seed + 90001 + safety,
                )
                steps_completed += advanced
                if advanced == 0:
                    break
                continue
            break

        # Ensure the conversation closes with a final assistant summary.
        last = transcript[-1] if transcript else None
        if not (
            last
            and last.get("role") == "assistant"
            and last.get("content")
            and not last.get("tool_calls")
        ):
            if not meta.completion_guarantee.get("triggered"):
                _mark_completion_guarantee(
                    meta,
                    reason="missing_final_summary",
                    steps_completed=steps_completed,
                )
            summary = det.compose_turn(
                plan,
                transcript,
                session,
                steps_completed=len(plan.step_plans),
                clarifications_taken=clarifications_taken,
                config=self._config,
                seed=seed,
            )
            if summary.kind == "final_summary" and summary.content:
                if meta.completion_guarantee.get("triggered"):
                    _record_deterministic_turn(meta, summary, len(plan.step_plans))
                transcript.append({"role": "assistant", "content": summary.content})

        meta.final_chain = current_chain
        return Conversation(
            conversation_id=conversation_id or f"conv_{seed:08d}",
            messages=transcript,
            plan=plan,
            metadata=_metadata_dict(meta, sampling_result, current_chain, current_transitions),
        )

    def _reprompt_stalled_assistant(
        self,
        *,
        plan: Plan,
        transcript: list[dict[str, Any]],
        session: ExecutorSession,
        steps_completed: int,
        clarifications_taken: int,
        seed: int,
        reason: str,
        meta: CoordinatorRunMeta,
    ) -> AssistantTurn | None:
        """Give an LLM assistant bounded directive retries before fallback.

        The directive is appended only to an internal transcript copy; it never
        becomes dataset content.
        """
        if self._assistant.__class__ is DeterministicAssistant:
            return None
        if not (0 <= steps_completed < len(plan.step_plans)):
            return None
        step = plan.step_plans[steps_completed]
        directive = {
            "role": "assistant",
            "content": (
                "Coordinator internal retry directive: the previous assistant "
                f"turn stalled because {reason}. The conversation is not done. "
                f"You must now continue step {steps_completed} by either asking "
                "one clarification for this step or emitting exactly one tool "
                f"call for endpoint {step.endpoint_id}. Do not summarize yet."
            ),
        }
        for attempt in range(self._config.max_stall_reprompts):
            meta.completion_guarantee["llm_reprompt_attempts"] += 1
            retry_turn = self._assistant.compose_turn(
                plan,
                [*transcript, directive],
                session,
                steps_completed=steps_completed,
                clarifications_taken=clarifications_taken,
                config=self._config,
                seed=seed + 70001 + attempt,
            )
            if retry_turn.kind == "clarification" and retry_turn.clarification_target is not None:
                meta.completion_guarantee["llm_reprompt_succeeded"] = True
                return retry_turn
            if retry_turn.kind == "tool_calls" and retry_turn.tool_calls:
                meta.completion_guarantee["llm_reprompt_succeeded"] = True
                return retry_turn
        return None

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
                    # Prefer the structured value the user agent returns; only
                    # fall back to parsing prose when it is absent (keeping the
                    # project's "don't trust free-text" principle — see DESIGN #11).
                    value = user_reply.clarified_value or _extract_clarified_value(
                        user_reply.content, target.parameter_name
                    )
                    if value:
                        pp.suggested_value = value
                        pp.reason = f"{pp.reason} User clarified."
                    pp.ambiguous = False
                    pp.confidence = 1.0
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
            proposal = _sanitize_tool_call(
                proposal,
                plan=plan,
                session=session,
                steps_completed=steps_completed + advanced,
            )
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
            proposal = _sanitize_tool_call(
                repair_turn.tool_calls[0],
                plan=plan,
                session=session,
                steps_completed=steps_completed,
            )
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
    """Fallback parse for a clarified value when the user agent did not return a
    structured `clarified_value`. Handles the deterministic template
    'For X, use Y.' and the common LLM phrasing 'Use the <param> Y.'"""
    marker = "use "
    idx = content.lower().find(marker)
    value = content[idx + len(marker) :] if idx != -1 else content
    value = value.strip().rstrip(".").strip()
    value = _strip_param_prefix(value, parameter_name)
    value = _strip_param_suffix(value, parameter_name)
    return value or None


def _strip_param_suffix(value: str, parameter_name: str) -> str:
    """Drop a trailing clause that restates the parameter, e.g.
    'Las Vegas Convention Center as the location' -> 'Las Vegas Convention Center'."""
    name_forms = {parameter_name.lower(), parameter_name.replace("_", " ").lower()}
    low = value.lower()
    for name in name_forms:
        if not name:
            continue
        for pattern in (f" as the {name}", f" as {name}", f" for the {name}", f" for {name}"):
            cut = low.rfind(pattern)
            if cut != -1:
                return value[:cut].strip()
    return value


def _strip_param_prefix(value: str, parameter_name: str) -> str:
    """Remove a leading article, a restatement of the parameter name, or a
    connector that an LLM tends to include ('the game_id valorant' -> 'valorant',
    'game_id is valorant' -> 'valorant')."""
    name_forms = {parameter_name.lower(), parameter_name.replace("_", " ").lower()}
    leading = ("the ", "a ", "an ", "is ", "to ", "= ", ": ")
    changed = True
    while changed and value:
        changed = False
        low = value.lower()
        for prefix in leading:
            if low.startswith(prefix):
                value = value[len(prefix):].strip()
                changed = True
                break
        if changed:
            continue
        for name in name_forms:
            if name and low.startswith(name + " "):
                value = value[len(name):].strip()
                changed = True
                break
    return value.strip()


def _sanitize_tool_call(
    proposal: ToolCallProposal,
    *,
    plan: Plan,
    session: ExecutorSession,
    steps_completed: int,
) -> ToolCallProposal:
    """Reconcile agent-supplied arguments against trusted runtime state.

    The LLM's prose is always kept. For tool-call *arguments* we keep what the
    LLM authored and override only the parameters that have a real
    grounding/provenance concern:

      - grounded-transition targets -> the value the prior step actually issued
        (prevents executor rejection of hallucinated IDs);
      - plan / user-clarified values -> the disambiguated value;
      - same-domain contextual fields (a value a prior step issued/passed under
        the same name/canonical, plus where/when cluster bridges) -> the prior
        value, for chain coherence.

    Everything else the LLM produced — titles, queries, free descriptions — is
    type-valid and carries no provenance constraint, so it is left untouched.
    Malformed (wrong-type) calls are passed through unchanged so the executor's
    typed error and the repair loop stay visible in the transcript.
    """
    if not (0 <= steps_completed < len(plan.step_plans)):
        return proposal
    step = plan.step_plans[steps_completed]
    if proposal.endpoint_id != step.endpoint_id:
        return proposal

    endpoint = session.endpoint_index.get(proposal.endpoint_id)
    if endpoint is None:
        return proposal
    if _has_obvious_type_error(proposal, endpoint):
        return proposal

    # Start from the LLM's own arguments, but drop any keys the endpoint does
    # not declare. The executor tolerates extras by default, so hallucinated
    # keys would otherwise survive into the transcript and hurt the judge.
    declared_names = {param.name for param in endpoint.parameters}
    args = {name: value for name, value in proposal.arguments.items() if name in declared_names}
    suggestions = session.suggest_arguments(proposal.endpoint_id)
    plan_values = _plan_values_for_step(step)
    user_clarified = _user_clarified_params_for_step(step)
    grounded = grounded_parameters_for_endpoint(
        proposal.endpoint_id, session.sampling_result.transitions
    )

    for param in endpoint.parameters:
        if not param.required:
            continue
        name = param.name

        # Fill a missing/empty required arg from the executor's suggestion.
        if args.get(name) in (None, "") and name in suggestions:
            args[name] = suggestions[name]

        # Grounded params must use the value the prior step issued — this is
        # what prevents the live-LLM grounding rejections. Settled here; plan
        # and contextual overrides below do not apply to grounded params.
        if name in grounded:
            issued = session.state.issued_ids(grounded[name]) or session.state.issued_ids(name)
            if issued:
                args[name] = issued[-1]
            elif name in suggestions:
                args[name] = suggestions[name]
            continue

        # Plan / user-clarified non-placeholder values win for free params.
        if (
            name in plan_values
            and plan_values[name] is not None
            and plan_values[name] != f"<{name}>"
        ):
            args[name] = plan_values[name]

        # Same-domain contextual coherence (respects user-clarified values).
        contextual = _contextual_value_for_param(
            name,
            param.canonical_name,
            session=session,
            plan_value=args.get(name),
            user_clarified=name in user_clarified,
        )
        if contextual is not None:
            args[name] = contextual

    return ToolCallProposal(
        endpoint_id=proposal.endpoint_id,
        arguments=args,
        call_confidence=proposal.call_confidence,
    )


def _plan_values_for_step(step) -> dict[str, Any]:
    return {pp.parameter_name: pp.suggested_value for pp in step.parameter_plans}


def _user_clarified_params_for_step(step) -> set[str]:
    return {
        pp.parameter_name
        for pp in step.parameter_plans
        if "user clarified" in pp.reason.lower()
    }


def _has_obvious_type_error(proposal: ToolCallProposal, endpoint) -> bool:
    """Let the executor/repair path handle malformed types.

    The sanitizer is for provenance conflicts in otherwise well-shaped LLM
    calls. If a call is structurally malformed, preserving it keeps typed
    executor errors and repair coverage visible in the transcript.
    """
    for param in endpoint.parameters:
        if param.name not in proposal.arguments:
            continue
        value = proposal.arguments[param.name]
        if param.type == "string" and not isinstance(value, str):
            return True
        if param.type == "integer" and not isinstance(value, int):
            return True
        if param.type == "number" and not isinstance(value, (int, float)):
            return True
        if param.type == "boolean" and not isinstance(value, bool):
            return True
        if param.type == "array" and not isinstance(value, list):
            return True
        if param.type == "object" and not isinstance(value, dict):
            return True
    return False


# Cross-field clusters where the registry keeps separate canonicals but the
# values are interchangeable when chaining a "where"/"when" value forward
# (enrichment maps venue->location and available_time->start_time). `city` is
# deliberately NOT bridged with location/venue: a city field should hold a city
# name, not a full venue/location string (e.g. "Las Vegas", not "Las Vegas
# Convention Center"). A city only chains from a prior `city` via the generic
# same-name lookup. Anything outside these clusters is matched generically by
# the parameter's own name/canonical, so newly chained fields (symbol,
# currency, ...) work without being added here.
_CONTEXTUAL_CLUSTERS: tuple[tuple[str, ...], ...] = (
    ("location", "venue"),
    ("start_time", "time"),
)


def _contextual_value_for_param(
    param_name: str,
    canonical_name: str | None,
    *,
    session: ExecutorSession,
    plan_value: Any,
    user_clarified: bool,
) -> Any | None:
    # An explicit user-clarified value is authoritative for every field.
    if user_clarified and plan_value not in (None, ""):
        return None

    keys = [param_name]
    if canonical_name and canonical_name != param_name:
        keys.append(canonical_name)

    # 1. Generic provenance: a value a prior step issued or passed under this
    #    field's own name or canonical name.
    for key in keys:
        value = _prior_value(session, key)
        if value is not None:
            return value

    # 2. Cluster bridge for interchangeable where/when fields.
    normalized = {key.lower() for key in keys}
    for cluster in _CONTEXTUAL_CLUSTERS:
        if normalized & set(cluster):
            for field in cluster:
                value = _prior_value(session, field)
                if value is not None:
                    return value

    # 3. date derived from a prior start_time.
    if normalized & {"date"}:
        start_times = session.state.issued_ids("start_time")
        if start_times and "T" in start_times[-1]:
            return start_times[-1].split("T", 1)[0]

    return None


def _prior_value(session: ExecutorSession, key: str) -> Any | None:
    """Most recent value passed into, or issued by, a prior tool call for `key`."""
    recent = _recent_tool_arg(session, key)
    if recent not in (None, ""):
        return recent
    issued = session.state.issued_ids(key)
    if issued:
        return issued[-1]
    return None


def _recent_tool_arg(session: ExecutorSession, arg_name: str) -> Any | None:
    """Most recent value passed into a *successful* prior tool call for `arg_name`.

    Failed calls are logged (record_call runs before validation), so a
    rejected bad-argument call would otherwise poison this lookup and get
    re-applied on every repair. We pair each tool_call with its outcome and
    only consider calls that produced a tool_response.
    """
    result: Any | None = None
    pending: dict[str, Any] | None = None
    for entry in session.state.log:
        if entry.kind == "tool_call":
            pending = entry.arguments
        elif entry.kind == "tool_response":
            if pending and pending.get(arg_name) not in (None, ""):
                result = pending[arg_name]
            pending = None
        elif entry.kind == "tool_error":
            pending = None  # failed call — its arguments are not provenance
    return result


def _mark_completion_guarantee(
    meta: CoordinatorRunMeta,
    *,
    reason: str,
    steps_completed: int,
) -> None:
    if not meta.completion_guarantee.get("triggered"):
        meta.completion_guarantee["reason"] = reason
        meta.completion_guarantee["unfinished_from_step"] = steps_completed
    meta.completion_guarantee["triggered"] = True


def _record_deterministic_turn(
    meta: CoordinatorRunMeta,
    turn: AssistantTurn,
    step_index: int,
) -> None:
    entry: dict[str, Any] = {"kind": turn.kind}
    if turn.kind in {"tool_calls", "clarification"}:
        entry["step_index"] = step_index
    if turn.kind == "tool_calls" and turn.tool_calls:
        entry["endpoint_id"] = turn.tool_calls[0].endpoint_id
    if turn.kind == "clarification" and turn.clarification_target is not None:
        entry["parameter_name"] = turn.clarification_target.parameter_name
    meta.completion_guarantee["deterministic_turns"].append(entry)


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
        "completion_guarantee": meta.completion_guarantee,
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
