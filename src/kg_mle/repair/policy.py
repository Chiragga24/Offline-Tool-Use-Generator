"""Repair trigger detection and quality banding."""

from __future__ import annotations

from typing import Any

from kg_mle.repair.models import RepairTrigger


class RepairPolicy:
    """Thresholds for deciding when a conversation needs repair."""

    def __init__(
        self,
        *,
        deterministic_threshold: float = 8.0,
        tool_trace_threshold: float = 8.0,
        argument_grounding_threshold: float = 8.0,
        task_completion_threshold: float = 9.0,
        naturalness_threshold: float = 5.0,
    ) -> None:
        self.deterministic_threshold = deterministic_threshold
        self.tool_trace_threshold = tool_trace_threshold
        self.argument_grounding_threshold = argument_grounding_threshold
        self.task_completion_threshold = task_completion_threshold
        self.naturalness_threshold = naturalness_threshold


def should_attempt_repair(record: dict[str, Any], policy: RepairPolicy) -> list[RepairTrigger]:
    triggers: list[RepairTrigger] = []
    if not record.get("schema_valid", False):
        triggers.append(RepairTrigger(kind="validation_error", reason="Conversation schema failed."))
    if not record.get("role_sequence_valid", False):
        triggers.append(
            RepairTrigger(kind="validation_error", reason="Role sequence validation failed.")
        )
    if int(record.get("n_tool_errors") or 0) > 0:
        triggers.append(RepairTrigger(kind="tool_error", reason="Tool error present in trace."))

    _add_score_trigger(
        triggers,
        record,
        score_name="deterministic_score",
        threshold=policy.deterministic_threshold,
        kind="low_deterministic_score",
    )

    judge = record.get("llm_judge")
    if isinstance(judge, dict) and "error" not in judge:
        _add_score_trigger(
            triggers,
            judge,
            score_name="tool_trace_validity",
            threshold=policy.tool_trace_threshold,
            kind="low_tool_trace_validity",
        )
        _add_score_trigger(
            triggers,
            judge,
            score_name="argument_grounding",
            threshold=policy.argument_grounding_threshold,
            kind="low_argument_grounding",
        )
        _add_score_trigger(
            triggers,
            judge,
            score_name="task_completion",
            threshold=policy.task_completion_threshold,
            kind="low_task_completion",
        )
        _add_score_trigger(
            triggers,
            judge,
            score_name="naturalness",
            threshold=policy.naturalness_threshold,
            kind="low_naturalness",
        )
    return triggers


def assign_quality_band(record: dict[str, Any]) -> tuple[str, bool]:
    if not record.get("schema_valid", False) or int(record.get("n_tool_errors") or 0) > 0:
        return "reject", False
    deterministic_score = float(record.get("deterministic_score") or 0.0)
    judge = record.get("llm_judge")
    if isinstance(judge, dict) and "error" not in judge:
        if (
            judge.get("tool_trace_validity", 0.0) < 8.0
            or judge.get("argument_grounding", 0.0) < 8.0
            or judge.get("task_completion", 0.0) < 9.0
        ):
            return "reject", False
        if deterministic_score >= 9.0 and not judge.get("issues"):
            return "gold", True
    if deterministic_score >= 9.0:
        return "gold", True
    if deterministic_score >= 8.0:
        return "silver", True
    return "reject", False


def _add_score_trigger(
    triggers: list[RepairTrigger],
    scores: dict[str, Any],
    *,
    score_name: str,
    threshold: float,
    kind: str,
) -> None:
    value = scores.get(score_name)
    if value is None:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return
    if numeric < threshold:
        triggers.append(
            RepairTrigger(
                kind=kind,
                reason=f"{score_name}={numeric} below threshold {threshold}.",
                score_name=score_name,
                score_value=numeric,
                threshold=threshold,
            )
        )
