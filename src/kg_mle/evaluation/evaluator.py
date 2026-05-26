"""Deterministic evaluator plus optional LLM judge aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kg_mle.evaluation.judge import LLMJudge
from kg_mle.generator.protocol import Conversation
from kg_mle.repair import (
    DeterministicRepairPlanner,
    RepairPolicy,
    RepairResult,
    assign_quality_band,
    should_attempt_repair,
)
from kg_mle.utils.paths import ensure_parent_dir


class EvaluatedConversation(dict):
    """Typed alias-by-subclass for JSON-serialisable per-record metrics."""


class DatasetEvaluation(dict):
    """Typed alias-by-subclass for JSON-serialisable aggregate metrics."""


def load_conversations_jsonl(path: Path) -> list[dict[str, Any]]:
    conversations = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                conversations.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return conversations


def evaluate_dataset(
    conversations: list[dict[str, Any]],
    *,
    judge: LLMJudge | None = None,
    max_judged_records: int | None = None,
    repair: bool = False,
    repair_policy: RepairPolicy | None = None,
    repair_planner: Any | None = None,
    max_repair_attempts: int = 1,
) -> DatasetEvaluation:
    records: list[dict[str, Any]] = []
    scored_conversations: list[dict[str, Any]] = []
    repair_results: list[dict[str, Any]] = []
    policy = repair_policy or RepairPolicy()
    active_repair_planner = repair_planner or DeterministicRepairPlanner()
    for idx, conversation in enumerate(conversations):
        judge_this = judge is not None and (
            max_judged_records is None or idx < max_judged_records
        )
        record, scored_conversation = _evaluate_one(
            conversation, judge=judge if judge_this else None
        )
        if repair:
            record, scored_conversation, repair_result = _maybe_repair_one(
                scored_conversation,
                record,
                judge=judge if judge_this else None,
                policy=policy,
                planner=active_repair_planner,
                max_attempts=max_repair_attempts,
            )
            if repair_result is not None:
                repair_results.append(repair_result.model_dump())
        records.append(record)
        scored_conversations.append(scored_conversation)
    return DatasetEvaluation(
        {
            "summary": _aggregate(records),
            "repair_summary": _repair_summary(repair, repair_results),
            "records": records,
            "scored_conversations": scored_conversations,
        }
    )


def save_evaluation(evaluation: DatasetEvaluation, path: Path) -> None:
    ensure_parent_dir(path)
    metrics = {k: v for k, v in evaluation.items() if k != "scored_conversations"}
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_scored_conversations(evaluation: DatasetEvaluation, path: Path) -> None:
    ensure_parent_dir(path)
    conversations = evaluation.get("scored_conversations", [])
    with path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(json.dumps(conversation, default=str) + "\n")


def _evaluate_one(
    conversation_payload: dict[str, Any],
    *,
    judge: LLMJudge | None,
) -> tuple[EvaluatedConversation, dict[str, Any]]:
    schema_valid = True
    schema_error = None
    try:
        conversation = Conversation.model_validate(conversation_payload)
        payload = conversation.model_dump()
    except ValidationError as exc:
        schema_valid = False
        schema_error = str(exc)
        payload = conversation_payload

    messages = payload.get("messages", [])
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    assistant_tool_calls = [
        call
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    ]
    tool_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    tool_errors = [
        message
        for message in tool_messages
        if isinstance(message.get("content"), dict) and "error" in message["content"]
    ]
    successful_tool_messages = [m for m in tool_messages if m not in tool_errors]
    expected_tool_calls = int(
        payload.get("metadata", {}).get("n_tool_calls") or len(assistant_tool_calls) or 0
    )
    role_sequence_valid = bool(roles) and roles[0] == "user" and all(
        role in {"user", "assistant", "tool", "system"} for role in roles
    )
    tool_response_coverage = _ratio(len(tool_messages), len(assistant_tool_calls))
    chain_completion = _ratio(len(successful_tool_messages), expected_tool_calls)
    error_free_trace = 1.0 if not tool_errors else max(0.0, 1.0 - len(tool_errors) / max(1, len(tool_messages)))
    deterministic_score = round(
        (
            (1.0 if schema_valid else 0.0)
            + (1.0 if role_sequence_valid else 0.0)
            + tool_response_coverage
            + chain_completion
            + error_free_trace
        )
        / 5
        * 10,
        4,
    )

    record = EvaluatedConversation(
        {
            "conversation_id": payload.get("conversation_id", "unknown"),
            "schema_valid": schema_valid,
            "schema_error": schema_error,
            "role_sequence_valid": role_sequence_valid,
            "n_messages": len(messages),
            "n_assistant_tool_calls": len(assistant_tool_calls),
            "n_tool_messages": len(tool_messages),
            "n_tool_errors": len(tool_errors),
            "tool_response_coverage": round(tool_response_coverage, 4),
            "chain_completion": round(chain_completion, 4),
            "error_free_trace": round(error_free_trace, 4),
            "deterministic_score": deterministic_score,
            "llm_judge": None,
        }
    )
    if judge is not None:
        try:
            record["llm_judge"] = judge.score(payload).model_dump()
        except Exception as exc:
            record["llm_judge"] = {"error": str(exc)}
    quality_band, usable_for_training = assign_quality_band(record)
    record["quality_band"] = quality_band
    record["usable_for_training"] = usable_for_training
    scored_conversation = dict(payload)
    metadata = dict(scored_conversation.get("metadata") or {})
    metadata["evaluation"] = {
        "schema_valid": record["schema_valid"],
        "role_sequence_valid": record["role_sequence_valid"],
        "tool_response_coverage": record["tool_response_coverage"],
        "chain_completion": record["chain_completion"],
        "error_free_trace": record["error_free_trace"],
        "deterministic_score": record["deterministic_score"],
        "llm_judge": record["llm_judge"],
        "quality_band": quality_band,
        "usable_for_training": usable_for_training,
    }
    scored_conversation["metadata"] = metadata
    return record, scored_conversation


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "conversation_count": 0,
            "mean_deterministic_score": 0.0,
            "mean_chain_completion": 0.0,
            "mean_tool_response_coverage": 0.0,
            "schema_valid_rate": 0.0,
            "llm_judged_count": 0,
            "mean_llm_overall_score": None,
            "usable_for_training_rate": 0.0,
        }
    llm_scores = [
        record["llm_judge"]["overall_score"]
        for record in records
        if isinstance(record.get("llm_judge"), dict)
        and "overall_score" in record["llm_judge"]
    ]
    return {
        "conversation_count": len(records),
        "mean_deterministic_score": round(_mean(r["deterministic_score"] for r in records), 4),
        "mean_chain_completion": round(_mean(r["chain_completion"] for r in records), 4),
        "mean_tool_response_coverage": round(
            _mean(r["tool_response_coverage"] for r in records), 4
        ),
        "schema_valid_rate": round(_mean(1.0 if r["schema_valid"] else 0.0 for r in records), 4),
        "llm_judged_count": len(llm_scores),
        "mean_llm_overall_score": round(_mean(llm_scores), 4) if llm_scores else None,
        "usable_for_training_rate": round(
            _mean(1.0 if r.get("usable_for_training") else 0.0 for r in records), 4
        ),
    }


def _maybe_repair_one(
    scored_conversation: dict[str, Any],
    record: dict[str, Any],
    *,
    judge: LLMJudge | None,
    policy: RepairPolicy,
    planner: DeterministicRepairPlanner,
    max_attempts: int,
) -> tuple[dict[str, Any], dict[str, Any], RepairResult | None]:
    triggers = should_attempt_repair(record, policy)
    if not triggers or max_attempts < 1:
        return record, scored_conversation, None
    plan = planner.plan(scored_conversation, triggers=triggers)
    before_scores = _score_snapshot(record)
    repaired_conversation, status = planner.apply(scored_conversation, plan)
    after_record, after_scored = _evaluate_one(repaired_conversation, judge=judge)
    after_scores = _score_snapshot(after_record)
    result = RepairResult(
        status=status,
        plan=plan,
        attempts=1,
        before_scores=before_scores,
        after_scores=after_scores,
        notes=[
            "Deterministic repair pass is bounded to one attempt.",
            "Coordinator-required repairs are planned and recorded, but not regenerated in-place.",
        ]
        if plan.requires_coordinator
        else ["Deterministic repair applied in evaluation pass."],
    )
    metadata = dict(after_scored.get("metadata") or {})
    history = list(metadata.get("repair_history") or [])
    history.append(result.model_dump())
    metadata["repair_history"] = history
    metadata.setdefault("evaluation", {})
    metadata["evaluation"]["repair"] = {
        "attempted": True,
        "status": result.status,
        "strategy": result.plan.strategy,
        "before_scores": result.before_scores,
        "after_scores": result.after_scores,
    }
    after_scored["metadata"] = metadata
    after_record["repair"] = metadata["evaluation"]["repair"]
    return after_record, after_scored, result


def _score_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    snapshot = {
        "deterministic_score": record.get("deterministic_score"),
        "quality_band": record.get("quality_band"),
        "usable_for_training": record.get("usable_for_training"),
    }
    judge = record.get("llm_judge")
    if isinstance(judge, dict):
        snapshot["llm_judge"] = {
            key: judge.get(key)
            for key in (
                "task_completion",
                "tool_trace_validity",
                "argument_grounding",
                "response_grounding",
                "naturalness",
                "overall_score",
            )
            if key in judge
        }
    return snapshot


def _repair_summary(enabled: bool, repair_results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for result in repair_results:
        status = str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "enabled": enabled,
        "attempted": len(repair_results),
        "repaired": status_counts.get("repaired", 0),
        "failed": status_counts.get("failed", 0),
        "rejected": status_counts.get("rejected", 0),
        "regenerated": status_counts.get("regenerated", 0),
        "status_counts": status_counts,
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return min(1.0, numerator / denominator)


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
