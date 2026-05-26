from __future__ import annotations

import pytest
from pydantic import ValidationError

from kg_mle.schema import EvaluationMetricsArtifact


def _metrics_payload() -> dict:
    return {
        "summary": {
            "conversation_count": 1,
            "mean_deterministic_score": 9.8,
            "mean_chain_completion": 1.0,
            "mean_tool_response_coverage": 1.0,
            "schema_valid_rate": 1.0,
            "llm_judged_count": 1,
            "mean_llm_overall_score": 8.0,
            "usable_for_training_rate": 1.0,
        },
        "repair_summary": {
            "enabled": True,
            "attempted": 1,
            "repaired": 1,
            "failed": 0,
            "rejected": 0,
            "regenerated": 0,
            "status_counts": {"repaired": 1},
        },
        "records": [
            {
                "conversation_id": "conv_001",
                "schema_valid": True,
                "schema_error": None,
                "role_sequence_valid": True,
                "n_messages": 4,
                "n_assistant_tool_calls": 1,
                "n_tool_messages": 1,
                "n_tool_errors": 0,
                "tool_response_coverage": 1.0,
                "chain_completion": 1.0,
                "error_free_trace": 1.0,
                "deterministic_score": 10.0,
                "llm_judge": {
                    "task_completion": 8.0,
                    "tool_trace_validity": 9.0,
                    "argument_grounding": 9.0,
                    "response_grounding": 8.0,
                    "naturalness": 7.0,
                    "overall_score": 8.0,
                    "confidence": 8.0,
                    "issues": [],
                    "rationale": "Valid trace with a concise answer.",
                },
                "quality_band": "gold",
                "usable_for_training": True,
                "repair": {
                    "attempted": True,
                    "status": "repaired",
                    "strategy": "rewrite_final_response",
                    "before_scores": {"deterministic_score": 6.0},
                    "after_scores": {"deterministic_score": 10.0},
                },
            }
        ],
    }


def test_evaluation_metrics_artifact_accepts_current_shape() -> None:
    artifact = EvaluationMetricsArtifact.model_validate(_metrics_payload())

    assert artifact.summary.conversation_count == 1
    assert artifact.records[0].llm_judge is not None


def test_evaluation_metrics_artifact_allows_llm_judge_error() -> None:
    payload = _metrics_payload()
    payload["records"][0]["llm_judge"] = {"error": "provider quota exceeded"}
    payload["summary"]["llm_judged_count"] = 0
    payload["summary"]["mean_llm_overall_score"] = None

    artifact = EvaluationMetricsArtifact.model_validate(payload)

    assert artifact.records[0].llm_judge is not None


def test_evaluation_metrics_artifact_rejects_score_out_of_range() -> None:
    payload = _metrics_payload()
    payload["records"][0]["deterministic_score"] = 10.1

    with pytest.raises(ValidationError):
        EvaluationMetricsArtifact.model_validate(payload)


def test_evaluation_metrics_artifact_rejects_unknown_top_level_field() -> None:
    payload = _metrics_payload()
    payload["scored_conversations"] = []

    with pytest.raises(ValidationError):
        EvaluationMetricsArtifact.model_validate(payload)
