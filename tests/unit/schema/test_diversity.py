from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from kg_mle.schema import DiversityReportArtifact


def _summary() -> dict:
    return {
        "conversation_count": 2,
        "mean_deterministic_score": 10.0,
        "mean_chain_completion": 1.0,
        "mean_tool_response_coverage": 1.0,
        "schema_valid_rate": 1.0,
        "llm_judged_count": 0,
        "mean_llm_overall_score": None,
        "usable_for_training_rate": 1.0,
    }


def _repair_summary() -> dict:
    return {
        "enabled": False,
        "attempted": 0,
        "repaired": 0,
        "failed": 0,
        "rejected": 0,
        "regenerated": 0,
        "status_counts": {},
    }


def _diversity() -> dict:
    return {
        "conversation_count": 2,
        "domain_entropy": 1.0,
        "endpoint_coverage_ratio": 0.5,
        "tool_coverage_ratio": 0.5,
        "distinct_endpoint_pair_ratio": 1.0,
        "domain_pattern_diversity": 2,
        "top_endpoint_share": 0.5,
        "chain_length_distribution": {"2": 2},
        "domain_counts": {"finance": 2, "events": 2},
        "endpoint_counts": {"finance/get_quote": 1, "events/create_calendar_event": 1},
        "endpoint_pair_counts": {"finance/get_quote->events/create_calendar_event": 1},
        "domain_pattern_counts": {"finance->events": 1},
    }


def _run(steering_enabled: bool) -> dict:
    return {
        "generation": {
            "steering_enabled": steering_enabled,
            "requested": 2,
            "generated": 2,
            "failures": 0,
            "plan_meta": {"seed": 42},
            "counters_summary": {"endpoint_count": 2},
        },
        "diversity": _diversity(),
        "quality": _summary(),
        "repair_summary": _repair_summary(),
    }


def _report_payload() -> dict:
    return {
        "config": {
            "count": 2,
            "seed": 42,
            "input_path": "data/sample_toolbench/tools.json",
            "repair": False,
            "llm_judge_enabled": False,
            "max_llm_judge_records": None,
        },
        "run_a_no_steering": _run(False),
        "run_b_steering": _run(True),
        "comparison": {
            "domain_entropy_delta": 0.1,
            "endpoint_coverage_delta": 0.1,
            "tool_coverage_delta": 0.1,
            "endpoint_pair_diversity_delta": -0.1,
            "top_endpoint_share_delta": -0.1,
            "mean_deterministic_score_delta": 0.0,
            "usable_for_training_rate_delta": 0.0,
        },
        "artifacts": {
            "run_a_dataset": "artifacts/diversity/run_a_no_steering.jsonl",
            "run_b_dataset": "artifacts/diversity/run_b_steering.jsonl",
            "run_a_metrics": "artifacts/diversity/run_a_metrics.json",
            "run_b_metrics": "artifacts/diversity/run_b_metrics.json",
            "run_a_scored": "artifacts/diversity/run_a_scored.jsonl",
            "run_b_scored": "artifacts/diversity/run_b_scored.jsonl",
        },
    }


def test_diversity_report_artifact_accepts_current_shape() -> None:
    report = DiversityReportArtifact.model_validate(_report_payload())

    assert report.run_b_steering.generation.steering_enabled is True
    assert report.comparison.endpoint_pair_diversity_delta == -0.1


def test_diversity_report_rejects_missing_run() -> None:
    payload = _report_payload()
    del payload["run_b_steering"]

    with pytest.raises(ValidationError):
        DiversityReportArtifact.model_validate(payload)


def test_diversity_report_rejects_invalid_rate() -> None:
    payload = copy.deepcopy(_report_payload())
    payload["run_a_no_steering"]["diversity"]["endpoint_coverage_ratio"] = 1.5

    with pytest.raises(ValidationError):
        DiversityReportArtifact.model_validate(payload)
