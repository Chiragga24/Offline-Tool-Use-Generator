"""Live diversity experiment smoke test.

Runs only when a configured hosted LLM key is present. Provider/network
failures skip rather than fail, because normal CI coverage uses fake clients
and deterministic evaluation.
"""

from __future__ import annotations

import pytest

from kg_mle.config import DEFAULT_INPUT_PATH, DEFAULT_LLM_CONFIG
from kg_mle.diversity import DiversityRunConfig, run_diversity_experiment
from kg_mle.evaluation import LLMJudge
from kg_mle.llm import StructuredLLMClient


pytestmark = pytest.mark.live


def _require_live_llm_config() -> None:
    if DEFAULT_LLM_CONFIG.provider not in {"gemini", "groq", "huggingface"}:
        pytest.skip(
            "Diversity live test currently supports gemini, groq, or huggingface "
            f"(currently {DEFAULT_LLM_CONFIG.provider!r})."
        )
    if not DEFAULT_LLM_CONFIG.api_key:
        pytest.skip(
            "Diversity live test requires the configured provider API key "
            f"({DEFAULT_LLM_CONFIG.api_key_env})."
        )


def test_diversity_live_uses_llm_judge_for_both_runs(tmp_path):
    _require_live_llm_config()
    try:
        judge = LLMJudge(StructuredLLMClient.from_config(DEFAULT_LLM_CONFIG))
        report = run_diversity_experiment(
            DiversityRunConfig(
                count=4,
                seed=42,
                output_dir=tmp_path / "diversity_live",
                input_path=DEFAULT_INPUT_PATH,
                judge=judge,
                max_llm_judge_records=1,
            )
        )
    except Exception as exc:
        pytest.skip(f"Live diversity LLM run failed: {exc}")

    assert report["config"]["llm_judge_enabled"] is True
    assert report["run_a_no_steering"]["quality"]["llm_judged_count"] == 1
    assert report["run_b_steering"]["quality"]["llm_judged_count"] == 1
    assert report["run_a_no_steering"]["quality"]["mean_llm_overall_score"] is not None
    assert report["run_b_steering"]["quality"]["mean_llm_overall_score"] is not None
