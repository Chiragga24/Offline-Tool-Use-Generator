from __future__ import annotations

import json

from typer.testing import CliRunner

from kg_mle.cli import app
from kg_mle.schema import EvaluationMetricsArtifact, GeneratedConversationRecord, ScoredConversationRecord


runner = CliRunner()


def test_e2e_build_generate_evaluate_100_with_llm_judge_threshold(tmp_path, monkeypatch):
    """Build artifacts, generate 100 samples, and score them with an LLM-judge adapter.

    The judge is faked at the provider boundary so this E2E test remains
    deterministic and runnable without network credentials. Live provider tests
    cover the real Gemini/Groq path separately.
    """

    class _FakeConfig:
        provider = "gemini"
        api_key_env = "GOOGLE_API_KEY"
        api_key = "test-key"

    class _FakeClient:
        @classmethod
        def from_config(cls, config):
            return cls()

    class _FakeJudge:
        def __init__(self, client):
            pass

        def score(self, conversation):
            tool_calls = sum(
                len(message.get("tool_calls", []))
                for message in conversation.get("messages", [])
                if message.get("role") == "assistant"
            )
            score = 9.0 if tool_calls >= 2 else 8.0

            class _Score:
                def model_dump(self):
                    return {
                        "task_completion": score,
                        "tool_trace_validity": 9.0,
                        "argument_grounding": 9.0,
                        "response_grounding": 9.0,
                        "naturalness": 8.0,
                        "overall_score": score,
                        "confidence": 9.0,
                        "issues": [],
                        "rationale": "Deterministic fake judge for offline E2E.",
                    }

            return _Score()

    monkeypatch.setattr("kg_mle.cli.DEFAULT_LLM_CONFIG", _FakeConfig())
    monkeypatch.setattr("kg_mle.cli.StructuredLLMClient", _FakeClient)
    monkeypatch.setattr("kg_mle.cli.LLMJudge", _FakeJudge)

    artifacts_dir = tmp_path / "artifacts"
    dataset_path = tmp_path / "conversations.jsonl"
    metrics_path = tmp_path / "metrics.json"
    scored_path = tmp_path / "scored.jsonl"

    build = runner.invoke(
        app,
        [
            "build",
            "--input",
            "data/sample_toolbench/tools.json",
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )
    assert build.exit_code == 0, build.output

    generate = runner.invoke(
        app,
        [
            "generate",
            "--count",
            "100",
            "--seed",
            "42",
            "--artifacts-dir",
            str(artifacts_dir),
            "--output",
            str(dataset_path),
        ],
    )
    assert generate.exit_code == 0, generate.output

    evaluate = runner.invoke(
        app,
        [
            "--use-llm",
            "evaluate",
            "--input",
            str(dataset_path),
            "--output",
            str(metrics_path),
            "--scored-output",
            str(scored_path),
            "--max-llm-judge-records",
            "100",
        ],
    )
    assert evaluate.exit_code == 0, evaluate.output

    conversations = [
        GeneratedConversationRecord.model_validate_json(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
    ]
    scored = [
        ScoredConversationRecord.model_validate_json(line)
        for line in scored_path.read_text(encoding="utf-8").splitlines()
    ]
    metrics = EvaluationMetricsArtifact.model_validate_json(
        metrics_path.read_text(encoding="utf-8")
    )

    assert len(conversations) == 100
    assert len(scored) == 100
    assert metrics.summary.conversation_count == 100
    assert metrics.summary.llm_judged_count == 100
    assert metrics.summary.mean_llm_overall_score is not None
    assert metrics.summary.mean_llm_overall_score >= 8.0
    assert metrics.summary.schema_valid_rate == 1.0
    assert metrics.summary.mean_tool_response_coverage >= 0.95

    multi_step_multi_tool = [
        conversation
        for conversation in conversations
        if conversation.metadata["n_tool_calls"] >= 3
        and len(set(conversation.metadata["tools_visited"])) >= 2
    ]
    assert len(multi_step_multi_tool) / len(conversations) >= 0.50

    clarification_turns = [
        message
        for conversation in conversations
        for message in conversation.messages
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
        and "could you tell me" in message["content"].lower()
    ]
    assert clarification_turns

    raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert raw_metrics["summary"]["mean_llm_overall_score"] >= 8.0
