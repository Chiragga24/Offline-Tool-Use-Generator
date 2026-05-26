import json

from typer.testing import CliRunner

from kg_mle import __version__
from kg_mle.cli import app
from kg_mle.repair.models import RepairPlan


runner = CliRunner()


def test_cli_help_shows_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "build" in result.output
    assert "generate" in result.output
    assert "evaluate" in result.output
    assert "diversity" in result.output
    assert "--use-llm" in result.output


def test_cli_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_build_command_writes_registry_and_graph(tmp_path):
    result = runner.invoke(
        app,
        [
            "build",
            "--input",
            "data/sample_toolbench/tools.json",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "built artifacts" in result.output
    assert "registry_enrichments=" in result.output
    assert (tmp_path / "registry.json").exists()
    assert (tmp_path / "tool_graph.json").exists()


def test_build_use_llm_enables_provider_neutral_registry_enrichment(tmp_path, monkeypatch):
    class _FakeConfig:
        provider = "gemini"
        api_key_env = "GOOGLE_API_KEY"
        api_key = "test-key"

    class _FakeEnricher:
        used = False

        def __init__(self, client):
            _FakeEnricher.used = True

        def suggest(self, endpoint):
            return []

    class _FakeClient:
        @classmethod
        def from_config(cls, config):
            return cls()

    monkeypatch.setattr("kg_mle.cli.DEFAULT_LLM_CONFIG", _FakeConfig())
    monkeypatch.setattr("kg_mle.cli.StructuredLLMClient", _FakeClient)
    monkeypatch.setattr("kg_mle.cli.StructuredLLMRegistryEnricher", _FakeEnricher)
    monkeypatch.setattr("kg_mle.cli._semantic_retriever", lambda backend: None)

    result = runner.invoke(
        app,
        [
            "--use-llm",
            "build",
            "--input",
            "data/sample_toolbench/tools.json",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert _FakeEnricher.used is True
    assert "semantic_edges=0" in result.output


def test_generate_command_writes_dataset_and_exposes_steering_toggle(tmp_path):
    output = tmp_path / "conversations.jsonl"
    result = runner.invoke(
        app,
        [
            "generate",
            "--count",
            "1",
            "--seed",
            "7",
            "--output",
            str(output),
            "--no-cross-conversation-steering",
        ],
    )

    assert result.exit_code == 0
    assert "generated dataset" in result.output
    assert "cross_conversation_steering=False" in result.output
    assert output.exists()
    assert len(output.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_generate_uses_built_artifacts_and_allows_semantic_edges(tmp_path):
    build_result = runner.invoke(
        app,
        [
            "build",
            "--input",
            "data/sample_toolbench/tools.json",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )
    assert build_result.exit_code == 0
    output = tmp_path / "conversations.jsonl"

    result = runner.invoke(
        app,
        [
            "generate",
            "--count",
            "1",
            "--seed",
            "7",
            "--artifacts-dir",
            str(tmp_path),
            "--allow-semantic-edges",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "semantic_edges_allowed=True" in result.output
    assert output.exists()


def test_global_use_llm_wires_llm_agents_into_generate(tmp_path, monkeypatch):
    """`--use-llm generate` must actually swap to LLM-backed agents.

    Without this, the global flag steers build (LLM enrichment) and
    evaluate (LLM judge / repair) but silently no-ops on generate —
    which would produce deterministic conversations while logging
    nothing to warn the reviewer.
    """
    from kg_mle.generator import (
        DeterministicAssistant,
        DeterministicPlanner,
        DeterministicUser,
    )

    class _FakeConfig:
        provider = "gemini"
        api_key_env = "GOOGLE_API_KEY"
        api_key = "test-key"

    class _FakeClient:
        @classmethod
        def from_config(cls, config):
            return cls()

    class _FakeAgentsBuilder:
        called_with: dict = {}

        def __call__(self, *, client, registry, config):
            _FakeAgentsBuilder.called_with = {
                "client": client,
                "registry": registry,
                "config": config,
            }
            # Return deterministic agents so the conversation still
            # generates; the test only cares that the LLM-agent code
            # path was taken.
            return (
                DeterministicPlanner(registry, config=config),
                DeterministicUser(),
                DeterministicAssistant(),
            )

    fake_builder = _FakeAgentsBuilder()
    monkeypatch.setattr("kg_mle.cli.DEFAULT_LLM_CONFIG", _FakeConfig())
    monkeypatch.setattr("kg_mle.cli.StructuredLLMClient", _FakeClient)
    monkeypatch.setattr("kg_mle.cli.make_llm_agents", fake_builder)
    monkeypatch.setattr("kg_mle.cli._semantic_retriever", lambda backend: None)

    output = tmp_path / "conversations.jsonl"
    result = runner.invoke(
        app,
        [
            "--use-llm",
            "generate",
            "--count",
            "1",
            "--seed",
            "7",
            "--output",
            str(output),
            "--no-cross-conversation-steering",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "agents=llm-with-fallback" in result.output
    assert output.exists()
    assert len(output.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert _FakeAgentsBuilder.called_with, (
        "make_llm_agents was not invoked; --use-llm did not reach generate's coordinator."
    )
    assert _FakeAgentsBuilder.called_with["registry"] is not None
    assert isinstance(_FakeAgentsBuilder.called_with["client"], _FakeClient)


def test_generate_without_use_llm_reports_deterministic_agents(tmp_path):
    output = tmp_path / "conversations.jsonl"
    result = runner.invoke(
        app,
        [
            "generate",
            "--count",
            "1",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "agents=deterministic" in result.output


def test_evaluate_command_writes_metrics(tmp_path):
    dataset = tmp_path / "conversations.jsonl"
    output = tmp_path / "metrics.json"
    scored_output = tmp_path / "scored.jsonl"
    conversation = {
        "conversation_id": "conv_cli",
        "messages": [
            {"role": "user", "content": "Find a stock quote."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "endpoint_id": "finance/get_quote",
                        "arguments": {"symbol": "SAP"},
                        "call_confidence": 1.0,
                    }
                ],
            },
            {
                "role": "tool",
                "endpoint": "finance/get_quote",
                "content": {"symbol": "SAP", "price": "199.25"},
            },
            {"role": "assistant", "content": "SAP is trading at 199.25."},
        ],
        "plan": {
            "conversation_intent": "Find a stock quote.",
            "user_character": "default",
            "plan_confidence": 1.0,
            "step_plans": [
                {
                    "step_index": 0,
                    "endpoint_id": "finance/get_quote",
                    "parameter_plans": [],
                }
            ],
            "ambiguous_step_indices": [],
        },
        "metadata": {"n_tool_calls": 1},
    }
    dataset.write_text(json.dumps(conversation) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--input",
            str(dataset),
            "--output",
            str(output),
            "--scored-output",
            str(scored_output),
        ],
    )

    assert result.exit_code == 0
    assert "evaluated dataset" in result.output
    assert output.exists()
    assert scored_output.exists()
    scored = json.loads(scored_output.read_text(encoding="utf-8").strip())
    assert scored["metadata"]["evaluation"]["deterministic_score"] == 10.0


def test_global_use_llm_can_enable_evaluate_judge(tmp_path, monkeypatch):
    dataset = tmp_path / "conversations.jsonl"
    output = tmp_path / "metrics.json"
    conversation = {
        "conversation_id": "conv_cli_llm",
        "messages": [{"role": "user", "content": "Hello."}],
        "plan": {
            "conversation_intent": "Greet.",
            "user_character": "default",
            "plan_confidence": 1.0,
            "step_plans": [],
            "ambiguous_step_indices": [],
        },
        "metadata": {"n_tool_calls": 0},
    }
    dataset.write_text(json.dumps(conversation) + "\n", encoding="utf-8")

    class _FakeConfig:
        provider = "gemini"
        api_key_env = "GOOGLE_API_KEY"
        api_key = "test-key"

    class _FakeJudge:
        def __init__(self, client):
            pass

        def score(self, conversation):
            class _Score:
                def model_dump(self):
                    return {
                        "task_completion": 8.0,
                        "tool_trace_validity": 8.0,
                        "argument_grounding": 8.0,
                        "response_grounding": 8.0,
                        "naturalness": 8.0,
                        "overall_score": 8.0,
                        "confidence": 8.0,
                        "issues": [],
                        "rationale": "Fake score.",
                    }

            return _Score()

    class _FakeClient:
        @classmethod
        def from_config(cls, config):
            return cls()

    monkeypatch.setattr("kg_mle.cli.DEFAULT_LLM_CONFIG", _FakeConfig())
    monkeypatch.setattr("kg_mle.cli.LLMJudge", _FakeJudge)
    monkeypatch.setattr("kg_mle.cli.StructuredLLMClient", _FakeClient)

    result = runner.invoke(
        app,
        [
            "--use-llm",
            "evaluate",
            "--input",
            str(dataset),
            "--output",
            str(output),
            "--max-llm-judge-records",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "use_llm=True" in result.output
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["summary"]["llm_judged_count"] == 1


def test_evaluate_repair_writes_repair_summary_and_history(tmp_path):
    dataset = tmp_path / "conversations.jsonl"
    output = tmp_path / "metrics.json"
    scored_output = tmp_path / "scored.jsonl"
    conversation = {
        "conversation_id": "conv_repair",
        "messages": [
            {"role": "user", "content": "Find a stock quote."},
            {
                "role": "tool",
                "endpoint": "finance/get_quote",
                "content": {"error": {"kind": "ungrounded_argument"}},
            },
        ],
        "plan": {
            "conversation_intent": "Find a stock quote.",
            "user_character": "default",
            "plan_confidence": 1.0,
            "step_plans": [],
            "ambiguous_step_indices": [],
        },
        "metadata": {"n_tool_calls": 1},
    }
    dataset.write_text(json.dumps(conversation) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--input",
            str(dataset),
            "--output",
            str(output),
            "--scored-output",
            str(scored_output),
            "--repair",
        ],
    )

    assert result.exit_code == 0
    assert "repair_attempted=1" in result.output
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["repair_summary"]["attempted"] == 1
    scored = json.loads(scored_output.read_text(encoding="utf-8").strip())
    assert scored["metadata"]["repair_history"][0]["status"] == "failed"


def test_global_use_llm_wires_llm_repair_planner(tmp_path, monkeypatch):
    dataset = tmp_path / "conversations.jsonl"
    output = tmp_path / "metrics.json"
    scored_output = tmp_path / "scored.jsonl"
    conversation = {
        "conversation_id": "conv_llm_repair",
        "messages": [
            {"role": "user", "content": "Find a stock quote."},
            {
                "role": "tool",
                "endpoint": "finance/get_quote",
                "content": {"error": {"kind": "ungrounded_argument"}},
            },
        ],
        "plan": {
            "conversation_intent": "Find a stock quote.",
            "user_character": "default",
            "plan_confidence": 1.0,
            "step_plans": [],
            "ambiguous_step_indices": [],
        },
        "metadata": {"n_tool_calls": 1},
    }
    dataset.write_text(json.dumps(conversation) + "\n", encoding="utf-8")

    class _FakeConfig:
        provider = "gemini"
        api_key_env = "GOOGLE_API_KEY"
        api_key = "test-key"

    class _FakeClient:
        @classmethod
        def from_config(cls, config):
            return cls()

    class _FakeRepairPlanner:
        used = False

        def __init__(self, client):
            pass

        def plan(self, conversation, *, triggers):
            _FakeRepairPlanner.used = True
            return RepairPlan(
                strategy="regenerate_conversation",
                triggers=triggers,
                reason="Fake LLM planner used.",
                requires_coordinator=True,
                confidence=8.0,
            )

        def apply(self, conversation, plan):
            return conversation, "failed"

    monkeypatch.setattr("kg_mle.cli.DEFAULT_LLM_CONFIG", _FakeConfig())
    monkeypatch.setattr("kg_mle.cli.StructuredLLMClient", _FakeClient)
    monkeypatch.setattr("kg_mle.cli.LLMRepairPlanner", _FakeRepairPlanner)

    result = runner.invoke(
        app,
        [
            "--use-llm",
            "evaluate",
            "--input",
            str(dataset),
            "--output",
            str(output),
            "--scored-output",
            str(scored_output),
            "--repair",
            "--max-llm-judge-records",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert _FakeRepairPlanner.used is True
    scored = json.loads(scored_output.read_text(encoding="utf-8").strip())
    assert scored["metadata"]["repair_history"][0]["plan"]["reason"] == "Fake LLM planner used."


def test_diversity_command_writes_expected_artifacts(tmp_path):
    output_dir = tmp_path / "diversity"

    result = runner.invoke(
        app,
        [
            "diversity",
            "--count",
            "4",
            "--seed",
            "5",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "diversity experiment complete" in result.output
    expected = [
        "run_a_no_steering.jsonl",
        "run_b_steering.jsonl",
        "run_a_metrics.json",
        "run_b_metrics.json",
        "run_a_scored.jsonl",
        "run_b_scored.jsonl",
        "diversity_report.json",
    ]
    for name in expected:
        assert (output_dir / name).exists(), f"missing {name}"
    report = json.loads((output_dir / "diversity_report.json").read_text(encoding="utf-8"))
    assert report["config"]["count"] == 4
    assert "endpoint_coverage_delta" in report["comparison"]


def test_diversity_use_llm_wires_provider_neutral_judge(tmp_path, monkeypatch):
    output_dir = tmp_path / "diversity_llm"

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
            class _Score:
                def model_dump(self):
                    return {
                        "task_completion": 10.0,
                        "tool_trace_validity": 10.0,
                        "argument_grounding": 10.0,
                        "response_grounding": 10.0,
                        "naturalness": 10.0,
                        "overall_score": 10.0,
                        "confidence": 10.0,
                        "issues": [],
                        "rationale": "Fake diversity judge score.",
                    }

            return _Score()

    class _FakeEnricher:
        def __init__(self, client):
            pass

        def suggest(self, endpoint):
            return []

    monkeypatch.setattr("kg_mle.cli.DEFAULT_LLM_CONFIG", _FakeConfig())
    monkeypatch.setattr("kg_mle.cli.StructuredLLMClient", _FakeClient)
    monkeypatch.setattr("kg_mle.cli.LLMJudge", _FakeJudge)
    monkeypatch.setattr("kg_mle.cli.StructuredLLMRegistryEnricher", _FakeEnricher)
    monkeypatch.setattr("kg_mle.diversity.experiment._semantic_retriever", lambda config: None)

    result = runner.invoke(
        app,
        [
            "--use-llm",
            "diversity",
            "--count",
            "2",
            "--seed",
            "5",
            "--output-dir",
            str(output_dir),
            "--max-llm-judge-records",
            "1",
        ],
    )

    assert result.exit_code == 0
    report = json.loads((output_dir / "diversity_report.json").read_text(encoding="utf-8"))
    assert report["config"]["llm_judge_enabled"] is True
    assert report["config"]["semantic_graph"] is True
    assert report["config"]["allow_semantic_edges"] is True
    assert report["config"]["llm_registry_enrichment"] is True
    assert report["config"]["llm_generation_enabled"] is True
    assert report["run_a_no_steering"]["generation"]["llm_generation"] is True
    assert report["run_b_steering"]["generation"]["llm_generation"] is True
    assert report["run_a_no_steering"]["quality"]["llm_judged_count"] == 1
    assert report["run_b_steering"]["quality"]["llm_judged_count"] == 1


def test_diversity_use_llm_invokes_llm_generator_agents(tmp_path, monkeypatch):
    """--use-llm diversity must route both runs through LLM generator agents
    (with deterministic fallback), not silently stay deterministic."""
    from kg_mle.generator import (
        DeterministicAssistant,
        DeterministicPlanner,
        DeterministicUser,
    )

    output_dir = tmp_path / "diversity_llm_gen"

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
            class _Score:
                def model_dump(self):
                    return {
                        "task_completion": 9.0,
                        "tool_trace_validity": 9.0,
                        "argument_grounding": 9.0,
                        "response_grounding": 9.0,
                        "naturalness": 9.0,
                        "overall_score": 9.0,
                        "confidence": 9.0,
                        "issues": [],
                        "rationale": "fake",
                    }

            return _Score()

    class _FakeEnricher:
        def __init__(self, client):
            pass

        def suggest(self, endpoint):
            return []

    builder_calls = {"count": 0, "clients": []}

    def fake_make_llm_agents(*, client, registry, config):
        builder_calls["count"] += 1
        builder_calls["clients"].append(client)
        return (
            DeterministicPlanner(registry, config=config),
            DeterministicUser(),
            DeterministicAssistant(),
        )

    monkeypatch.setattr("kg_mle.cli.DEFAULT_LLM_CONFIG", _FakeConfig())
    monkeypatch.setattr("kg_mle.cli.StructuredLLMClient", _FakeClient)
    monkeypatch.setattr("kg_mle.cli.LLMJudge", _FakeJudge)
    monkeypatch.setattr("kg_mle.cli.StructuredLLMRegistryEnricher", _FakeEnricher)
    monkeypatch.setattr("kg_mle.diversity.experiment._semantic_retriever", lambda config: None)
    monkeypatch.setattr("kg_mle.diversity.experiment.make_llm_agents", fake_make_llm_agents)

    result = runner.invoke(
        app,
        [
            "--use-llm",
            "diversity",
            "--count",
            "2",
            "--seed",
            "5",
            "--output-dir",
            str(output_dir),
            "--max-llm-judge-records",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "llm_generation=True" in result.output
    # make_llm_agents is built once per run (Run A and Run B) = 2 invocations.
    assert builder_calls["count"] == 2
    # Both runs received the same single client the CLI constructed.
    assert isinstance(builder_calls["clients"][0], _FakeClient)


def test_diversity_default_keeps_deterministic_generation(tmp_path):
    """Without --use-llm, both runs report llm_generation=False."""
    output_dir = tmp_path / "diversity_det"
    result = runner.invoke(
        app,
        ["diversity", "--count", "2", "--seed", "5", "--output-dir", str(output_dir)],
    )
    assert result.exit_code == 0
    report = json.loads((output_dir / "diversity_report.json").read_text(encoding="utf-8"))
    assert report["config"]["llm_generation_enabled"] is False
    assert report["run_a_no_steering"]["generation"]["llm_generation"] is False
    assert report["run_b_steering"]["generation"]["llm_generation"] is False
