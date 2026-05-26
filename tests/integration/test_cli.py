import json

from typer.testing import CliRunner

from kg_mle import __version__
from kg_mle.cli import app


runner = CliRunner()


def test_cli_help_shows_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "build" in result.output
    assert "generate" in result.output
    assert "evaluate" in result.output


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
