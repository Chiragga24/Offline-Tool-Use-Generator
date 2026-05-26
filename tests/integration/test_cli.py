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


def test_build_command_writes_registry(tmp_path):
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
    assert "built registry" in result.output
    assert (tmp_path / "registry.json").exists()


def test_generate_command_exposes_steering_toggle():
    result = runner.invoke(
        app,
        ["generate", "--count", "1", "--seed", "7", "--no-cross-conversation-steering"],
    )

    assert result.exit_code == 0
    assert "cross_conversation_steering=False" in result.output

