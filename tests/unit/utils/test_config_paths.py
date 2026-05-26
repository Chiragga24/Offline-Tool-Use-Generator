from pathlib import Path

from typer.testing import CliRunner

from kg_mle import config
from kg_mle.cli import app
from kg_mle.utils.paths import ensure_dir, ensure_parent_dir


runner = CliRunner()


def test_default_paths_are_project_relative():
    assert config.DEFAULT_INPUT_PATH == config.PROJECT_ROOT / "data" / "sample_toolbench" / "tools.json"
    assert config.DEFAULT_ARTIFACTS_DIR == config.PROJECT_ROOT / "artifacts"
    assert config.DEFAULT_DATASET_PATH == config.PROJECT_ROOT / "data" / "outputs" / "conversations.jsonl"


def test_path_helpers_create_directories(tmp_path):
    directory = tmp_path / "nested" / "dir"
    file_path = tmp_path / "outputs" / "result.json"

    assert ensure_dir(directory) == directory
    assert directory.exists()

    assert ensure_parent_dir(file_path) == file_path
    assert file_path.parent.exists()


def test_cli_build_creates_artifacts_dir(tmp_path):
    artifacts_dir = tmp_path / "artifacts"

    result = runner.invoke(
        app,
        [
            "build",
            "--input",
            str(Path("data/sample_toolbench/tools.json")),
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )

    assert result.exit_code == 0
    assert artifacts_dir.exists()

