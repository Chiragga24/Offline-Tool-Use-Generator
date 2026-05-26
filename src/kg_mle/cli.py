from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from kg_mle import __version__
from kg_mle.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_EVALUATION_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_REGISTRY_PATH,
)
from kg_mle.registry import load_registry, save_registry
from kg_mle.utils.logging import configure_logging, get_logger
from kg_mle.utils.paths import ensure_dir, ensure_parent_dir


app = typer.Typer(
    name="kgmle",
    help="Offline ToolBench-style synthetic multi-agent tool-use conversation generator.",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


def version_callback(value: bool) -> None:
    if value:
        console.print(f"kg-mle {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, help="Show package version."),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Run the KG MLE synthetic data pipeline."""
    configure_logging(log_level)


@app.command()
def build(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="ToolBench-style JSON file or directory.",
        ),
    ] = DEFAULT_INPUT_PATH,
    artifacts_dir: Annotated[
        Path,
        typer.Option("--artifacts-dir", "-a", help="Directory for derived artifacts."),
    ] = DEFAULT_ARTIFACTS_DIR,
) -> None:
    """Ingest ToolBench-style definitions and build derived artifacts."""
    ensure_dir(artifacts_dir)
    logger.info("Build command starting")
    registry = load_registry(input_path)
    registry_path = artifacts_dir / DEFAULT_REGISTRY_PATH.name
    save_registry(registry, registry_path)
    console.print(
        "[green]built registry[/green]: "
        f"tools={len(registry.tools)} endpoints={registry.endpoint_count()} path={registry_path}"
    )


@app.command()
def generate(
    count: Annotated[int, typer.Option("--count", "-n", min=1, help="Number of records.")] = 10,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output JSONL path."),
    ] = DEFAULT_DATASET_PATH,
    cross_conversation_steering: Annotated[
        bool,
        typer.Option(
            "--cross-conversation-steering/--no-cross-conversation-steering",
            help="Enable or disable corpus-level diversity steering.",
        ),
    ] = True,
) -> None:
    """Generate synthetic tool-use conversations."""
    ensure_parent_dir(output)
    logger.info("Generate command starting")
    console.print(
        "[yellow]generate is scaffolded[/yellow]: "
        f"count={count} seed={seed} output={output} "
        f"cross_conversation_steering={cross_conversation_steering}"
    )


@app.command()
def evaluate(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Generated dataset JSONL path."),
    ] = DEFAULT_DATASET_PATH,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Evaluation metrics JSON path."),
    ] = DEFAULT_EVALUATION_PATH,
) -> None:
    """Validate generated conversations and compute evaluation metrics."""
    ensure_parent_dir(output)
    logger.info("Evaluate command starting")
    console.print(f"[yellow]evaluate is scaffolded[/yellow]: input={input_path} output={output}")
