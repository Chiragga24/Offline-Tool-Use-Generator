from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from kg_mle import __version__
from kg_mle.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_EVALUATION_PATH,
    DEFAULT_GRAPH_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_INPUT_PATH,
    DEFAULT_LLM_CONFIG,
    DEFAULT_MEM0_LLM_CONFIG,
    DEFAULT_REGISTRY_PATH,
    DEFAULT_SEMANTIC_BACKEND,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_SEMANTIC_TOP_K,
)
from kg_mle.evaluation import (
    LLMJudge,
    evaluate_dataset,
    load_conversations_jsonl,
    save_evaluation,
    save_scored_conversations,
)
from kg_mle.executor import OfflineExecutor
from kg_mle.generator import (
    ConversationCoordinator,
    DeterministicAssistant,
    DeterministicPlanner,
    DeterministicUser,
    GeneratorConfig,
)
from kg_mle.graph import build_tool_graph, save_tool_graph
from kg_mle.graph.semantic import Mem0SemanticRetriever, SentenceTransformerSemanticRetriever
from kg_mle.llm import StructuredLLMClient
from kg_mle.registry import HuggingFaceRegistryEnricher, enrich_registry, load_registry, save_registry
from kg_mle.repair import LLMRepairPlanner, RepairPolicy
from kg_mle.sampler import CorpusPlanner, ToolChainSampler
from kg_mle.utils.logging import configure_logging, get_logger
from kg_mle.utils.paths import ensure_dir, ensure_parent_dir


app = typer.Typer(
    name="kgmle",
    help="Offline ToolBench-style synthetic multi-agent tool-use conversation generator.",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


class _CliState:
    use_llm: bool = False


cli_state = _CliState()


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
    use_llm: Annotated[
        bool,
        typer.Option(
            "--use-llm/--no-use-llm",
            help="Enable optional hosted-LLM features for commands that support them.",
        ),
    ] = False,
) -> None:
    """Run the KG MLE synthetic data pipeline."""
    cli_state.use_llm = use_llm
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
    semantic_graph: Annotated[
        bool,
        typer.Option(
            "--semantic-graph/--no-semantic-graph",
            help="Enable optional Mem0 semantic graph expansion.",
        ),
    ] = False,
    semantic_backend: Annotated[
        str,
        typer.Option("--semantic-backend", help="Semantic backend: local or mem0."),
    ] = DEFAULT_SEMANTIC_BACKEND,
    semantic_threshold: Annotated[
        float,
        typer.Option("--semantic-threshold", help="Minimum semantic edge score."),
    ] = DEFAULT_SEMANTIC_THRESHOLD,
    semantic_top_k: Annotated[
        int,
        typer.Option("--semantic-top-k", min=1, help="Semantic neighbors per endpoint."),
    ] = DEFAULT_SEMANTIC_TOP_K,
    enrich_registry_fields: Annotated[
        bool,
        typer.Option(
            "--enrich-registry-fields/--no-enrich-registry-fields",
            help="Apply deterministic registry field alias/type enrichment.",
        ),
    ] = True,
    llm_enrich_registry: Annotated[
        bool,
        typer.Option(
            "--llm-enrich-registry/--no-llm-enrich-registry",
            help="Use configured LLM for structured registry alias/type enrichment.",
        ),
    ] = False,
    registry_enrichment_threshold: Annotated[
        float,
        typer.Option("--registry-enrichment-threshold", help="Minimum LLM enrichment confidence."),
    ] = 0.80,
    max_llm_registry_endpoints: Annotated[
        int,
        typer.Option(
            "--max-llm-registry-endpoints",
            min=1,
            help="Maximum endpoints to send to live LLM registry enrichment.",
        ),
    ] = 5,
) -> None:
    """Ingest ToolBench-style definitions and build derived artifacts."""
    ensure_dir(artifacts_dir)
    logger.info("Build command starting")
    registry = load_registry(input_path)
    enrichment_report = None
    if enrich_registry_fields:
        registry_enricher = None
        if llm_enrich_registry:
            if DEFAULT_LLM_CONFIG.provider != "huggingface":
                raise typer.BadParameter(
                    "registry LLM enrichment currently supports KG_MLE_LLM_PROVIDER=huggingface"
                )
            registry_enricher = HuggingFaceRegistryEnricher(
                model=DEFAULT_LLM_CONFIG.model,
                api_key=DEFAULT_LLM_CONFIG.api_key,
                provider=DEFAULT_LLM_CONFIG.extra.get("hf_provider"),
            )
        enrichment_report = enrich_registry(
            registry,
            enricher=registry_enricher,
            confidence_threshold=registry_enrichment_threshold,
            max_llm_endpoints=max_llm_registry_endpoints if registry_enricher else None,
        )
    registry_path = artifacts_dir / DEFAULT_REGISTRY_PATH.name
    save_registry(registry, registry_path)
    semantic_retriever = None
    if semantic_graph:
        if semantic_backend == "mem0":
            semantic_retriever = Mem0SemanticRetriever(
                embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                llm_provider=DEFAULT_MEM0_LLM_CONFIG.provider,
                llm_model=DEFAULT_MEM0_LLM_CONFIG.model,
                llm_api_key=DEFAULT_MEM0_LLM_CONFIG.api_key,
                llm_base_url=DEFAULT_MEM0_LLM_CONFIG.base_url,
            )
        elif semantic_backend == "local":
            semantic_retriever = SentenceTransformerSemanticRetriever(model_name=DEFAULT_EMBEDDING_MODEL)
        else:
            raise typer.BadParameter("semantic backend must be one of: local, mem0")
    graph = build_tool_graph(
        registry,
        semantic_retriever=semantic_retriever,
        semantic_threshold=semantic_threshold,
        semantic_top_k=semantic_top_k,
    )
    graph_path = artifacts_dir / DEFAULT_GRAPH_PATH.name
    save_tool_graph(graph, graph_path)
    console.print(
        "[green]built artifacts[/green]: "
        f"tools={len(registry.tools)} endpoints={registry.endpoint_count()} "
        f"nodes={graph.node_count()} edges={graph.edge_count()} "
        f"registry_enrichments={len(enrichment_report.accepted) if enrichment_report else 0} "
        f"registry={registry_path} graph={graph_path}"
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
    registry = load_registry(DEFAULT_INPUT_PATH)
    enrich_registry(registry)
    graph = build_tool_graph(registry)
    sampler = ToolChainSampler(graph)
    report = CorpusPlanner(
        sampler,
        steering_enabled=cross_conversation_steering,
        seed=seed,
    ).sample_corpus(count)
    executor = OfflineExecutor(registry)
    config = GeneratorConfig()
    coordinator = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=DeterministicPlanner(registry, config=config),
        user_simulator=DeterministicUser(),
        assistant=DeterministicAssistant(),
        config=config,
    )
    with output.open("w", encoding="utf-8") as handle:
        for idx, chain in enumerate(report.results):
            conversation = coordinator.run(
                chain,
                seed=chain.seed,
                conversation_id=f"conv_{seed}_{idx:05d}",
            )
            handle.write(conversation.model_dump_json() + "\n")
    console.print(
        "[green]generated dataset[/green]: "
        f"requested={count} records={len(report.results)} failures={len(report.failures)} "
        f"seed={seed} output={output} "
        f"cross_conversation_steering={cross_conversation_steering}"
    )


@app.command()
def evaluate(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", exists=True, file_okay=True, readable=True, help="Generated dataset JSONL path."),
    ] = DEFAULT_DATASET_PATH,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Evaluation metrics JSON path."),
    ] = DEFAULT_EVALUATION_PATH,
    llm_judge: Annotated[
        bool,
        typer.Option(
            "--llm-judge/--no-llm-judge",
            help="Use the configured hosted LLM as an optional judge.",
        ),
    ] = False,
    max_llm_judge_records: Annotated[
        int,
        typer.Option(
            "--max-llm-judge-records",
            min=1,
            help="Maximum records to send to the live LLM judge.",
        ),
    ] = 10,
    scored_output: Annotated[
        Path | None,
        typer.Option(
            "--scored-output",
            help="Optional JSONL path for conversations with metadata.evaluation populated.",
        ),
    ] = None,
    repair: Annotated[
        bool,
        typer.Option(
            "--repair/--no-repair",
            help="Attempt one bounded deterministic repair pass for failed or low-scoring records.",
        ),
    ] = False,
    repair_threshold: Annotated[
        float,
        typer.Option("--repair-threshold", min=0.0, max=10.0, help="Deterministic score repair threshold."),
    ] = 8.0,
    max_repair_attempts: Annotated[
        int,
        typer.Option("--max-repair-attempts", min=0, max=1, help="Maximum repair attempts per record."),
    ] = 1,
) -> None:
    """Validate generated conversations and compute evaluation metrics."""
    ensure_parent_dir(output)
    logger.info("Evaluate command starting")
    conversations = load_conversations_jsonl(input_path)
    judge = None
    use_llm_for_judge = cli_state.use_llm or llm_judge
    use_llm_for_repair = repair and cli_state.use_llm
    llm_client = None
    if use_llm_for_judge or use_llm_for_repair:
        if not DEFAULT_LLM_CONFIG.api_key and DEFAULT_LLM_CONFIG.provider not in {"lmstudio", "vllm"}:
            raise typer.BadParameter(
                f"LLM features require {DEFAULT_LLM_CONFIG.api_key_env} "
                f"for provider {DEFAULT_LLM_CONFIG.provider!r}."
            )
        llm_client = StructuredLLMClient.from_config(DEFAULT_LLM_CONFIG)
    if use_llm_for_judge and llm_client is not None:
        judge = LLMJudge(llm_client)
    repair_planner = (
        LLMRepairPlanner(client=llm_client)
        if use_llm_for_repair and llm_client is not None
        else None
    )
    evaluation = evaluate_dataset(
        conversations,
        judge=judge,
        max_judged_records=max_llm_judge_records if judge else None,
        repair=repair,
        repair_policy=RepairPolicy(deterministic_threshold=repair_threshold),
        repair_planner=repair_planner,
        max_repair_attempts=max_repair_attempts,
    )
    save_evaluation(evaluation, output)
    scored_output_path = scored_output or output.with_name(f"{output.stem}_scored.jsonl")
    save_scored_conversations(evaluation, scored_output_path)
    summary = evaluation["summary"]
    console.print(
        "[green]evaluated dataset[/green]: "
        f"records={summary['conversation_count']} "
        f"mean_score={summary['mean_deterministic_score']} "
        f"llm_judged={summary['llm_judged_count']} "
        f"use_llm={use_llm_for_judge} "
        f"repair_attempted={evaluation['repair_summary']['attempted']} "
        f"output={output} scored_output={scored_output_path}"
    )
