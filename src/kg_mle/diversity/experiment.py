"""Run steering-on/off diversity experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kg_mle.diversity.metrics import compute_diversity_metrics
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
    make_llm_agents,
)
from kg_mle.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_MEM0_LLM_CONFIG,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_SEMANTIC_TOP_K,
)
from kg_mle.graph import build_tool_graph
from kg_mle.graph.semantic import Mem0SemanticRetriever, SentenceTransformerSemanticRetriever
from kg_mle.registry import enrich_registry, load_registry
from kg_mle.sampler import CorpusPlanner, ToolChainSampler
from kg_mle.utils.paths import ensure_dir


@dataclass(frozen=True)
class DiversityRunConfig:
    count: int
    seed: int
    output_dir: Path
    input_path: Path
    repair: bool = False
    judge: LLMJudge | None = None
    max_llm_judge_records: int | None = None
    semantic_graph: bool = False
    semantic_backend: str = "local"
    allow_semantic_edges: bool = False
    registry_enricher: Any | None = None
    llm_registry_enrichment: bool = False
    generator_client: Any | None = None
    """Optional StructuredLLMClient. When set, both runs use LLM generator
    agents (with deterministic fallback per turn). When None, both runs use
    deterministic agents — the reproducible default. Diversity metrics are
    chain-based (from the sampler), so they primarily reflect steering either
    way; LLM generation mainly changes the quality metrics and any
    LLM-accepted chain deviations."""


def run_diversity_experiment(config: DiversityRunConfig) -> dict[str, Any]:
    ensure_dir(config.output_dir)
    registry = load_registry(config.input_path)
    enrich_registry(
        registry,
        enricher=config.registry_enricher,
        max_llm_endpoints=5 if config.registry_enricher else None,
    )
    graph = build_tool_graph(
        registry,
        semantic_retriever=_semantic_retriever(config) if config.semantic_graph else None,
        semantic_threshold=DEFAULT_SEMANTIC_THRESHOLD,
        semantic_top_k=DEFAULT_SEMANTIC_TOP_K,
    )
    sampler = ToolChainSampler(graph)
    total_endpoints = registry.endpoint_count()
    total_tools = len(registry.tools)
    endpoint_to_tool = {
        endpoint.endpoint_id: endpoint.tool_name for endpoint in registry.endpoints
    }

    run_a = _generate_run(
        registry=registry,
        graph=graph,
        sampler=sampler,
        count=config.count,
        seed=config.seed,
        steering_enabled=False,
        allow_semantic_edges=config.allow_semantic_edges,
        output_path=config.output_dir / "run_a_no_steering.jsonl",
        llm_client=config.generator_client,
    )
    run_b = _generate_run(
        registry=registry,
        graph=graph,
        sampler=sampler,
        count=config.count,
        seed=config.seed,
        steering_enabled=True,
        allow_semantic_edges=config.allow_semantic_edges,
        output_path=config.output_dir / "run_b_steering.jsonl",
        llm_client=config.generator_client,
    )

    eval_a = evaluate_dataset(
        run_a["conversations"],
        repair=config.repair,
        judge=config.judge,
        max_judged_records=config.max_llm_judge_records,
    )
    eval_b = evaluate_dataset(
        run_b["conversations"],
        repair=config.repair,
        judge=config.judge,
        max_judged_records=config.max_llm_judge_records,
    )
    save_evaluation(eval_a, config.output_dir / "run_a_metrics.json")
    save_evaluation(eval_b, config.output_dir / "run_b_metrics.json")
    save_scored_conversations(eval_a, config.output_dir / "run_a_scored.jsonl")
    save_scored_conversations(eval_b, config.output_dir / "run_b_scored.jsonl")

    diversity_a = compute_diversity_metrics(
        eval_a["scored_conversations"],
        total_endpoints=total_endpoints,
        total_tools=total_tools,
        endpoint_to_tool=endpoint_to_tool,
    )
    diversity_b = compute_diversity_metrics(
        eval_b["scored_conversations"],
        total_endpoints=total_endpoints,
        total_tools=total_tools,
        endpoint_to_tool=endpoint_to_tool,
    )
    report = {
        "config": {
            "count": config.count,
            "seed": config.seed,
            "input_path": str(config.input_path),
            "repair": config.repair,
            "llm_judge_enabled": config.judge is not None,
            "llm_generation_enabled": config.generator_client is not None,
            "max_llm_judge_records": config.max_llm_judge_records,
            "semantic_graph": config.semantic_graph,
            "semantic_backend": config.semantic_backend,
            "allow_semantic_edges": config.allow_semantic_edges,
            "llm_registry_enrichment": config.llm_registry_enrichment,
        },
        "run_a_no_steering": {
            "generation": run_a["generation"],
            "diversity": diversity_a,
            "quality": eval_a["summary"],
            "repair_summary": eval_a["repair_summary"],
        },
        "run_b_steering": {
            "generation": run_b["generation"],
            "diversity": diversity_b,
            "quality": eval_b["summary"],
            "repair_summary": eval_b["repair_summary"],
        },
        "comparison": _compare(diversity_a, diversity_b, eval_a["summary"], eval_b["summary"]),
        "artifacts": {
            "run_a_dataset": str(config.output_dir / "run_a_no_steering.jsonl"),
            "run_b_dataset": str(config.output_dir / "run_b_steering.jsonl"),
            "run_a_metrics": str(config.output_dir / "run_a_metrics.json"),
            "run_b_metrics": str(config.output_dir / "run_b_metrics.json"),
            "run_a_scored": str(config.output_dir / "run_a_scored.jsonl"),
            "run_b_scored": str(config.output_dir / "run_b_scored.jsonl"),
        },
    }
    report_path = config.output_dir / "diversity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _generate_run(
    *,
    registry,
    graph,
    sampler: ToolChainSampler,
    count: int,
    seed: int,
    steering_enabled: bool,
    allow_semantic_edges: bool,
    output_path: Path,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    report = CorpusPlanner(
        sampler,
        steering_enabled=steering_enabled,
        seed=seed,
        allow_semantic_edges=allow_semantic_edges,
    ).sample_corpus(count)
    executor = OfflineExecutor(registry)
    config = GeneratorConfig()
    if llm_client is not None:
        planner, user_simulator, assistant = make_llm_agents(
            client=llm_client, registry=registry, config=config
        )
    else:
        planner = DeterministicPlanner(registry, config=config)
        user_simulator = DeterministicUser()
        assistant = DeterministicAssistant()
    coordinator = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=planner,
        user_simulator=user_simulator,
        assistant=assistant,
        config=config,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, chain in enumerate(report.results):
            conversation = coordinator.run(
                chain,
                seed=chain.seed,
                conversation_id=f"{'steered' if steering_enabled else 'unsteered'}_{seed}_{idx:05d}",
            )
            handle.write(conversation.model_dump_json() + "\n")
    conversations = load_conversations_jsonl(output_path)
    return {
        "conversations": conversations,
        "generation": {
            "steering_enabled": steering_enabled,
            "llm_generation": llm_client is not None,
            "requested": count,
            "generated": len(report.results),
            "failures": len(report.failures),
            "plan_meta": report.plan_meta,
            "counters_summary": report.counters_summary,
        },
    }


def _semantic_retriever(config: DiversityRunConfig):
    if config.semantic_backend == "local":
        return SentenceTransformerSemanticRetriever(model_name=DEFAULT_EMBEDDING_MODEL)
    if config.semantic_backend == "mem0":
        return Mem0SemanticRetriever(
            embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            llm_provider=DEFAULT_MEM0_LLM_CONFIG.provider,
            llm_model=DEFAULT_MEM0_LLM_CONFIG.model,
            llm_api_key=DEFAULT_MEM0_LLM_CONFIG.api_key,
            llm_base_url=DEFAULT_MEM0_LLM_CONFIG.base_url,
        )
    raise ValueError("semantic_backend must be one of: local, mem0")


def _compare(
    diversity_a: dict[str, Any],
    diversity_b: dict[str, Any],
    quality_a: dict[str, Any],
    quality_b: dict[str, Any],
) -> dict[str, Any]:
    return {
        "domain_entropy_delta": round(diversity_b["domain_entropy"] - diversity_a["domain_entropy"], 4),
        "endpoint_coverage_delta": round(
            diversity_b["endpoint_coverage_ratio"] - diversity_a["endpoint_coverage_ratio"], 4
        ),
        "tool_coverage_delta": round(
            diversity_b["tool_coverage_ratio"] - diversity_a["tool_coverage_ratio"], 4
        ),
        "endpoint_pair_diversity_delta": round(
            diversity_b["distinct_endpoint_pair_ratio"] - diversity_a["distinct_endpoint_pair_ratio"],
            4,
        ),
        "top_endpoint_share_delta": round(
            diversity_b["top_endpoint_share"] - diversity_a["top_endpoint_share"], 4
        ),
        "mean_deterministic_score_delta": round(
            quality_b["mean_deterministic_score"] - quality_a["mean_deterministic_score"], 4
        ),
        "usable_for_training_rate_delta": round(
            quality_b["usable_for_training_rate"] - quality_a["usable_for_training_rate"], 4
        ),
    }
