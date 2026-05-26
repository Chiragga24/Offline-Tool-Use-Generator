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
)
from kg_mle.graph import build_tool_graph
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


def run_diversity_experiment(config: DiversityRunConfig) -> dict[str, Any]:
    ensure_dir(config.output_dir)
    registry = load_registry(config.input_path)
    enrich_registry(registry)
    graph = build_tool_graph(registry)
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
        output_path=config.output_dir / "run_a_no_steering.jsonl",
    )
    run_b = _generate_run(
        registry=registry,
        graph=graph,
        sampler=sampler,
        count=config.count,
        seed=config.seed,
        steering_enabled=True,
        output_path=config.output_dir / "run_b_steering.jsonl",
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
            "max_llm_judge_records": config.max_llm_judge_records,
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
    output_path: Path,
) -> dict[str, Any]:
    report = CorpusPlanner(
        sampler,
        steering_enabled=steering_enabled,
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
            "requested": count,
            "generated": len(report.results),
            "failures": len(report.failures),
            "plan_meta": report.plan_meta,
            "counters_summary": report.counters_summary,
        },
    }


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
