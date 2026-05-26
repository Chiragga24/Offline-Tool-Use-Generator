"""Diversity report artifact schema."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from kg_mle.schema.common import MetricsSummary, RepairSummary, StrictArtifactModel


class DiversityMetrics(StrictArtifactModel):
    conversation_count: int = Field(ge=0)
    domain_entropy: float = Field(ge=0.0)
    endpoint_coverage_ratio: float = Field(ge=0.0, le=1.0)
    tool_coverage_ratio: float = Field(ge=0.0, le=1.0)
    distinct_endpoint_pair_ratio: float = Field(ge=0.0, le=1.0)
    domain_pattern_diversity: int = Field(ge=0)
    top_endpoint_share: float = Field(ge=0.0, le=1.0)
    chain_length_distribution: dict[str, int]
    domain_counts: dict[str, int]
    endpoint_counts: dict[str, int]
    endpoint_pair_counts: dict[str, int]
    domain_pattern_counts: dict[str, int]


class DiversityGenerationSummary(StrictArtifactModel):
    steering_enabled: bool
    requested: int = Field(ge=0)
    generated: int = Field(ge=0)
    failures: int = Field(ge=0)
    plan_meta: dict[str, Any]
    counters_summary: dict[str, Any]


class DiversityRunArtifact(StrictArtifactModel):
    generation: DiversityGenerationSummary
    diversity: DiversityMetrics
    quality: MetricsSummary
    repair_summary: RepairSummary


class DiversityReportConfig(StrictArtifactModel):
    count: int = Field(ge=0)
    seed: int
    input_path: str
    repair: bool
    llm_judge_enabled: bool
    max_llm_judge_records: int | None = Field(default=None, ge=0)
    semantic_graph: bool = False
    semantic_backend: str = "local"
    allow_semantic_edges: bool = False
    llm_registry_enrichment: bool = False


class DiversityComparison(StrictArtifactModel):
    domain_entropy_delta: float
    endpoint_coverage_delta: float
    tool_coverage_delta: float
    endpoint_pair_diversity_delta: float
    top_endpoint_share_delta: float
    mean_deterministic_score_delta: float
    usable_for_training_rate_delta: float


class DiversityArtifacts(StrictArtifactModel):
    run_a_dataset: str
    run_b_dataset: str
    run_a_metrics: str
    run_b_metrics: str
    run_a_scored: str
    run_b_scored: str


class DiversityReportArtifact(StrictArtifactModel):
    config: DiversityReportConfig
    run_a_no_steering: DiversityRunArtifact
    run_b_steering: DiversityRunArtifact
    comparison: DiversityComparison
    artifacts: DiversityArtifacts
