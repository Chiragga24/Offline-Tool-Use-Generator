"""Evaluation metrics artifact schema."""

from __future__ import annotations

from pydantic import Field

from kg_mle.schema.common import (
    LLMJudgeMetadata,
    MetricsSummary,
    RepairEvaluationMetadata,
    RepairSummary,
    StrictArtifactModel,
)


class EvaluationRecordMetrics(StrictArtifactModel):
    conversation_id: str
    schema_valid: bool
    schema_error: str | None = None
    role_sequence_valid: bool
    n_messages: int = Field(ge=0)
    n_assistant_tool_calls: int = Field(ge=0)
    n_tool_messages: int = Field(ge=0)
    n_tool_errors: int = Field(ge=0)
    tool_response_coverage: float = Field(ge=0.0, le=1.0)
    chain_completion: float = Field(ge=0.0, le=1.0)
    error_free_trace: float = Field(ge=0.0, le=1.0)
    deterministic_score: float = Field(ge=0.0, le=10.0)
    llm_judge: LLMJudgeMetadata = None
    quality_band: str
    usable_for_training: bool
    repair: RepairEvaluationMetadata | None = None


class EvaluationMetricsArtifact(StrictArtifactModel):
    summary: MetricsSummary
    repair_summary: RepairSummary
    records: list[EvaluationRecordMetrics]
