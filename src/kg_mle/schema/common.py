"""Common schema helpers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FlexibleMetadataModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class LLMJudgeScore(StrictArtifactModel):
    task_completion: float = Field(ge=0.0, le=10.0)
    tool_trace_validity: float = Field(ge=0.0, le=10.0)
    argument_grounding: float = Field(ge=0.0, le=10.0)
    response_grounding: float = Field(ge=0.0, le=10.0)
    naturalness: float = Field(ge=0.0, le=10.0)
    overall_score: float = Field(ge=0.0, le=10.0)
    confidence: float = Field(ge=0.0, le=10.0)
    issues: list[str] = Field(default_factory=list)
    rationale: str = ""


class LLMJudgeError(StrictArtifactModel):
    error: str


LLMJudgeMetadata = LLMJudgeScore | LLMJudgeError | None


class EvaluationMetadata(FlexibleMetadataModel):
    schema_valid: bool
    role_sequence_valid: bool
    tool_response_coverage: float = Field(ge=0.0, le=1.0)
    chain_completion: float = Field(ge=0.0, le=1.0)
    error_free_trace: float = Field(ge=0.0, le=1.0)
    deterministic_score: float = Field(ge=0.0, le=10.0)
    llm_judge: LLMJudgeMetadata = None
    quality_band: str
    usable_for_training: bool


class MetricsSummary(StrictArtifactModel):
    conversation_count: int = Field(ge=0)
    mean_deterministic_score: float = Field(ge=0.0, le=10.0)
    mean_chain_completion: float = Field(ge=0.0, le=1.0)
    mean_tool_response_coverage: float = Field(ge=0.0, le=1.0)
    schema_valid_rate: float = Field(ge=0.0, le=1.0)
    llm_judged_count: int = Field(ge=0)
    mean_llm_overall_score: float | None = Field(default=None, ge=0.0, le=10.0)
    usable_for_training_rate: float = Field(ge=0.0, le=1.0)


class RepairSummary(StrictArtifactModel):
    enabled: bool
    attempted: int = Field(ge=0)
    repaired: int = Field(ge=0)
    failed: int = Field(ge=0)
    rejected: int = Field(ge=0)
    regenerated: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _counts_are_consistent(self):
        if self.attempted < self.repaired + self.failed + self.rejected + self.regenerated:
            raise ValueError("repair attempted count is lower than status totals")
        return self


class RepairEvaluationMetadata(FlexibleMetadataModel):
    attempted: bool
    status: Literal["repaired", "failed", "rejected", "regenerated"]
    strategy: str
    before_scores: dict[str, Any]
    after_scores: dict[str, Any]
