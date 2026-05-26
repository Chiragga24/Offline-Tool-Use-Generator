"""Pydantic models for repair planning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RepairStrategy = Literal[
    "fix_tool_arguments",
    "rewrite_final_response",
    "insert_clarification",
    "apply_graph_verified_chain_change",
    "regenerate_conversation",
    "mark_rejected",
]


class RepairTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    reason: str
    score_name: str | None = None
    score_value: float | None = None
    threshold: float | None = None


class RepairPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: RepairStrategy
    triggers: list[RepairTrigger] = Field(default_factory=list)
    reason: str
    target_step: int | None = None
    target_endpoint: str | None = None
    proposed_arguments: dict[str, Any] | None = None
    proposed_final_response: str | None = None
    proposed_chain_change: dict[str, Any] | None = None
    requires_coordinator: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=10.0)


class RepairResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["repaired", "unchanged", "failed", "regenerated", "rejected"]
    plan: RepairPlan
    attempts: int
    notes: list[str] = Field(default_factory=list)
    before_scores: dict[str, Any] = Field(default_factory=dict)
    after_scores: dict[str, Any] = Field(default_factory=dict)
