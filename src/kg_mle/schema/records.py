"""Conversation JSONL artifact schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from kg_mle.generator.protocol import Conversation
from kg_mle.schema.common import EvaluationMetadata, FlexibleMetadataModel


class GeneratedConversationMetadata(FlexibleMetadataModel):
    original_chain: list[str]
    final_chain: list[str]
    n_tool_calls: int = Field(ge=0)
    domains: list[str]
    tools_visited: list[str]
    transition_summary: list[dict[str, Any]]


class ScoredConversationMetadata(GeneratedConversationMetadata):
    evaluation: EvaluationMetadata


class GeneratedConversationRecord(Conversation):
    """A generated conversation before evaluation."""

    @model_validator(mode="after")
    def _metadata_has_generation_fields(self):
        GeneratedConversationMetadata.model_validate(self.metadata)
        return self


class ScoredConversationRecord(Conversation):
    """A generated conversation after evaluation/scoring."""

    @model_validator(mode="after")
    def _metadata_has_evaluation_fields(self):
        ScoredConversationMetadata.model_validate(self.metadata)
        return self
