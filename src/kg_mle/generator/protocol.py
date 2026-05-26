"""Pydantic protocol for the multi-agent generator.

All agent I/O is validated through these models. The coordinator never
accepts free-text from an agent without round-tripping through the
relevant Pydantic class — that is the structured-output guarantee the
assignment requires.

Confidence values are present at three levels:

- per-parameter (`ParameterPlan.confidence`): the planner's certainty
  about a suggested value. Below `planner_param_low_confidence`, the
  assistant may take initiative on clarification.
- per-clarification (`AssistantTurn.assistant_clarification_confidence`):
  how strongly the assistant believes a clarifying question is needed
  beyond what the planner already marked. Below
  `assistant_clarification_threshold`, the coordinator ignores the
  assistant's initiative.
- per-deviation (`ChainDeviation.deviation_confidence`): how strongly
  the assistant believes the chain should change. Below
  `assistant_deviation_threshold`, the coordinator records the proposal
  in metadata but rejects it.

Every gate is configurable via `GeneratorConfig`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ParameterPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter_name: str
    suggested_value: Any = None
    """Optional. None means the value is supplied at runtime by the
    executor (grounded params) or by user clarification."""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ambiguous: bool = False
    reason: str = ""


class StepPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    endpoint_id: str
    parameter_plans: list[ParameterPlan] = Field(default_factory=list)


class Plan(BaseModel):
    """High-level plan for one conversation."""

    model_config = ConfigDict(extra="forbid")

    conversation_intent: str
    """A one-sentence description of the user's underlying goal."""

    user_character: str = "default"
    """Optional persona for the user simulator. The deterministic user
    ignores this; the LLM user may use it for tone variation."""

    plan_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    step_plans: list[StepPlan]
    ambiguous_step_indices: list[int] = Field(default_factory=list)
    """Steps for which the planner pre-decided a clarifying turn should
    happen. Drives planner-driven disambiguation; the assistant may add
    more via confidence-gated initiative."""


class ToolCallProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint_id: str
    arguments: dict[str, Any]
    call_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ChainDeviation(BaseModel):
    """Assistant's proposal to add or modify a step.

    Coordinator accepts when `deviation_confidence >=
    assistant_deviation_threshold` AND the proposed endpoint exists
    AND the graph supports the new transitions. Otherwise the proposal
    is recorded in metadata as `rejected_deviations` and the chain
    proceeds unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["add_step", "modify_step"]
    endpoint_id: str
    position: int
    reasoning: str
    deviation_confidence: float = Field(ge=0.0, le=1.0)


class UserTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    is_initial_request: bool = False
    is_clarification_reply: bool = False


class ClarificationTarget(BaseModel):
    """Which parameter the assistant is asking the user about.

    Carried on AssistantTurn so the coordinator can route the user
    simulator's reply without having to parse natural-language content.
    """

    model_config = ConfigDict(extra="forbid")

    step_index: int
    parameter_name: str


class AssistantTurn(BaseModel):
    """One turn from the assistant.

    `tool_calls` is intentionally a list so future parallel-call support
    is purely additive at the data-model layer. Sequential chains
    produce length-1 lists when `kind == "tool_calls"`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["clarification", "tool_calls", "final_summary"]
    content: str | None = None
    tool_calls: list[ToolCallProposal] = Field(default_factory=list)
    clarification_target: ClarificationTarget | None = None
    assistant_clarification_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    chain_deviation: ChainDeviation | None = None


class GeneratorConfig(BaseModel):
    """Configurable gates for the generator.

    Defaults are tuned for the curated fixture; tests can override per
    case to exercise threshold behaviour without monkey-patching."""

    model_config = ConfigDict(extra="forbid")

    planner_param_low_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    assistant_clarification_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    assistant_deviation_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_llm_retries: int = Field(default=1, ge=0, le=5)
    max_repair_attempts: int = Field(default=1, ge=0, le=3)
    ambiguity_fraction: float = Field(default=0.4, ge=0.0, le=1.0)
    """Fraction of conversations the deterministic planner injects
    ambiguity into. The LLM planner is free to vary."""


class Conversation(BaseModel):
    """One generated conversation, serialisable to the dataset's JSONL line."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    messages: list[dict[str, Any]]
    plan: Plan
    metadata: dict[str, Any]
