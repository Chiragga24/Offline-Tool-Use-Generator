"""Optional LLM-as-judge for generated conversations."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kg_mle.generator.llm_agents import _extract_json_object
from kg_mle.llm.clients import StructuredLLMClient


class JudgeScore(BaseModel):
    """Structured judge output for one conversation."""

    model_config = ConfigDict(extra="forbid")

    task_completion: float = Field(ge=0.0, le=10.0)
    tool_trace_validity: float = Field(ge=0.0, le=10.0)
    argument_grounding: float = Field(ge=0.0, le=10.0)
    response_grounding: float = Field(ge=0.0, le=10.0)
    naturalness: float = Field(ge=0.0, le=10.0)
    overall_score: float = Field(ge=0.0, le=10.0)
    confidence: float = Field(ge=0.0, le=10.0)
    issues: list[str] = Field(default_factory=list)
    rationale: str = ""


_JUDGE_SYSTEM_PROMPT = """You are an evaluator for synthetic tool-use conversations.
Score only what is visible in the provided JSON. Do not assume real API behavior.

Return JSON only with this schema:
{
  "task_completion": float 0..10,
  "tool_trace_validity": float 0..10,
  "argument_grounding": float 0..10,
  "response_grounding": float 0..10,
  "naturalness": float 0..10,
  "overall_score": float 0..10,
  "confidence": float 0..10,
  "issues": list[str],
  "rationale": string
}

Scoring scale:
- 10 = excellent, no visible issue
- 8 = good, minor issue
- 5 = partially correct but materially flawed
- 2 = mostly broken
- 0 = invalid or unusable

Guardrails:
- Penalize missing role tags, missing tool responses, invented tool IDs, and unresolved tool errors.
- Penalize ungrounded arguments when the tool trace shows an executor error.
- Penalize incomplete tasks even when individual tool calls are valid.
- Reward concise clarification when the plan marks ambiguity.
- Keep rationale under 60 words.
"""


class LLMJudge:
    """Provider-backed judge with validation and deterministic failure mode."""

    def __init__(self, client: StructuredLLMClient, *, max_retries: int = 1) -> None:
        self._client = client
        self._max_retries = max_retries

    def score(self, conversation: dict[str, Any]) -> JudgeScore:
        prompt = json.dumps(
            {
                "conversation": conversation,
                "rubric": {
                    "task_completion": "Did the final assistant response complete the user's task?",
                    "tool_trace_validity": "Are the selected tools valid and sequenced correctly?",
                    "argument_grounding": "Do tool-call arguments come from user input, plan values, or prior tool outputs?",
                    "response_grounding": "Does the final answer stay grounded in tool outputs?",
                    "naturalness": "Does the dialogue read like a plausible assistant/user exchange?",
                    "overall_score": "Holistic quality, not a simple average if there are severe failures.",
                },
            },
            indent=2,
            default=str,
        )
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            user_prompt = (
                f"{prompt}\n\nPrevious-attempt error: {last_error}" if last_error else prompt
            )
            try:
                content = self._client.complete_json(
                    system=_JUDGE_SYSTEM_PROMPT,
                    user=user_prompt,
                    temperature=0.0,
                )
                return JudgeScore.model_validate(_extract_json_object(content))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                continue
        raise ValueError(f"LLM judge failed structured validation: {last_error}")
