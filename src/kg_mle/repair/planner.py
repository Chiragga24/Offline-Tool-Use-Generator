"""Repair planners and small safe repairs."""

from __future__ import annotations

import json
from typing import Any

from kg_mle.repair.models import RepairPlan, RepairTrigger
from kg_mle.generator.llm_agents import _extract_json_object
from kg_mle.llm.clients import StructuredLLMClient


class DeterministicRepairPlanner:
    """Creates a conservative repair plan from validation/quality triggers."""

    def plan(
        self,
        conversation: dict[str, Any],
        *,
        triggers: list[RepairTrigger],
    ) -> RepairPlan:
        trigger_kinds = {trigger.kind for trigger in triggers}
        if {"validation_error", "tool_error"} & trigger_kinds:
            return RepairPlan(
                strategy="regenerate_conversation",
                triggers=triggers,
                reason="Trace-level validation/tool errors require coordinator regeneration.",
                requires_coordinator=True,
                confidence=8.0,
            )
        if "low_argument_grounding" in trigger_kinds or "low_tool_trace_validity" in trigger_kinds:
            return RepairPlan(
                strategy="apply_graph_verified_chain_change",
                triggers=triggers,
                reason="Tool trace or argument grounding is low; coordinator should validate or replace the chain.",
                requires_coordinator=True,
                confidence=7.0,
            )
        if "low_task_completion" in trigger_kinds:
            return RepairPlan(
                strategy="rewrite_final_response",
                triggers=triggers,
                reason="Task completion is low; rewrite final response from tool outputs.",
                proposed_final_response=_grounded_final_response(conversation),
                requires_coordinator=False,
                confidence=8.0,
            )
        if "low_naturalness" in trigger_kinds:
            return RepairPlan(
                strategy="rewrite_final_response",
                triggers=triggers,
                reason="Naturalness is low; polish final response without changing tool calls.",
                proposed_final_response=_grounded_final_response(conversation),
                requires_coordinator=False,
                confidence=7.0,
            )
        return RepairPlan(
            strategy="mark_rejected",
            triggers=triggers,
            reason="No safe deterministic repair is available for the detected triggers.",
            requires_coordinator=False,
            confidence=5.0,
        )

    def apply(self, conversation: dict[str, Any], plan: RepairPlan) -> tuple[dict[str, Any], str]:
        repaired = _deepcopy_jsonish(conversation)
        if plan.strategy == "rewrite_final_response" and plan.proposed_final_response:
            _replace_or_append_final_assistant(repaired, plan.proposed_final_response)
            _append_repair_log(repaired, plan)
            return repaired, "repaired"
        if plan.strategy in {"regenerate_conversation", "apply_graph_verified_chain_change"}:
            _append_repair_log(repaired, plan)
            return repaired, "failed"
        _append_repair_log(repaired, plan)
        return repaired, "rejected"


_REPAIR_SYSTEM_PROMPT = """You are a repair planner for synthetic tool-use conversations.
Return JSON only. Do not rewrite the whole conversation.

Allowed strategies:
- fix_tool_arguments
- rewrite_final_response
- insert_clarification
- apply_graph_verified_chain_change
- regenerate_conversation
- mark_rejected

Rules:
- Prefer rewrite_final_response when final answer quality/naturalness/grounding is the only issue.
- Prefer fix_tool_arguments for low argument_grounding when the trace contains enough prior tool outputs.
- Prefer apply_graph_verified_chain_change for low tool_trace_validity.
- Prefer regenerate_conversation for schema errors, tool errors, or impossible traces.
- Set requires_coordinator=true for chain changes, argument fixes, clarification insertion, or regeneration.
- Use confidence on a 0-10 scale.
- Return only fields in the RepairPlan schema.
"""


class LLMRepairPlanner:
    """LLM repair planner with deterministic fallback.

    The LLM only proposes a `RepairPlan`. Application still goes through the
    deterministic planner so unsafe mutation rules stay centralized.
    """

    def __init__(
        self,
        *,
        client: StructuredLLMClient,
        fallback: DeterministicRepairPlanner | None = None,
        max_retries: int = 1,
    ) -> None:
        self._client = client
        self._fallback = fallback or DeterministicRepairPlanner()
        self._max_retries = max_retries
        self.last_run: dict[str, Any] = {}

    def plan(
        self,
        conversation: dict[str, Any],
        *,
        triggers: list[RepairTrigger],
    ) -> RepairPlan:
        prompt = json.dumps(
            {
                "conversation_id": conversation.get("conversation_id"),
                "messages": conversation.get("messages", []),
                "metadata": conversation.get("metadata", {}),
                "triggers": [trigger.model_dump() for trigger in triggers],
                "repair_plan_schema": {
                    "strategy": "one allowed strategy string",
                    "reason": "short explanation",
                    "target_step": "int or null",
                    "target_endpoint": "string or null",
                    "proposed_arguments": "object or null",
                    "proposed_final_response": "string or null",
                    "proposed_chain_change": "object or null",
                    "requires_coordinator": "boolean",
                    "confidence": "float 0..10",
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
                    system=_REPAIR_SYSTEM_PROMPT,
                    user=user_prompt,
                    temperature=0.0,
                )
                payload = _extract_json_object(content)
                payload["triggers"] = [trigger.model_dump() for trigger in triggers]
                plan = RepairPlan.model_validate(payload)
                self.last_run = {"path": "llm", "retries": attempt}
                return plan
            except Exception as exc:
                last_error = str(exc)
                continue
        self.last_run = {"path": "fallback", "reason": last_error, "retries": self._max_retries + 1}
        return self._fallback.plan(conversation, triggers=triggers)

    def apply(self, conversation: dict[str, Any], plan: RepairPlan) -> tuple[dict[str, Any], str]:
        return self._fallback.apply(conversation, plan)


def _grounded_final_response(conversation: dict[str, Any]) -> str:
    tool_summaries: list[str] = []
    for message in conversation.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        endpoint = message.get("endpoint", "tool")
        content = message.get("content")
        if not isinstance(content, dict) or "error" in content:
            continue
        fields = ", ".join(f"{key}={value}" for key, value in list(content.items())[:3])
        tool_summaries.append(f"{endpoint} returned {fields}")
    if not tool_summaries:
        return "I could not complete the request from the available tool results."
    return "Done. I used the tool results: " + "; ".join(tool_summaries) + "."


def _replace_or_append_final_assistant(conversation: dict[str, Any], content: str) -> None:
    messages = conversation.setdefault("messages", [])
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content"):
            message["content"] = content
            return
    messages.append({"role": "assistant", "content": content})


def _append_repair_log(conversation: dict[str, Any], plan: RepairPlan) -> None:
    messages = conversation.setdefault("messages", [])
    repair_message = {
        "role": "system",
        "name": "repair",
        "content": {
            "strategy": plan.strategy,
            "reason": plan.reason,
            "requires_coordinator": plan.requires_coordinator,
        },
    }
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], dict) and messages[idx].get("role") == "assistant":
            messages.insert(idx, repair_message)
            return
    messages.append(repair_message)


def _deepcopy_jsonish(value: dict[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(json.dumps(value, default=str))
