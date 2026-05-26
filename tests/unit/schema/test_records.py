from __future__ import annotations

import pytest
from pydantic import ValidationError

from kg_mle.schema import GeneratedConversationRecord, ScoredConversationRecord


def _conversation_payload() -> dict:
    return {
        "conversation_id": "conv_001",
        "messages": [
            {"role": "user", "content": "Track SAP and tell me if it moved."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "finance/get_quote",
                            "arguments": {"symbol": "SAP"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "finance/get_quote",
                "content": {"symbol": "SAP", "price": 199.25, "currency": "USD"},
            },
            {"role": "assistant", "content": "SAP is trading at 199.25 USD."},
        ],
        "plan": {
            "conversation_intent": "Check the current SAP stock quote.",
            "user_character": "default",
            "plan_confidence": 1.0,
            "step_plans": [
                {
                    "step_index": 0,
                    "endpoint_id": "finance/get_quote",
                    "parameter_plans": [
                        {
                            "parameter_name": "symbol",
                            "suggested_value": "SAP",
                            "confidence": 1.0,
                            "ambiguous": False,
                            "reason": "User named the ticker.",
                        }
                    ],
                }
            ],
            "ambiguous_step_indices": [],
        },
        "metadata": {
            "original_chain": ["finance/get_quote"],
            "final_chain": ["finance/get_quote"],
            "n_tool_calls": 1,
            "domains": ["finance"],
            "tools_visited": ["market_data"],
            "transition_summary": [],
            "seed": 42,
        },
    }


def _evaluation_payload() -> dict:
    return {
        "schema_valid": True,
        "role_sequence_valid": True,
        "tool_response_coverage": 1.0,
        "chain_completion": 1.0,
        "error_free_trace": 1.0,
        "deterministic_score": 10.0,
        "llm_judge": None,
        "quality_band": "gold",
        "usable_for_training": True,
    }


def test_generated_conversation_record_accepts_required_metadata() -> None:
    record = GeneratedConversationRecord.model_validate(_conversation_payload())

    assert record.conversation_id == "conv_001"
    assert record.metadata["tools_visited"] == ["market_data"]


def test_generated_conversation_record_rejects_missing_generation_metadata() -> None:
    payload = _conversation_payload()
    del payload["metadata"]["tools_visited"]

    with pytest.raises(ValidationError):
        GeneratedConversationRecord.model_validate(payload)


def test_scored_conversation_record_requires_evaluation_metadata() -> None:
    payload = _conversation_payload()
    payload["metadata"]["evaluation"] = _evaluation_payload()

    record = ScoredConversationRecord.model_validate(payload)

    assert record.metadata["evaluation"]["deterministic_score"] == 10.0


def test_scored_conversation_record_rejects_missing_evaluation_metadata() -> None:
    with pytest.raises(ValidationError):
        ScoredConversationRecord.model_validate(_conversation_payload())
