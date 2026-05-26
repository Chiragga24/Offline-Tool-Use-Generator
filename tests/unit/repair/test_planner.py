from kg_mle.repair import DeterministicRepairPlanner, LLMRepairPlanner
from kg_mle.repair.models import RepairTrigger


def test_planner_rewrites_final_response_for_low_naturalness():
    conversation = {
        "messages": [
            {"role": "user", "content": "Book it."},
            {
                "role": "tool",
                "endpoint": "events/book_tickets",
                "content": {"ticket_booking_id": "tic_123", "status": "confirmed"},
            },
            {"role": "assistant", "content": "ok"},
        ]
    }
    planner = DeterministicRepairPlanner()

    plan = planner.plan(
        conversation,
        triggers=[RepairTrigger(kind="low_naturalness", reason="naturalness below threshold")],
    )
    repaired, status = planner.apply(conversation, plan)

    assert plan.strategy == "rewrite_final_response"
    assert status == "repaired"
    assert repaired["messages"][-2]["role"] == "system"
    assert "events/book_tickets returned ticket_booking_id=tic_123" in repaired["messages"][-1]["content"]


def test_planner_requires_coordinator_for_trace_level_errors():
    planner = DeterministicRepairPlanner()

    plan = planner.plan(
        {"messages": []},
        triggers=[RepairTrigger(kind="tool_error", reason="tool error present")],
    )

    assert plan.strategy == "regenerate_conversation"
    assert plan.requires_coordinator is True


class _FakeRepairClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def complete_json(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls += 1
        return self.content


def test_llm_repair_planner_returns_validated_plan():
    client = _FakeRepairClient(
        """
        {
          "strategy": "rewrite_final_response",
          "reason": "Final response needs grounding.",
          "target_step": null,
          "target_endpoint": null,
          "proposed_arguments": null,
          "proposed_final_response": "Done from tool outputs.",
          "proposed_chain_change": null,
          "requires_coordinator": false,
          "confidence": 8.0
        }
        """
    )
    planner = LLMRepairPlanner(client=client)

    plan = planner.plan(
        {"conversation_id": "conv_test", "messages": []},
        triggers=[RepairTrigger(kind="low_naturalness", reason="naturalness below threshold")],
    )

    assert plan.strategy == "rewrite_final_response"
    assert plan.triggers[0].kind == "low_naturalness"
    assert planner.last_run["path"] == "llm"


def test_llm_repair_planner_falls_back_on_invalid_output():
    client = _FakeRepairClient('{"strategy": "not_allowed"}')
    planner = LLMRepairPlanner(client=client, max_retries=0)

    plan = planner.plan(
        {"messages": []},
        triggers=[RepairTrigger(kind="tool_error", reason="tool error present")],
    )

    assert plan.strategy == "regenerate_conversation"
    assert planner.last_run["path"] == "fallback"
