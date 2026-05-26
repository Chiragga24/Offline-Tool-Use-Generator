import json

from kg_mle.evaluation import LLMJudge, evaluate_dataset, load_conversations_jsonl
from kg_mle.repair import LLMRepairPlanner


class _FakeJSONClient:
    def complete_json(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        return json.dumps(
            {
                "task_completion": 9.0,
                "tool_trace_validity": 8.0,
                "argument_grounding": 8.5,
                "response_grounding": 8.5,
                "naturalness": 7.5,
                "overall_score": 8.2,
                "confidence": 7.0,
                "issues": ["minor wording issue"],
                "rationale": "Trace is coherent with one minor style issue.",
            }
        )


def _conversation():
    return {
        "conversation_id": "conv_test",
        "messages": [
            {"role": "user", "content": "Find a stock quote."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "endpoint_id": "finance/get_quote",
                        "arguments": {"symbol": "SAP"},
                        "call_confidence": 1.0,
                    }
                ],
            },
            {
                "role": "tool",
                "endpoint": "finance/get_quote",
                "content": {"symbol": "SAP", "price": "199.25"},
            },
            {"role": "assistant", "content": "SAP is trading at 199.25."},
        ],
        "plan": {
            "conversation_intent": "Find a stock quote.",
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
                            "reason": "Provided by user.",
                        }
                    ],
                }
            ],
            "ambiguous_step_indices": [],
        },
        "metadata": {"n_tool_calls": 1},
    }


def test_evaluate_dataset_computes_deterministic_metrics():
    evaluation = evaluate_dataset([_conversation()])

    assert evaluation["summary"]["conversation_count"] == 1
    assert evaluation["summary"]["mean_deterministic_score"] == 10.0
    assert evaluation["records"][0]["chain_completion"] == 1.0
    assert evaluation["records"][0]["llm_judge"] is None
    scored = evaluation["scored_conversations"][0]
    assert scored["metadata"]["evaluation"]["deterministic_score"] == 10.0


def test_evaluate_dataset_attaches_llm_judge_score():
    judge = LLMJudge(_FakeJSONClient())

    evaluation = evaluate_dataset([_conversation()], judge=judge, max_judged_records=1)

    assert evaluation["summary"]["llm_judged_count"] == 1
    assert evaluation["summary"]["mean_llm_overall_score"] == 8.2
    assert evaluation["records"][0]["llm_judge"]["tool_trace_validity"] == 8.0
    assert (
        evaluation["scored_conversations"][0]["metadata"]["evaluation"]["llm_judge"][
            "argument_grounding"
        ]
        == 8.5
    )


def test_evaluate_dataset_repair_rewrites_low_naturalness_response():
    class LowNaturalnessClient(_FakeJSONClient):
        calls = 0

        def complete_json(self, *, system: str, user: str, temperature: float = 0.0) -> str:
            self.calls += 1
            naturalness = 4.0 if self.calls == 1 else 8.0
            return json.dumps(
                {
                    "task_completion": 9.0,
                    "tool_trace_validity": 9.0,
                    "argument_grounding": 9.0,
                    "response_grounding": 9.0,
                    "naturalness": naturalness,
                    "overall_score": 8.5,
                    "confidence": 8.0,
                    "issues": ["awkward wording"] if naturalness < 5 else [],
                    "rationale": "Fake score.",
                }
            )

    judge = LLMJudge(LowNaturalnessClient())

    evaluation = evaluate_dataset(
        [_conversation()],
        judge=judge,
        max_judged_records=1,
        repair=True,
    )

    assert evaluation["repair_summary"]["attempted"] == 1
    assert evaluation["repair_summary"]["repaired"] == 1
    scored = evaluation["scored_conversations"][0]
    assert scored["metadata"]["repair_history"][0]["plan"]["strategy"] == "rewrite_final_response"
    assert scored["metadata"]["evaluation"]["repair"]["status"] == "repaired"
    assert scored["metadata"]["evaluation"]["llm_judge"]["naturalness"] == 8.0
    assert scored["messages"][-2]["role"] == "system"
    assert scored["messages"][-2]["name"] == "repair"


def test_evaluate_dataset_can_use_llm_repair_planner():
    class LowNaturalnessClient(_FakeJSONClient):
        calls = 0

        def complete_json(self, *, system: str, user: str, temperature: float = 0.0) -> str:
            self.calls += 1
            if "repair planner" in system.lower():
                return json.dumps(
                    {
                        "strategy": "rewrite_final_response",
                        "reason": "Use a grounded final answer.",
                        "target_step": None,
                        "target_endpoint": None,
                        "proposed_arguments": None,
                        "proposed_final_response": "Done from the quoted tool result.",
                        "proposed_chain_change": None,
                        "requires_coordinator": False,
                        "confidence": 8.0,
                    }
                )
            naturalness = 4.0 if self.calls == 1 else 8.0
            return json.dumps(
                {
                    "task_completion": 9.0,
                    "tool_trace_validity": 9.0,
                    "argument_grounding": 9.0,
                    "response_grounding": 9.0,
                    "naturalness": naturalness,
                    "overall_score": 8.5,
                    "confidence": 8.0,
                    "issues": [],
                    "rationale": "Fake score.",
                }
            )

    client = LowNaturalnessClient()
    judge = LLMJudge(client)
    repair_planner = LLMRepairPlanner(client=client)

    evaluation = evaluate_dataset(
        [_conversation()],
        judge=judge,
        max_judged_records=1,
        repair=True,
        repair_planner=repair_planner,
    )

    scored = evaluation["scored_conversations"][0]
    assert repair_planner.last_run["path"] == "llm"
    assert scored["metadata"]["repair_history"][0]["plan"]["reason"] == "Use a grounded final answer."
    assert scored["messages"][-1]["content"] == "Done from the quoted tool result."


def test_evaluate_dataset_repair_marks_tool_error_as_failed_coordinator_repair():
    broken = _conversation()
    broken["messages"].append(
        {
            "role": "tool",
            "endpoint": "finance/get_quote",
            "content": {"error": {"kind": "ungrounded_argument"}},
        }
    )

    evaluation = evaluate_dataset([broken], repair=True)

    assert evaluation["repair_summary"]["attempted"] == 1
    assert evaluation["repair_summary"]["failed"] == 1
    scored = evaluation["scored_conversations"][0]
    assert scored["metadata"]["repair_history"][0]["plan"]["strategy"] == "regenerate_conversation"
    assert scored["metadata"]["evaluation"]["repair"]["status"] == "failed"
    assert scored["metadata"]["evaluation"]["usable_for_training"] is False


def test_load_conversations_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "conversations.jsonl"
    path.write_text(json.dumps(_conversation()) + "\n\n", encoding="utf-8")

    conversations = load_conversations_jsonl(path)

    assert len(conversations) == 1
    assert conversations[0]["conversation_id"] == "conv_test"
