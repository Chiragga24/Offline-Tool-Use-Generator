from kg_mle.repair import RepairPolicy, assign_quality_band, should_attempt_repair


def test_policy_triggers_for_low_scores_and_tool_errors():
    record = {
        "schema_valid": True,
        "role_sequence_valid": True,
        "n_tool_errors": 1,
        "deterministic_score": 7.5,
        "llm_judge": {
            "task_completion": 8.5,
            "tool_trace_validity": 7.0,
            "argument_grounding": 7.5,
            "naturalness": 4.0,
        },
    }

    triggers = should_attempt_repair(record, RepairPolicy())

    assert {trigger.kind for trigger in triggers} == {
        "tool_error",
        "low_deterministic_score",
        "low_task_completion",
        "low_tool_trace_validity",
        "low_argument_grounding",
        "low_naturalness",
    }


def test_quality_band_rejects_bad_grounding_even_with_high_deterministic_score():
    record = {
        "schema_valid": True,
        "n_tool_errors": 0,
        "deterministic_score": 10.0,
        "llm_judge": {
            "task_completion": 10.0,
            "tool_trace_validity": 10.0,
            "argument_grounding": 6.0,
        },
    }

    quality_band, usable = assign_quality_band(record)

    assert quality_band == "reject"
    assert usable is False
