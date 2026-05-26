"""Pydantic-protocol tests.

These guarantee the structured-output contract every agent must honor.
A malformed LLM output is caught here, not deep in the coordinator.
"""

import pytest
from pydantic import ValidationError

from kg_mle.generator import (
    AssistantTurn,
    ChainDeviation,
    ClarificationTarget,
    GeneratorConfig,
    ParameterPlan,
    Plan,
    StepPlan,
    ToolCallProposal,
)


def test_parameter_plan_confidence_bounded():
    with pytest.raises(ValidationError):
        ParameterPlan(parameter_name="city", confidence=1.5)
    with pytest.raises(ValidationError):
        ParameterPlan(parameter_name="city", confidence=-0.1)


def test_plan_rejects_unknown_fields():
    """extra=forbid means an LLM that hallucinates extra keys is rejected."""
    with pytest.raises(ValidationError):
        Plan(
            conversation_intent="hi",
            step_plans=[],
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_chain_deviation_requires_kind_and_confidence():
    with pytest.raises(ValidationError):
        ChainDeviation(  # type: ignore[call-arg]
            kind="invalid",  # type: ignore[arg-type]
            endpoint_id="travel/x",
            position=1,
            reasoning="r",
            deviation_confidence=0.9,
        )
    # Confidence above 1 rejected
    with pytest.raises(ValidationError):
        ChainDeviation(
            kind="add_step",
            endpoint_id="travel/x",
            position=1,
            reasoning="r",
            deviation_confidence=1.5,
        )


def test_assistant_turn_default_tool_calls_empty_list():
    turn = AssistantTurn(kind="clarification", content="?")
    assert turn.tool_calls == []
    assert turn.chain_deviation is None
    assert turn.assistant_clarification_confidence == 1.0


def test_assistant_turn_with_clarification_target():
    turn = AssistantTurn(
        kind="clarification",
        content="?",
        clarification_target=ClarificationTarget(step_index=0, parameter_name="city"),
    )
    assert turn.clarification_target is not None
    assert turn.clarification_target.step_index == 0


def test_tool_call_proposal_round_trips():
    proposal = ToolCallProposal(
        endpoint_id="travel/search_hotels",
        arguments={"city": "Paris", "max_price": 200},
        call_confidence=0.95,
    )
    dumped = proposal.model_dump()
    revived = ToolCallProposal.model_validate(dumped)
    assert revived == proposal


def test_generator_config_defaults_match_design():
    config = GeneratorConfig()
    assert config.planner_param_low_confidence == 0.6
    assert config.assistant_clarification_threshold == 0.7
    assert config.assistant_deviation_threshold == 0.85
    assert config.max_llm_retries == 1
    assert config.ambiguity_fraction == 0.4


def test_step_plan_indexes_zero_based():
    sp = StepPlan(step_index=0, endpoint_id="travel/search_hotels")
    assert sp.step_index == 0
    assert sp.parameter_plans == []
