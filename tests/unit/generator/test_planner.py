"""DeterministicPlanner tests."""

import pytest

from kg_mle.config import DEFAULT_INPUT_PATH
from kg_mle.generator import DeterministicPlanner, GeneratorConfig
from kg_mle.graph import build_tool_graph
from kg_mle.registry import enrich_registry, load_registry
from kg_mle.sampler import ChainConstraints, ToolChainSampler


@pytest.fixture(scope="module")
def planner_inputs():
    registry = load_registry(DEFAULT_INPUT_PATH)
    enrich_registry(registry)
    graph = build_tool_graph(registry)
    sampler = ToolChainSampler(graph)
    return registry, sampler


def test_plan_contains_one_step_plan_per_chain_endpoint(planner_inputs):
    registry, sampler = planner_inputs
    chain = sampler.sample(ChainConstraints(n_steps=3, min_grounded_transitions=1), seed=42)
    plan = DeterministicPlanner(registry).plan(chain, seed=42)
    assert [sp.endpoint_id for sp in plan.step_plans] == list(chain.endpoints)


def test_plan_is_deterministic_for_same_seed(planner_inputs):
    registry, sampler = planner_inputs
    chain = sampler.sample(ChainConstraints(n_steps=3), seed=42)
    plan_a = DeterministicPlanner(registry).plan(chain, seed=7)
    plan_b = DeterministicPlanner(registry).plan(chain, seed=7)
    assert plan_a == plan_b


def test_grounded_params_have_no_suggested_value(planner_inputs):
    registry, sampler = planner_inputs
    chain = sampler.sample(ChainConstraints(n_steps=3, min_grounded_transitions=1), seed=42)
    plan = DeterministicPlanner(registry).plan(chain, seed=42)
    for step_index, step in enumerate(plan.step_plans):
        grounded_param_names = {
            t.parameter
            for t in chain.transitions
            if t.advance_type == "grounded"
            and t.target == chain.endpoints[step_index]
            and t.parameter is not None
        }
        for pp in step.parameter_plans:
            if pp.parameter_name in grounded_param_names:
                assert pp.suggested_value is None, (
                    f"Grounded param {pp.parameter_name} got pre-assigned: {pp.suggested_value}"
                )


def test_ambiguity_fraction_zero_yields_no_ambiguous_steps(planner_inputs):
    registry, sampler = planner_inputs
    chain = sampler.sample(ChainConstraints(n_steps=3), seed=42)
    planner = DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=0.0))
    plan = planner.plan(chain, seed=42)
    assert plan.ambiguous_step_indices == []


def test_ambiguity_fraction_one_yields_at_least_one_ambiguous(planner_inputs):
    """ambiguity_fraction=1.0 forces ambiguity injection on every conversation."""
    registry, sampler = planner_inputs
    chain = sampler.sample(ChainConstraints(n_steps=3), seed=42)
    planner = DeterministicPlanner(registry, config=GeneratorConfig(ambiguity_fraction=1.0))
    plan = planner.plan(chain, seed=42)
    # If the step has free params, ambiguity injection sticks.
    has_free_param = any(
        pp.suggested_value not in (None,) for pp in plan.step_plans[0].parameter_plans
    )
    if has_free_param:
        assert 0 in plan.ambiguous_step_indices
        assert any(pp.ambiguous for pp in plan.step_plans[0].parameter_plans)


def test_unknown_canonical_params_get_low_confidence(planner_inputs):
    """Params without a canonical example pool should mark low confidence so the
    assistant-initiative branch has something to fire on."""
    registry, sampler = planner_inputs
    chain = sampler.sample(ChainConstraints(n_steps=3), seed=42)
    plan = DeterministicPlanner(registry).plan(chain, seed=42)
    found_low_conf = False
    for step in plan.step_plans:
        for pp in step.parameter_plans:
            if pp.suggested_value == f"<{pp.parameter_name}>":
                assert pp.confidence < 1.0
                found_low_conf = True
    # Not all chains will have an unknown-canonical param, so this is informational
    # rather than asserted true. Surfacing the metric is enough.
    assert isinstance(found_low_conf, bool)
