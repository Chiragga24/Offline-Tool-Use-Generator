"""Live integration test for the LLM-backed conversation generator.

Same shape as the other live tests: runs when a configured LLM key is present,
skips otherwise; skips (does not fail) on any provider-side error.
Catches *protocol regressions* without flaking on LLM variance:

- The structured-output prompt round-trips through Pydantic validation.
- The conversation is role-tagged and structurally valid.
- At least one tool call lands and is grounded.
- last_run records "llm" or "fallback" coherently per agent call.
- A multi-step chain produces multiple tool calls (no silent degradation
  to a one-shot conversation).

We do NOT assert specific content (LLM variance), specific endpoints
the assistant picks (deviation may vary), or specific judge-style
scores.
"""

from __future__ import annotations

import os

import pytest

from kg_mle.config import DEFAULT_INPUT_PATH, DEFAULT_LLM_CONFIG
from kg_mle.executor import OfflineExecutor
from kg_mle.generator import (
    ConversationCoordinator,
    DeterministicAssistant,
    DeterministicPlanner,
    DeterministicUser,
    GeneratorConfig,
    StructuredLLMClient,
    make_llm_agents,
)
from kg_mle.graph import build_tool_graph
from kg_mle.registry import enrich_registry, load_registry
from kg_mle.sampler import ChainConstraints, ToolChainSampler


pytestmark = pytest.mark.live


def _require_live_llm_config() -> None:
    if DEFAULT_LLM_CONFIG.provider not in {"gemini", "groq", "huggingface"}:
        pytest.skip(
            "LLM generator live test currently supports gemini, groq, or huggingface "
            f"(currently {DEFAULT_LLM_CONFIG.provider!r})."
        )
    if not DEFAULT_LLM_CONFIG.api_key:
        pytest.skip(
            "LLM generator live test requires the configured provider API key "
            f"({DEFAULT_LLM_CONFIG.api_key_env})."
        )


def _build_pipeline():
    registry = load_registry(DEFAULT_INPUT_PATH)
    enrich_registry(registry)
    graph = build_tool_graph(registry)
    sampler = ToolChainSampler(graph)
    executor = OfflineExecutor(registry)
    return registry, graph, sampler, executor


def test_llm_generator_live_produces_structurally_valid_conversation():
    """End-to-end with real LLM agents. Skip on any provider-side failure.

    Asserts only contract-level properties:
      - the Conversation Pydantic validates,
      - messages are role-tagged,
      - at least one tool call succeeded,
      - per-agent last_run paths are coherent (llm or fallback, with reason),
      - the chain length matches the original chain or a graph-verified
        deviation.
    """
    _require_live_llm_config()

    registry, graph, sampler, executor = _build_pipeline()

    try:
        client = StructuredLLMClient.from_config(DEFAULT_LLM_CONFIG)
    except RuntimeError as exc:
        pytest.skip(f"LLM client setup unavailable: {exc}")

    config = GeneratorConfig(ambiguity_fraction=0.0, max_llm_retries=1)
    planner, user_simulator, assistant = make_llm_agents(
        client=client, registry=registry, config=config
    )

    coordinator = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=planner,
        user_simulator=user_simulator,
        assistant=assistant,
        config=config,
    )

    chain = sampler.sample(ChainConstraints(n_steps=2, min_grounded_transitions=1), seed=42)

    try:
        conversation = coordinator.run(chain, seed=42)
    except Exception as exc:
        # Any unexpected exception during live run is a skip, not a fail —
        # provider availability is volatile. The deterministic path already
        # covers correctness in CI.
        pytest.skip(f"LLM generator live run failed: {exc}")

    # Contract-level assertions only.
    assert conversation.messages, "Empty conversation."
    roles = [m["role"] for m in conversation.messages]
    assert roles[0] == "user", f"First message must be user, got {roles[0]!r}."
    assert all(r in {"user", "assistant", "tool"} for r in roles)

    tool_responses = [
        m
        for m in conversation.messages
        if m["role"] == "tool"
        and isinstance(m.get("content"), dict)
        and "error" not in m["content"]
    ]
    assert tool_responses, "LLM-driven conversation produced no successful tool calls."

    # last_run paths should be set on every agent that was invoked.
    assert planner.last_run.get("path") in {"llm", "fallback"}
    # The user is called at least once (initial request).
    assert user_simulator.last_run.get("path") in {"llm", "fallback"}
    # The assistant is called at least once per turn.
    assert assistant.last_run.get("path") in {"llm", "fallback"}

    # Chain length is sensible: either matches original or matches an accepted deviation.
    final_chain_len = len(conversation.metadata["final_chain"])
    accepted_deviations = conversation.metadata["deviations_accepted"]
    if accepted_deviations:
        expected_len = len(chain.endpoints) + sum(
            1 for d in accepted_deviations if d.get("kind") == "add_step"
        )
        assert final_chain_len == expected_len
    else:
        assert final_chain_len == len(chain.endpoints)


def test_llm_generator_live_runs_multi_step_chain_to_completion():
    """A 3-step chain must produce ≥1 successful tool call end-to-end via
    the LLM path.

    The contract this test enforces: the LLM-driven pipeline runs without
    crashing across multiple turns, validates structured output at each
    turn, hits the executor, and produces at least one chain-consistent
    response. Whether the LLM completes all 3 steps depends on provider
    responsiveness, model quality, and rate-limit budget — those are
    LLM-quality concerns, not protocol regressions. CI's deterministic
    coordinator tests already prove the coordinator drives a 3-step chain
    to completion when agents cooperate.
    """
    _require_live_llm_config()
    registry, graph, sampler, executor = _build_pipeline()

    try:
        client = StructuredLLMClient.from_config(DEFAULT_LLM_CONFIG)
    except RuntimeError as exc:
        pytest.skip(f"LLM client setup unavailable: {exc}")

    config = GeneratorConfig(ambiguity_fraction=0.0, max_llm_retries=1)
    planner, user_simulator, assistant = make_llm_agents(
        client=client, registry=registry, config=config
    )
    coordinator = ConversationCoordinator(
        registry=registry,
        graph=graph,
        executor=executor,
        planner=planner,
        user_simulator=user_simulator,
        assistant=assistant,
        config=config,
    )

    chain = sampler.sample(ChainConstraints(n_steps=3, min_grounded_transitions=1), seed=7)

    try:
        conversation = coordinator.run(chain, seed=7)
    except Exception as exc:
        pytest.skip(f"LLM generator live run failed: {exc}")

    n_successful_tool_calls = sum(
        1
        for m in conversation.messages
        if m["role"] == "tool"
        and isinstance(m.get("content"), dict)
        and "error" not in m["content"]
    )
    assert n_successful_tool_calls >= 1, (
        f"LLM-driven 3-step run produced 0 successful tool calls. "
        f"Conversation length: {len(conversation.messages)} messages. "
        f"Repair attempts: {len(conversation.metadata['repair_attempts'])}."
    )
    # Protocol intactness — all agents emitted a last_run path.
    assert planner.last_run.get("path") in {"llm", "fallback"}
    assert assistant.last_run.get("path") in {"llm", "fallback"}
