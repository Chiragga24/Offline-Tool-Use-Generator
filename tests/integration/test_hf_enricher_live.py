"""Live integration test for the Hugging Face registry enricher.

Like the Mem0 live test, this is a *protocol smoke test*, not a correctness
test. It proves the wire path is intact end-to-end against a real provider,
without making strong claims about specific suggestions.

LLM output is non-deterministic and provider availability is volatile
(models get retired, providers add billing walls). The test therefore:

- caps to 2 endpoints to keep cost minimal and stay within free-tier limits,
- skips (does not fail) on any provider-side error: missing credentials,
  402 billing rejection, model not available, transient HF outage,
- asserts only that the path completes and produces a well-formed report.

What this catches:
- the HuggingFace InferenceClient API changes shape,
- our JSON-extraction parser stops handling the model's response format,
- the structured-output validation pipeline crashes on real LLM output.

What this does NOT guarantee:
- that any specific field gets enriched (LLM may decline),
- that suggestions are semantically correct (LLM may be wrong, and the
  confidence threshold may reject them — that's the design).
"""

from __future__ import annotations

import os

import pytest

from kg_mle.config import DEFAULT_INPUT_PATH, DEFAULT_LLM_CONFIG
from kg_mle.registry import (
    HuggingFaceRegistryEnricher,
    enrich_registry,
    load_registry,
)


pytestmark = pytest.mark.live


def _require_hf_token() -> None:
    if DEFAULT_LLM_CONFIG.provider != "huggingface":
        pytest.skip(
            "HF enricher live test requires KG_MLE_LLM_PROVIDER=huggingface "
            f"(currently {DEFAULT_LLM_CONFIG.provider!r})."
        )
    if not DEFAULT_LLM_CONFIG.api_key:
        pytest.skip("HF enricher live test requires HF_TOKEN.")


def test_hf_enricher_live_completes_for_two_endpoints():
    """Run the HF enricher against 2 endpoints and assert the report is
    well-formed. Skip on any provider-side failure.

    The test does not assert that any specific suggestion is accepted —
    only that the full path (HTTP call -> JSON extraction -> Pydantic
    validation -> confidence gate -> field application) does not crash
    on real LLM output.
    """
    _require_hf_token()

    try:
        enricher = HuggingFaceRegistryEnricher(
            model=DEFAULT_LLM_CONFIG.model,
            api_key=DEFAULT_LLM_CONFIG.api_key,
            provider=DEFAULT_LLM_CONFIG.extra.get("hf_provider"),
        )
    except RuntimeError as exc:
        pytest.skip(f"HF enricher setup unavailable: {exc}")

    registry = load_registry(DEFAULT_INPUT_PATH)

    try:
        report = enrich_registry(
            registry,
            enricher=enricher,
            confidence_threshold=0.80,
            max_llm_endpoints=2,
        )
    except RuntimeError as exc:
        # HuggingFaceRegistryEnricher wraps provider errors in RuntimeError.
        # 402 Payment Required, model-unavailable, and transient outages all
        # surface this way. Skipping here keeps CI green on a healthy code
        # path that just doesn't have a live provider available.
        pytest.skip(f"HF enricher live call failed: {exc}")

    accepted_llm = [item for item in report.accepted if item.source == "llm"]
    rejected_llm = [item for item in report.rejected if item.source == "llm"]

    # The report itself should always be well-formed regardless of LLM verdict.
    for suggestion in (*accepted_llm, *rejected_llm):
        assert suggestion.endpoint_id
        assert 0.0 <= suggestion.confidence <= 1.0
        if suggestion.canonical_name is not None:
            from kg_mle.registry.enrichment import CANONICAL_NAMES

            assert suggestion.canonical_name in CANONICAL_NAMES, (
                f"LLM produced canonical_name {suggestion.canonical_name!r} not in whitelist — "
                "Pydantic validator should have rejected this."
            )

    # If anything was accepted, the field must reflect it.
    for suggestion in accepted_llm:
        endpoint = next(
            (
                endpoint
                for endpoint in registry.endpoints
                if endpoint.endpoint_id == suggestion.endpoint_id
            ),
            None,
        )
        assert endpoint is not None, f"accepted suggestion references unknown endpoint {suggestion.endpoint_id!r}"
        field_pool = (
            endpoint.parameters if suggestion.target == "parameter" else endpoint.response_fields
        )
        field = next((item for item in field_pool if item.name == suggestion.field_name), None)
        assert field is not None, f"accepted suggestion field {suggestion.field_name!r} not found on endpoint"
        if suggestion.canonical_name:
            assert field.canonical_name == suggestion.canonical_name
        if suggestion.type_hint:
            assert field.type == suggestion.type_hint
        assert field.enrichment_source == "llm"
