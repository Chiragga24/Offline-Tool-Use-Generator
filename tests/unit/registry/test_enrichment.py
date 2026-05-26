import pytest
from pydantic import ValidationError

from kg_mle.config import DEFAULT_INPUT_PATH
from kg_mle.registry import (
    FakeRegistryEnricher,
    FieldEnrichmentSuggestion,
    enrich_registry,
    load_registry,
    unresolved_fields,
)


def test_deterministic_registry_enrichment_adds_alias_metadata():
    registry = load_registry(DEFAULT_INPUT_PATH)
    report = enrich_registry(registry)

    endpoint = next(endpoint for endpoint in registry.endpoints if endpoint.endpoint_id == "travel/search_travel_deals")
    destination = next(parameter for parameter in endpoint.parameters if parameter.name == "destination")

    assert report.accepted
    assert destination.canonical_name == "city"
    assert destination.aliases == ["destination"]
    assert destination.alias_confidence == 1.0
    assert destination.enrichment_source == "deterministic"


def test_llm_style_registry_enrichment_accepts_valid_high_confidence_suggestion():
    registry = load_registry(DEFAULT_INPUT_PATH)
    enricher = FakeRegistryEnricher(
        [
            FieldEnrichmentSuggestion(
                endpoint_id="entertainment/get_streaming_availability",
                target="parameter",
                field_name="country",
                canonical_name="location",
                aliases=["country"],
                type_hint="string",
                confidence=0.91,
                reason="Country is a location-like field.",
                source="fake",
            )
        ]
    )

    report = enrich_registry(registry, enricher=enricher, confidence_threshold=0.80)
    endpoint = next(
        endpoint
        for endpoint in registry.endpoints
        if endpoint.endpoint_id == "entertainment/get_streaming_availability"
    )
    country = next(parameter for parameter in endpoint.parameters if parameter.name == "country")

    assert len([item for item in report.accepted if item.source == "fake"]) == 1
    assert country.canonical_name == "location"
    assert country.type == "string"
    assert country.alias_confidence == 0.91
    assert country.enrichment_source == "fake"


def test_registry_enrichment_rejects_low_confidence_suggestion():
    registry = load_registry(DEFAULT_INPUT_PATH)
    enricher = FakeRegistryEnricher(
        [
            FieldEnrichmentSuggestion(
                endpoint_id="entertainment/get_streaming_availability",
                target="parameter",
                field_name="country",
                canonical_name="location",
                confidence=0.50,
                reason="Too uncertain.",
                source="fake",
            )
        ]
    )

    report = enrich_registry(registry, enricher=enricher, confidence_threshold=0.80)

    assert len([item for item in report.rejected if item.source == "fake"]) == 1


def test_registry_enrichment_rejects_unknown_canonical_name():
    with pytest.raises(ValidationError):
        FieldEnrichmentSuggestion(
            endpoint_id="x/y",
            target="parameter",
            field_name="foo",
            canonical_name="totally_unknown",
            confidence=0.99,
        )


def test_registry_enrichment_limits_llm_endpoint_calls():
    registry = load_registry(DEFAULT_INPUT_PATH)
    enricher = FakeRegistryEnricher(
        [
            FieldEnrichmentSuggestion(
                endpoint_id="entertainment/get_streaming_availability",
                target="parameter",
                field_name="country",
                canonical_name="location",
                confidence=0.91,
                source="fake",
            ),
            FieldEnrichmentSuggestion(
                endpoint_id="travel/search_travel_deals",
                target="parameter",
                field_name="destination",
                canonical_name="city",
                confidence=0.91,
                source="fake",
            ),
        ]
    )

    report = enrich_registry(
        registry,
        enricher=enricher,
        confidence_threshold=0.80,
        max_llm_endpoints=1,
    )

    assert len([item for item in report.accepted if item.source == "fake"]) == 1


def test_unresolved_fields_returns_noncanonical_fields():
    registry = load_registry(DEFAULT_INPUT_PATH)
    endpoint = next(endpoint for endpoint in registry.endpoints if endpoint.endpoint_id == "finance/get_quote")

    unresolved = unresolved_fields(endpoint)

    assert ("response_field", "change_pct") in unresolved


def test_enrich_registry_contains_enricher_provider_errors():
    """A provider quota/outage error in the enricher must not abort the
    pipeline. Deterministic enrichment still applies; the failure is
    recorded in report.errors and the LLM pass stops."""

    class _ExplodingEnricher:
        def suggest(self, endpoint):
            raise RuntimeError("Provider HTTP 429: quota exceeded")

    registry = load_registry(DEFAULT_INPUT_PATH)
    report = enrich_registry(registry, enricher=_ExplodingEnricher(), max_llm_endpoints=5)

    # Did not raise; deterministic aliases still applied (destination -> city).
    deal_endpoint = next(
        e for e in registry.endpoints if e.endpoint_id == "travel/search_travel_deals"
    )
    destination = next(p for p in deal_endpoint.parameters if p.name == "destination")
    assert destination.canonical_name == "city"
    # Error recorded, LLM pass stopped after the first failure.
    assert len(report.errors) == 1
    assert "429" in report.errors[0]
    assert not [item for item in report.accepted if item.source == "llm"]
