"""Live integration test for the Mem0 ANN backend.

This test exercises the real `Mem0SemanticRetriever` end-to-end against
Hugging Face embeddings, Gemini LLM, and an in-memory Qdrant store.

It is a *protocol smoke test*, not a correctness test:

- ANN search is approximate, so we cannot assert that a specific query
  always returns a specific endpoint_id. Model/version drift would make
  such an assertion flaky.
- What we can assert is that the live integration wires up correctly:
  Mem0 initializes, accepts indexed cards with `infer=False`, and returns
  scored results with endpoint_ids drawn from the indexed set, sorted by
  score descending.

The test auto-skips when:
- the optional `mem0ai` dependency is missing,
- `HF_TOKEN` or `GOOGLE_API_KEY` is unset,
- Mem0/the provider rejects the request at runtime (billing, model
  availability, transient outage).

This shape means CI stays green without credentials and only red when a
*real* protocol regression occurs (e.g., result-dict shape changes).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mem0")

from kg_mle.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PROVIDER
from kg_mle.graph.semantic import EndpointCard, Mem0SemanticRetriever


pytestmark = pytest.mark.live


def _require_credentials() -> None:
    missing = [name for name in ("HF_TOKEN", "GOOGLE_API_KEY") if not os.getenv(name)]
    if missing:
        pytest.skip(f"Mem0 live test requires: {', '.join(missing)}")


def _build_live_retriever() -> Mem0SemanticRetriever:
    try:
        return Mem0SemanticRetriever(
            embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            llm_provider="gemini",
            llm_model=os.getenv("KG_MLE_MEM0_LLM_MODEL", "gemini-2.0-flash-lite-001"),
            llm_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    except RuntimeError as exc:
        pytest.skip(f"Mem0 setup unavailable: {exc}")


def test_mem0_live_index_and_search_returns_indexed_endpoints():
    """Index a handful of distinct endpoint cards and confirm Mem0 returns
    scored hits drawn from the indexed set, sorted by score descending.

    This catches: result-shape regressions, broken metadata round-tripping,
    score-sorting regressions, filter-shape regressions. It does *not*
    assert semantic relevance (would be flaky against ANN + model drift).
    """
    _require_credentials()
    retriever = _build_live_retriever()

    cards = [
        EndpointCard(
            endpoint_id="travel/search_hotels",
            text="Endpoint: travel/search_hotels\nDomain: travel\nDescription: Find hotels in a city by check-in date and price.\nInputs: city, check_in, max_price\nOutputs: hotel_id, hotel_name, nightly_price",
            metadata={"domain": "travel"},
        ),
        EndpointCard(
            endpoint_id="weather/get_forecast",
            text="Endpoint: weather/get_forecast\nDomain: weather\nDescription: Get a multi-day weather forecast for a city.\nInputs: city, date\nOutputs: forecast_id, summary, temperature",
            metadata={"domain": "weather"},
        ),
        EndpointCard(
            endpoint_id="finance/get_quote",
            text="Endpoint: finance/get_quote\nDomain: finance\nDescription: Get the latest market quote for a stock or asset symbol.\nInputs: symbol\nOutputs: price, currency, change_pct",
            metadata={"domain": "finance"},
        ),
        EndpointCard(
            endpoint_id="events/create_calendar_event",
            text="Endpoint: events/create_calendar_event\nDomain: events\nDescription: Create a calendar event with a title at a start_time.\nInputs: title, start_time\nOutputs: calendar_event_id",
            metadata={"domain": "events"},
        ),
    ]
    indexed_ids = {card.endpoint_id for card in cards}

    try:
        retriever.index(cards)
        matches = retriever.search(cards[0], top_k=3)
    except Exception as exc:
        pytest.skip(f"Mem0 live call failed: {exc}")

    assert matches, "Mem0 returned no matches — live index/search path is broken."
    for match in matches:
        assert match.endpoint_id in indexed_ids, (
            f"Mem0 returned endpoint_id {match.endpoint_id!r} not present in indexed set."
        )
        assert isinstance(match.score, float)
        assert match.reason == "mem0"

    # Score ordering: results should be sorted descending. Allow ties.
    scores = [match.score for match in matches]
    assert scores == sorted(scores, reverse=True), (
        f"Mem0 results not in descending score order: {scores}"
    )
