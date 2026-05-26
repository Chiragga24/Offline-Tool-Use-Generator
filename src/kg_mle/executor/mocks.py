"""Schema-derived mock response generation.

The mock layer's job is to produce a response dict for one endpoint call
that is:

1. Schema-consistent — every declared response field is present with a
   type-correct value, so a downstream parser cannot tell the difference
   between a mocked call and a real one.
2. Chain-consistent — ID-shaped fields produce fresh values *and* are
   registered into `SessionState` so the next call's grounded argument
   references them rather than a hallucinated string.
3. Realistic enough — canonical-named fields (city, date, country, …)
   draw from a small example pool so the dataset doesn't read like
   `name_a3f9` everywhere. Unknown fields fall back to a deterministic
   template; the realism is bounded but the determinism is total.

Determinism is achieved with a per-session `random.Random` instance.
Same session seed + same chain of calls = byte-identical responses.
"""

from __future__ import annotations

import random
import string
from typing import Any

from kg_mle.executor.state import SessionState
from kg_mle.registry.models import Endpoint, ResponseField


# Realistic value pools keyed by canonical_name. Kept short on purpose:
# tests can assert that values are drawn from these pools without
# enumerating thousands of options.
CANONICAL_EXAMPLES: dict[str, tuple[str, ...]] = {
    "city": ("Paris", "Tokyo", "Berlin", "São Paulo", "Cape Town", "Mumbai", "Lisbon"),
    "country": ("FR", "JP", "DE", "BR", "ZA", "IN", "PT", "US"),
    "location": ("Downtown", "Midtown", "Old Town", "Harbor District"),
    "address": ("12 Rue de Rivoli", "1 Shibuya Crossing", "200 Hauptstraße"),
    "date": ("2026-04-11", "2026-05-22", "2026-06-04", "2026-07-19"),
    "time": ("09:00", "14:30", "19:00", "21:15"),
    "start_time": ("2026-04-11T09:00", "2026-05-22T14:30", "2026-06-04T19:00"),
    "available_time": ("2026-04-11T18:00", "2026-05-22T20:00"),
    "venue": ("Le Grand Hall", "Tokyo Dome", "Festhalle", "Estádio Central"),
    "query": ("hotels near downtown", "weekend events", "vegetarian restaurants"),
    "symbol": ("AAPL", "MSFT", "BTC", "ETH", "TSLA", "NVDA"),
    "price": ("175.00", "299.50", "42.99", "1250.00"),
}


# Per-canonical-name ID prefixes. Anything not listed gets a prefix
# derived from the field name itself (first letter + length).
ID_PREFIXES: dict[str, str] = {
    "hotel_id": "htl_",
    "flight_id": "flt_",
    "booking_id": "bk_",
    "event_id": "evt_",
    "ticket_offer_id": "tkt_",
    "calendar_event_id": "cal_",
    "menu_id": "menu_",
    "restaurant_id": "rest_",
    "slot_id": "slot_",
    "team_id": "team_",
    "player_id": "plr_",
    "game_id": "gm_",
    "tournament_id": "trn_",
    "model_id": "mdl_",
    "eval_job_id": "eval_",
    "asset_id": "ast_",
    "alert_id": "alrt_",
    "comparison_id": "cmp_",
    "deal_id": "deal_",
    "forecast_id": "fcst_",
    "movie_id": "mov_",
    "article_id": "art_",
}


class MockResponseGenerator:
    """Generates one response dict per endpoint call.

    The generator owns a `random.Random` instance keyed off the session
    seed; the SessionState tracks issued IDs for later grounding.
    """

    def __init__(self, *, seed: int) -> None:
        self._rng = random.Random(seed)
        self._id_counter = 0

    def generate(
        self,
        endpoint: Endpoint,
        session_state: SessionState,  # kept for signature stability; registration
                                       # is done by ExecutorSession after this call.
    ) -> dict[str, Any]:
        response: dict[str, Any] = {}
        for field in endpoint.response_fields:
            response[field.name] = self._field_value(field)
        return response

    def _field_value(self, field: ResponseField) -> Any:
        if _is_id_field(field):
            return self._mint_id(field)

        canonical = field.canonical_name or field.name
        if canonical in CANONICAL_EXAMPLES:
            return self._rng.choice(CANONICAL_EXAMPLES[canonical])

        return self._value_for_type(field)

    def _mint_id(self, field: ResponseField) -> str:
        canonical = field.canonical_name or field.name
        prefix = ID_PREFIXES.get(canonical) or _derive_prefix(field.name)
        self._id_counter += 1
        suffix = "".join(self._rng.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{prefix}{suffix}"

    def _value_for_type(self, field: ResponseField) -> Any:
        if field.type == "string":
            return f"{field.name.replace('_', ' ').title()} {self._rng.randint(1, 99)}"
        if field.type == "integer":
            return self._rng.randint(1, 100)
        if field.type == "number":
            return round(self._rng.uniform(1.0, 500.0), 2)
        if field.type == "boolean":
            return self._rng.choice([True, False])
        if field.type == "array":
            # Two short string items; downstream judge can spot-check structure.
            return [
                f"{field.name}_item_{i}_{self._rng.randint(100, 999)}"
                for i in range(2)
            ]
        if field.type == "object":
            return {"summary": f"{field.name} payload"}
        return f"{field.name}_value"


def _is_id_field(field: ResponseField) -> bool:
    """A field is treated as ID-shaped if its name or canonical_name ends in
    `_id` (or equals `id`). This drives both prefix-based generation and
    session tracking for grounded transitions."""
    candidates = {field.name}
    if field.canonical_name:
        candidates.add(field.canonical_name)
    return any(name == "id" or name.endswith("_id") for name in candidates)


def _derive_prefix(field_name: str) -> str:
    """Fallback prefix when the field name isn't in ID_PREFIXES."""
    stem = field_name.removesuffix("_id") or field_name
    return f"{stem[:3]}_" if stem else "id_"


def is_id_parameter(name: str, canonical_name: str | None) -> bool:
    """Public helper used by the validator to know which parameters require
    chain-grounded values."""
    candidates = {name}
    if canonical_name:
        candidates.add(canonical_name)
    return any(item == "id" or item.endswith("_id") for item in candidates)
