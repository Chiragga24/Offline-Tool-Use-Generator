import json
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from kg_mle.registry.models import Endpoint, ParameterType, ToolRegistry


SuggestionTarget = Literal["parameter", "response_field"]
EnrichmentSource = Literal["deterministic", "llm", "fake"]


CANONICAL_NAMES = {
    "address",
    "alert_id",
    "asset_id",
    "available_time",
    "booking_id",
    "calendar_event_id",
    "city",
    "comparison_id",
    "country",
    "date",
    "deal_id",
    "destination",
    "eval_job_id",
    "event_id",
    "flight_id",
    "forecast_id",
    "game_id",
    "hotel_id",
    "location",
    "menu_id",
    "model_id",
    "movie_id",
    "player_id",
    "price",
    "query",
    "restaurant_id",
    "slot_id",
    "start_time",
    "symbol",
    "team_id",
    "ticket_offer_id",
    "time",
    "tournament_id",
    "venue",
}


DETERMINISTIC_ALIASES = {
    "destination": "city",
    "venue": "location",
    "available_time": "start_time",
    "check_in": "date",
}


class FieldEnrichmentSuggestion(BaseModel):
    endpoint_id: str
    target: SuggestionTarget
    field_name: str
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    type_hint: ParameterType | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    source: EnrichmentSource = "llm"

    @field_validator("canonical_name")
    @classmethod
    def validate_canonical_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in CANONICAL_NAMES:
            raise ValueError(f"Unsupported canonical_name: {value}")
        return value


class RegistryEnrichmentReport(BaseModel):
    accepted: list[FieldEnrichmentSuggestion] = Field(default_factory=list)
    rejected: list[FieldEnrichmentSuggestion] = Field(default_factory=list)


class RegistryEnricher(Protocol):
    def suggest(self, endpoint: Endpoint) -> list[FieldEnrichmentSuggestion]:
        """Return structured enrichment suggestions for one endpoint."""


class FakeRegistryEnricher:
    def __init__(self, suggestions: list[FieldEnrichmentSuggestion]) -> None:
        self.suggestions = suggestions

    def suggest(self, endpoint: Endpoint) -> list[FieldEnrichmentSuggestion]:
        return [
            suggestion
            for suggestion in self.suggestions
            if suggestion.endpoint_id == endpoint.endpoint_id
        ]


class HuggingFaceRegistryEnricher:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        provider: str | None = None,
        max_unresolved_fields: int = 8,
    ) -> None:
        if not api_key:
            raise RuntimeError("Hugging Face registry enrichment requires HF_TOKEN.")
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face registry enrichment requires huggingface-hub."
            ) from exc

        self.model = model
        self.max_unresolved_fields = max_unresolved_fields
        self._client = InferenceClient(model=model, token=api_key, provider=provider)

    def suggest(self, endpoint: Endpoint) -> list[FieldEnrichmentSuggestion]:
        unresolved = unresolved_fields(endpoint)[: self.max_unresolved_fields]
        if not unresolved:
            return []

        prompt = _build_enrichment_prompt(endpoint, unresolved)
        try:
            content = self._complete_json(prompt)
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face registry enrichment failed for model {self.model}. "
                "The configured model may not be available through the Hugging Face "
                "Inference API/router for chat or text-generation. Use a hosted "
                "instruction model, configure a local provider later, or run without "
                "--llm-enrich-registry."
            ) from exc
        payload = _extract_json_object(content)
        raw_suggestions = payload.get("suggestions", [])
        if not isinstance(raw_suggestions, list):
            return []

        suggestions: list[FieldEnrichmentSuggestion] = []
        for raw in raw_suggestions:
            if not isinstance(raw, dict):
                continue
            raw.setdefault("endpoint_id", endpoint.endpoint_id)
            raw.setdefault("source", "llm")
            try:
                suggestions.append(FieldEnrichmentSuggestion.model_validate(raw))
            except ValidationError:
                continue
        return suggestions

    def _complete_json(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You normalize API schema fields. Return only valid JSON. "
                    "Do not include markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(self._client, "chat_completion"):
            try:
                response = self._client.chat_completion(
                    messages=messages,
                    max_tokens=700,
                    temperature=0.0,
                    top_p=1.0,
                    response_format={"type": "json_object"},
                )
                message = response.choices[0].message
                return message.content or ""
            except Exception:
                # Some Hugging Face-hosted instruction models are text-generation
                # models, not chat-completion models. Fall back to a single prompt.
                pass

        response = self._client.text_generation(
            (
                "Return only valid JSON for the following structured API "
                f"normalization task:\n{prompt}"
            ),
            max_new_tokens=700,
            return_full_text=False,
        )
        return str(response)


def enrich_registry(
    registry: ToolRegistry,
    *,
    enricher: RegistryEnricher | None = None,
    confidence_threshold: float = 0.80,
    max_llm_endpoints: int | None = None,
) -> RegistryEnrichmentReport:
    report = RegistryEnrichmentReport()
    _apply_deterministic_aliases(registry, report)

    if enricher is None:
        return report

    enriched_endpoint_count = 0
    for endpoint in registry.endpoints:
        if max_llm_endpoints is not None and enriched_endpoint_count >= max_llm_endpoints:
            break
        suggestions = enricher.suggest(endpoint)
        if suggestions:
            enriched_endpoint_count += 1
        for suggestion in suggestions:
            if suggestion.confidence < confidence_threshold:
                report.rejected.append(suggestion)
                continue
            if _apply_suggestion(endpoint, suggestion):
                report.accepted.append(suggestion)
            else:
                report.rejected.append(suggestion)

    return report


def unresolved_fields(endpoint: Endpoint) -> list[tuple[SuggestionTarget, str]]:
    unresolved: list[tuple[SuggestionTarget, str]] = []
    for parameter in endpoint.parameters:
        if parameter.canonical_name is None and parameter.name not in CANONICAL_NAMES:
            unresolved.append(("parameter", parameter.name))
    for field in endpoint.response_fields:
        if field.canonical_name is None and field.name not in CANONICAL_NAMES:
            unresolved.append(("response_field", field.name))
    return unresolved


def _build_enrichment_prompt(
    endpoint: Endpoint,
    unresolved: list[tuple[SuggestionTarget, str]],
) -> str:
    allowed_types = ["string", "number", "integer", "boolean", "array", "object"]
    return json.dumps(
        {
            "task": "Suggest canonical aliases and type hints for unresolved API fields.",
            "rules": [
                "Return JSON with a top-level suggestions array.",
                "Only suggest fields listed in unresolved_fields.",
                "Use canonical_name only from allowed_canonical_names.",
                "Use type_hint only from allowed_types.",
                "Use confidence between 0 and 1.",
                "Do not delete or rename original fields.",
                "If uncertain, return no suggestion for that field.",
            ],
            "output_schema": {
                "suggestions": [
                    {
                        "target": "parameter | response_field",
                        "field_name": "original field name",
                        "canonical_name": "allowed canonical name or null",
                        "aliases": ["zero or more aliases"],
                        "type_hint": "allowed type or null",
                        "confidence": 0.0,
                        "reason": "short reason",
                    }
                ]
            },
            "allowed_canonical_names": sorted(CANONICAL_NAMES),
            "allowed_types": allowed_types,
            "endpoint": {
                "endpoint_id": endpoint.endpoint_id,
                "description": endpoint.description,
                "parameters": [parameter.model_dump() for parameter in endpoint.parameters],
                "response_fields": [field.model_dump() for field in endpoint.response_fields],
            },
            "unresolved_fields": [
                {"target": target, "field_name": field_name}
                for target, field_name in unresolved
            ],
        },
        indent=2,
    )


def _extract_json_object(content: str) -> dict:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    return json.loads(stripped[start : end + 1])


def _apply_deterministic_aliases(
    registry: ToolRegistry,
    report: RegistryEnrichmentReport,
) -> None:
    for endpoint in registry.endpoints:
        for parameter in endpoint.parameters:
            canonical = DETERMINISTIC_ALIASES.get(parameter.name)
            if canonical:
                parameter.canonical_name = canonical
                parameter.aliases = sorted(set([*parameter.aliases, parameter.name]))
                parameter.alias_confidence = 1.0
                parameter.enrichment_source = "deterministic"
                report.accepted.append(
                    FieldEnrichmentSuggestion(
                        endpoint_id=endpoint.endpoint_id,
                        target="parameter",
                        field_name=parameter.name,
                        canonical_name=canonical,
                        aliases=[parameter.name],
                        confidence=1.0,
                        reason="Known deterministic alias.",
                        source="deterministic",
                    )
                )

        for field in endpoint.response_fields:
            canonical = DETERMINISTIC_ALIASES.get(field.name)
            if canonical:
                field.canonical_name = canonical
                field.aliases = sorted(set([*field.aliases, field.name]))
                field.alias_confidence = 1.0
                field.enrichment_source = "deterministic"
                report.accepted.append(
                    FieldEnrichmentSuggestion(
                        endpoint_id=endpoint.endpoint_id,
                        target="response_field",
                        field_name=field.name,
                        canonical_name=canonical,
                        aliases=[field.name],
                        confidence=1.0,
                        reason="Known deterministic alias.",
                        source="deterministic",
                    )
                )


def _apply_suggestion(endpoint: Endpoint, suggestion: FieldEnrichmentSuggestion) -> bool:
    fields = endpoint.parameters if suggestion.target == "parameter" else endpoint.response_fields
    field = next((item for item in fields if item.name == suggestion.field_name), None)
    if field is None:
        return False

    if suggestion.canonical_name:
        field.canonical_name = suggestion.canonical_name
        field.alias_confidence = suggestion.confidence
    if suggestion.aliases:
        field.aliases = sorted(set([*field.aliases, *suggestion.aliases]))
        field.alias_confidence = suggestion.confidence
    if suggestion.type_hint:
        field.type = suggestion.type_hint
        field.type_confidence = suggestion.confidence
    field.enrichment_source = suggestion.source
    return True
