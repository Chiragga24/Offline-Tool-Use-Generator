import json

import pytest

from kg_mle.config import DEFAULT_INPUT_PATH
from kg_mle.registry import load_registry, normalize_tools, save_registry
from kg_mle.registry.loader import normalize_type


def test_load_registry_normalizes_messy_fixture():
    registry = load_registry(DEFAULT_INPUT_PATH)

    assert len(registry.tools) == 9
    assert registry.endpoint_count() == 45

    endpoints = {endpoint.endpoint_id: endpoint for endpoint in registry.endpoints}

    assert endpoints["finance/get_quote"].method == "GET"
    assert endpoints["finance/get_quote"].parameters[0].type == "string"
    assert endpoints["finance/get_company_news"].response_fields[0].name == "articles"
    assert endpoints["sports/get_player_stats"].description == "Get Player Stats endpoint."
    assert endpoints["sports/get_player_stats"].parameters[0].type == "string"
    assert endpoints["ai_ml/create_eval_job"].method == "GET"
    assert endpoints["entertainment/search_movies"].response_fields == []
    assert endpoints["travel/search_flights"].path == "/travel/search_flights"
    assert endpoints["travel/search_hotels"].parameters[-1].required is False
    assert endpoints["gaming/get_player_profile"].response_fields == []
    assert endpoints["weather/get_active_alerts"].description == "Get Active Alerts endpoint."
    assert endpoints["weather/get_hourly_forecast"].parameters[0].required is True


def test_normalize_type_defaults_unknown_or_missing_to_string():
    assert normalize_type("STRING") == "string"
    assert normalize_type("integer") == "integer"
    assert normalize_type("ARRAY") == "array"
    assert normalize_type("UNKNOWN") == "string"
    assert normalize_type(None) == "string"


def test_normalize_tools_handles_minimal_raw_tool():
    registry = normalize_tools(
        [
            {
                "category": "Custom Category",
                "tool_name": "Minimal Tool",
                "api_list": [
                    {
                        "api_name": "Do Thing",
                        "required": [{"name": "item_id"}],
                    }
                ],
            }
        ]
    )

    endpoint = registry.endpoints[0]

    assert registry.tools[0].domain == "custom_category"
    assert registry.tools[0].tool_name == "minimal_tool"
    assert endpoint.endpoint_id == "custom_category/do_thing"
    assert endpoint.method == "GET"
    assert endpoint.path == "/custom_category/do_thing"
    assert endpoint.description == "Do Thing endpoint."
    assert endpoint.parameters[0].name == "item_id"
    assert endpoint.parameters[0].type == "string"
    assert endpoint.parameters[0].required is True


def test_normalize_tools_handles_flat_response_field_map():
    """ToolBench data often uses flat {field: type} response maps, not JSON Schema."""
    registry = normalize_tools(
        [
            {
                "category": "Travel",
                "tool_name": "flat_hotels",
                "api_list": [
                    {
                        "name": "search",
                        "required_parameters": [{"name": "city", "type": "string"}],
                        "response_schema": {
                            "hotel_id": "string",
                            "name": "STRING",
                            "price": "number",
                        },
                    }
                ],
            }
        ]
    )

    endpoint = registry.endpoints[0]
    field_types = {field.name: field.type for field in endpoint.response_fields}
    assert field_types == {"hotel_id": "string", "name": "string", "price": "number"}


def test_normalize_tools_handles_list_response_field_shape():
    """Some ToolBench-style schemas use [{name, type, description}, ...]."""
    registry = normalize_tools(
        [
            {
                "category": "Finance",
                "tool_name": "quotes",
                "api_list": [
                    {
                        "name": "get_quote",
                        "required_parameters": [{"name": "symbol", "type": "string"}],
                        "response_schema": [
                            {"name": "price", "type": "number", "description": "Last trade."},
                            {"name": "currency", "type": "string"},
                            {"type": "string", "description": "skipped — no name."},
                            {"field": "exchange", "type": "string"},
                        ],
                    }
                ],
            }
        ]
    )

    endpoint = registry.endpoints[0]
    field_names = [field.name for field in endpoint.response_fields]
    assert field_names == ["price", "currency", "exchange"]
    assert endpoint.response_fields[0].description == "Last trade."


def test_normalize_tools_does_not_misinterpret_json_schema_shell_as_flat_map():
    """A response_schema with only `type` and `properties` should stay JSON-Schema."""
    registry = normalize_tools(
        [
            {
                "category": "Sports",
                "tool_name": "teams",
                "api_list": [
                    {
                        "name": "get_team",
                        "required_parameters": [{"name": "team_id", "type": "string"}],
                        "response_schema": {
                            "type": "object",
                            "properties": {"team_id": {"type": "string"}},
                        },
                    }
                ],
            }
        ]
    )

    endpoint = registry.endpoints[0]
    assert [field.name for field in endpoint.response_fields] == ["team_id"]


def test_normalize_tools_raises_on_duplicate_endpoint_id():
    raw_tools = [
        {
            "category": "Travel",
            "tool_name": "flights_tool",
            "api_list": [{"name": "search", "required": [{"name": "city"}]}],
        },
        {
            "category": "Travel",
            "tool_name": "hotels_tool",
            "api_list": [{"name": "search", "required": [{"name": "city"}]}],
        },
    ]

    with pytest.raises(ValueError, match="Duplicate endpoint_id 'travel/search'"):
        normalize_tools(raw_tools)


def test_save_registry_round_trips_json(tmp_path):
    registry = load_registry(DEFAULT_INPUT_PATH)
    path = save_registry(registry, tmp_path / "registry.json")

    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["tools"][0]["domain"] == "finance"
    assert sum(len(tool["endpoints"]) for tool in saved["tools"]) == 45

