import json

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


def test_save_registry_round_trips_json(tmp_path):
    registry = load_registry(DEFAULT_INPUT_PATH)
    path = save_registry(registry, tmp_path / "registry.json")

    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["tools"][0]["domain"] == "finance"
    assert sum(len(tool["endpoints"]) for tool in saved["tools"]) == 45

