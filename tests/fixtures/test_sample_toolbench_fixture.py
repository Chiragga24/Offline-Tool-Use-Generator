import json
from pathlib import Path


FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_toolbench" / "tools.json"


def test_sample_toolbench_fixture_is_valid_json():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert isinstance(data, list)
    assert len(data) == 9
    assert sum(len(tool.get("api_list", [])) for tool in data) == 45


def test_sample_toolbench_fixture_has_expected_categories():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    categories = {tool["category"] for tool in data}

    assert categories == {
        "Finance",
        "Sports",
        "Artificial_Intelligence_Machine_Learning",
        "Entertainment",
        "Travel",
        "Gaming",
        "Events",
        "Food",
        "Weather",
    }


def test_each_toolbench_endpoint_has_required_shape():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    for tool in data:
        assert tool["category"]
        assert tool["tool_name"]
        assert tool["tool_description"]
        assert tool["api_list"]

        for api in tool["api_list"]:
            assert api["name"]
            assert (api.get("url") or api.get("path")).startswith("/")
            assert isinstance(
                api.get("required_parameters") or api.get("required") or [],
                list,
            )
            assert isinstance(
                api.get("optional_parameters")
                or api.get("optional_params")
                or api.get("optionalParameters")
                or [],
                list,
            )


def test_endpoint_names_are_unique_within_category():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    for tool in data:
        names = [api["name"] for api in tool["api_list"]]
        assert len(names) == len(set(names)), tool["tool_name"]


def test_fixture_contains_intentionally_messy_toolbench_cases():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    apis = [api for tool in data for api in tool["api_list"]]

    assert any(api.get("method") == "get" for api in apis)
    assert any("path" in api and "url" not in api for api in apis)
    assert any(api.get("response_schema") in ({}, None) for api in apis)
    assert any("response" in api and "response_schema" not in api for api in apis)
    assert any("required" in api and "required_parameters" not in api for api in apis)
    assert any(
        any("type" not in param for param in api.get("required_parameters", []))
        for api in apis
    )

