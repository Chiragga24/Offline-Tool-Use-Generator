from kg_mle.registry.enrichment import _extract_json_object


def test_extract_json_object_handles_markdown_fence():
    payload = _extract_json_object(
        """```json
        {"suggestions": [{"field_name": "country"}]}
        ```"""
    )

    assert payload["suggestions"][0]["field_name"] == "country"

