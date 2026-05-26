import json
import re
from pathlib import Path
from typing import Any

from kg_mle.registry.models import Endpoint, Parameter, ParameterType, ResponseField, Tool, ToolRegistry


TYPE_ALIASES: dict[str, ParameterType] = {
    "str": "string",
    "string": "string",
    "text": "string",
    "number": "number",
    "float": "number",
    "double": "number",
    "integer": "integer",
    "int": "integer",
    "bool": "boolean",
    "boolean": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
    "json": "object",
}


METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def load_registry(input_path: Path | str) -> ToolRegistry:
    path = Path(input_path)
    raw_tools = _load_raw_tools(path)
    return normalize_tools(raw_tools)


def normalize_tools(raw_tools: list[dict[str, Any]]) -> ToolRegistry:
    tools: list[Tool] = []
    for raw_tool in raw_tools:
        category = str(raw_tool.get("category") or "uncategorized")
        domain = _normalize_domain(category)
        tool_name = _slugify(str(raw_tool.get("tool_name") or category or "tool"))
        description = str(
            raw_tool.get("tool_description")
            or raw_tool.get("description")
            or f"ToolBench-style APIs for {category}."
        )

        endpoints = [
            _normalize_endpoint(raw_api, category=category, domain=domain, tool_name=tool_name)
            for raw_api in raw_tool.get("api_list", []) or []
        ]
        tools.append(
            Tool(
                domain=domain,
                category=category,
                tool_name=tool_name,
                description=description,
                endpoints=endpoints,
            )
        )

    return ToolRegistry(tools=tools)


def save_registry(registry: ToolRegistry, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    return path


def _load_raw_tools(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return _read_json_file(path)

    if path.is_dir():
        tools: list[dict[str, Any]] = []
        for json_path in sorted(path.rglob("*.json")):
            tools.extend(_read_json_file(json_path))
        return tools

    raise FileNotFoundError(path)


def _read_json_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON root in {path}: {type(data).__name__}")


def _normalize_endpoint(
    raw_api: dict[str, Any],
    *,
    category: str,
    domain: str,
    tool_name: str,
) -> Endpoint:
    name = _slugify(str(raw_api.get("name") or raw_api.get("api_name") or "unnamed_endpoint"))
    method = str(raw_api.get("method") or "GET").upper()
    if method not in METHODS:
        method = "GET"

    path = str(raw_api.get("url") or raw_api.get("path") or f"/{domain}/{name}")
    description = str(raw_api.get("description") or f"{name.replace('_', ' ').title()} endpoint.")

    parameters = [
        *_normalize_parameters(raw_api, "required", required=True),
        *_normalize_parameters(raw_api, "optional", required=False),
    ]
    response_fields = _normalize_response_fields(raw_api)

    return Endpoint(
        endpoint_id=f"{domain}/{name}",
        domain=domain,
        category=category,
        tool_name=tool_name,
        name=name,
        method=method,  # type: ignore[arg-type]
        path=path,
        description=description,
        parameters=parameters,
        response_fields=response_fields,
        raw_schema=raw_api,
    )


def _normalize_parameters(raw_api: dict[str, Any], kind: str, *, required: bool) -> list[Parameter]:
    keys = (
        ["required_parameters", "required"]
        if kind == "required"
        else ["optional_parameters", "optional_params", "optionalParameters", "optional"]
    )
    raw_params: Any = None
    for key in keys:
        if key in raw_api:
            raw_params = raw_api[key]
            break

    if not isinstance(raw_params, list):
        return []

    params: list[Parameter] = []
    for raw_param in raw_params:
        if not isinstance(raw_param, dict):
            continue
        name = str(raw_param.get("name") or "").strip()
        if not name:
            continue
        param_type = normalize_type(raw_param.get("type"))
        description = str(
            raw_param.get("description")
            or raw_param.get("desc")
            or f"{name.replace('_', ' ')} parameter."
        )
        params.append(
            Parameter(
                name=name,
                type=param_type,
                required=required,
                description=description,
            )
        )
    return params


def _normalize_response_fields(raw_api: dict[str, Any]) -> list[ResponseField]:
    schema = raw_api.get("response_schema")
    if not isinstance(schema, dict) or not schema:
        schema = raw_api.get("response")
    if not isinstance(schema, dict):
        return []

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []

    fields: list[ResponseField] = []
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            field_schema = {}
        fields.append(
            ResponseField(
                name=str(field_name),
                type=normalize_type(field_schema.get("type")),
                description=str(field_schema.get("description") or ""),
            )
        )
    return fields


def normalize_type(raw_type: Any) -> ParameterType:
    if not raw_type:
        return "string"
    key = str(raw_type).strip().lower()
    return TYPE_ALIASES.get(key, "string")


def _normalize_domain(category: str) -> str:
    if category == "Artificial_Intelligence_Machine_Learning":
        return "ai_ml"
    return _slugify(category)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"

