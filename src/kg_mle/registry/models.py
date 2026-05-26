from typing import Any, Literal

from pydantic import BaseModel, Field


ParameterType = Literal["string", "number", "integer", "boolean", "array", "object"]
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


class Parameter(BaseModel):
    name: str
    type: ParameterType = "string"
    required: bool
    description: str


class ResponseField(BaseModel):
    name: str
    type: ParameterType = "string"
    description: str = ""


class Endpoint(BaseModel):
    endpoint_id: str
    domain: str
    category: str
    tool_name: str
    name: str
    method: HttpMethod = "GET"
    path: str
    description: str
    parameters: list[Parameter] = Field(default_factory=list)
    response_fields: list[ResponseField] = Field(default_factory=list)
    raw_schema: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    domain: str
    category: str
    tool_name: str
    description: str
    endpoints: list[Endpoint] = Field(default_factory=list)


class ToolRegistry(BaseModel):
    tools: list[Tool] = Field(default_factory=list)

    @property
    def endpoints(self) -> list[Endpoint]:
        return [endpoint for tool in self.tools for endpoint in tool.endpoints]

    def endpoint_count(self) -> int:
        return len(self.endpoints)

