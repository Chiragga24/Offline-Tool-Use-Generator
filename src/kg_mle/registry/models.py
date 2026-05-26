from typing import Literal

from pydantic import BaseModel, Field


ParameterType = Literal["string", "number", "integer", "boolean", "array", "object"]
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


class Parameter(BaseModel):
    name: str
    type: ParameterType = "string"
    required: bool
    description: str
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    type_confidence: float | None = None
    alias_confidence: float | None = None
    enrichment_source: str | None = None


class ResponseField(BaseModel):
    name: str
    type: ParameterType = "string"
    description: str = ""
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    type_confidence: float | None = None
    alias_confidence: float | None = None
    enrichment_source: str | None = None


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
