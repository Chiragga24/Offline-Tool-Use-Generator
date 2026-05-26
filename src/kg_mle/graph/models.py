from typing import Any, Literal

from pydantic import BaseModel, Field


NodeType = Literal["domain", "tool", "endpoint", "parameter", "response_field"]
EdgeType = Literal[
    "contains_tool",
    "exposes_endpoint",
    "requires_parameter",
    "returns_field",
    "output_satisfies_input",
    "same_domain",
    "bridge",
    "semantic_related",
]


class GraphNode(BaseModel):
    node_id: str
    type: NodeType
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    type: EdgeType
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    def node_count(self, node_type: NodeType | None = None) -> int:
        if node_type is None:
            return len(self.nodes)
        return sum(1 for node in self.nodes if node.type == node_type)

    def edge_count(self, edge_type: EdgeType | None = None) -> int:
        if edge_type is None:
            return len(self.edges)
        return sum(1 for edge in self.edges if edge.type == edge_type)

    def get_node(self, node_id: str) -> GraphNode | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def has_edge(self, source: str, target: str, edge_type: EdgeType | None = None) -> bool:
        return any(
            edge.source == source
            and edge.target == target
            and (edge_type is None or edge.type == edge_type)
            for edge in self.edges
        )
