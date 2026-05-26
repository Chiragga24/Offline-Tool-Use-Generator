from collections import defaultdict
from pathlib import Path

from kg_mle.graph.models import GraphEdge, GraphNode, ToolGraph
from kg_mle.graph.semantic import EndpointCard, SemanticRetriever
from kg_mle.registry.models import Endpoint, Parameter, ResponseField, ToolRegistry


def build_tool_graph(
    registry: ToolRegistry,
    *,
    semantic_retriever: SemanticRetriever | None = None,
    semantic_threshold: float = 0.72,
    semantic_top_k: int = 5,
) -> ToolGraph:
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    edge_pairs: set[tuple[str, str]] = set()

    def add_node(node: GraphNode) -> None:
        nodes.setdefault(node.node_id, node)

    def add_edge(edge: GraphEdge) -> None:
        key = (edge.source, edge.target, edge.type)
        if edge.source == edge.target or key in edge_keys:
            return
        edge_keys.add(key)
        edge_pairs.add((edge.source, edge.target))
        edges.append(edge)

    for tool in registry.tools:
        domain_node_id = f"domain:{tool.domain}"
        tool_node_id = f"tool:{tool.domain}/{tool.tool_name}"
        add_node(
            GraphNode(
                node_id=domain_node_id,
                type="domain",
                label=tool.domain,
                metadata={"category": tool.category},
            )
        )
        add_node(
            GraphNode(
                node_id=tool_node_id,
                type="tool",
                label=tool.tool_name,
                metadata={"domain": tool.domain, "description": tool.description},
            )
        )
        add_edge(GraphEdge(source=domain_node_id, target=tool_node_id, type="contains_tool"))

        for endpoint in tool.endpoints:
            endpoint_node_id = _endpoint_node_id(endpoint.endpoint_id)
            add_node(
                GraphNode(
                    node_id=endpoint_node_id,
                    type="endpoint",
                    label=endpoint.name,
                    metadata={
                        "endpoint_id": endpoint.endpoint_id,
                        "domain": endpoint.domain,
                        "method": endpoint.method,
                        "path": endpoint.path,
                        "description": endpoint.description,
                    },
                )
            )
            add_edge(GraphEdge(source=tool_node_id, target=endpoint_node_id, type="exposes_endpoint"))

            for parameter in endpoint.parameters:
                parameter_node_id = f"parameter:{endpoint.endpoint_id}.{parameter.name}"
                add_node(
                    GraphNode(
                        node_id=parameter_node_id,
                        type="parameter",
                        label=parameter.name,
                        metadata={
                            "endpoint_id": endpoint.endpoint_id,
                            "type": parameter.type,
                            "required": parameter.required,
                            "description": parameter.description,
                        },
                    )
                )
                add_edge(
                    GraphEdge(
                        source=endpoint_node_id,
                        target=parameter_node_id,
                        type="requires_parameter",
                        metadata={"required": parameter.required},
                    )
                )

            for field in endpoint.response_fields:
                field_node_id = f"response_field:{endpoint.endpoint_id}.{field.name}"
                add_node(
                    GraphNode(
                        node_id=field_node_id,
                        type="response_field",
                        label=field.name,
                        metadata={
                            "endpoint_id": endpoint.endpoint_id,
                            "type": field.type,
                            "description": field.description,
                        },
                    )
                )
                add_edge(GraphEdge(source=endpoint_node_id, target=field_node_id, type="returns_field"))

    endpoints = registry.endpoints
    _add_output_to_input_edges(endpoints, add_edge)
    _add_same_domain_edges(endpoints, add_edge)

    if semantic_retriever is not None:
        _add_semantic_edges(
            endpoints,
            add_edge,
            semantic_retriever=semantic_retriever,
            threshold=semantic_threshold,
            top_k=semantic_top_k,
            existing_pairs=edge_pairs,
        )

    return ToolGraph(nodes=sorted(nodes.values(), key=lambda node: node.node_id), edges=edges)


def save_tool_graph(graph: ToolGraph, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_tool_graph(input_path: Path | str) -> ToolGraph:
    path = Path(input_path)
    return ToolGraph.model_validate_json(path.read_text(encoding="utf-8"))


def endpoint_card(endpoint: Endpoint) -> EndpointCard:
    inputs = ", ".join(parameter.name for parameter in endpoint.parameters) or "none"
    outputs = ", ".join(field.name for field in endpoint.response_fields) or "unknown"
    text = "\n".join(
        [
            f"Endpoint: {endpoint.endpoint_id}",
            f"Domain: {endpoint.domain}",
            f"Description: {endpoint.description}",
            f"Inputs: {inputs}",
            f"Outputs: {outputs}",
        ]
    )
    return EndpointCard(
        endpoint_id=endpoint.endpoint_id,
        text=text,
        metadata={"domain": endpoint.domain, "name": endpoint.name},
    )


def _add_output_to_input_edges(endpoints: list[Endpoint], add_edge) -> None:
    for source in endpoints:
        if not source.response_fields:
            continue
        field_index = _build_field_identifier_index(source.response_fields)
        for target in endpoints:
            if source.endpoint_id == target.endpoint_id:
                continue
            for parameter in target.parameters:
                if not parameter.required:
                    continue
                match = _find_field_match(field_index, parameter)
                if match is None:
                    continue
                field, match_type = match
                add_edge(
                    GraphEdge(
                        source=_endpoint_node_id(source.endpoint_id),
                        target=_endpoint_node_id(target.endpoint_id),
                        type="output_satisfies_input",
                        metadata={
                            "source_endpoint": source.endpoint_id,
                            "target_endpoint": target.endpoint_id,
                            "parameter": parameter.name,
                            "source_field": field.name,
                            "match_type": match_type,
                        },
                    )
                )


def _add_same_domain_edges(endpoints: list[Endpoint], add_edge) -> None:
    by_domain: dict[str, list[Endpoint]] = defaultdict(list)
    for endpoint in endpoints:
        by_domain[endpoint.domain].append(endpoint)

    for domain_endpoints in by_domain.values():
        for source in domain_endpoints:
            for target in domain_endpoints:
                if source.endpoint_id == target.endpoint_id:
                    continue
                add_edge(
                    GraphEdge(
                        source=_endpoint_node_id(source.endpoint_id),
                        target=_endpoint_node_id(target.endpoint_id),
                        type="same_domain",
                        metadata={"domain": source.domain},
                    )
                )


def _add_semantic_edges(
    endpoints: list[Endpoint],
    add_edge,
    *,
    semantic_retriever: SemanticRetriever,
    threshold: float,
    top_k: int,
    existing_pairs: set[tuple[str, str]],
) -> None:
    cards = [endpoint_card(endpoint) for endpoint in endpoints]
    semantic_retriever.index(cards)
    endpoint_ids = {endpoint.endpoint_id for endpoint in endpoints}

    for card in cards:
        matches = [
            match
            for match in semantic_retriever.search(card, top_k=top_k)
            if match.endpoint_id in endpoint_ids
            and match.endpoint_id != card.endpoint_id
            and match.score >= threshold
            and (_endpoint_node_id(card.endpoint_id), _endpoint_node_id(match.endpoint_id))
            not in existing_pairs
        ]
        matches = sorted(matches, key=lambda match: (-match.score, match.endpoint_id))
        for match in matches:
            add_edge(
                GraphEdge(
                    source=_endpoint_node_id(card.endpoint_id),
                    target=_endpoint_node_id(match.endpoint_id),
                    type="semantic_related",
                    metadata={
                        "score": match.score,
                        "reason": match.reason,
                        "threshold": threshold,
                        "top_k": top_k,
                    },
                )
            )


def _build_field_identifier_index(
    fields: list[ResponseField],
) -> dict[str, ResponseField]:
    index: dict[str, ResponseField] = {}
    for field in fields:
        for identifier in _field_identifiers(field):
            index.setdefault(identifier, field)
    return index


def _find_field_match(
    field_index: dict[str, ResponseField],
    parameter: Parameter,
) -> tuple[ResponseField, str] | None:
    if parameter.name in field_index:
        field = field_index[parameter.name]
        match_type = "exact_name" if field.name == parameter.name else "canonical"
        return field, match_type
    if parameter.canonical_name and parameter.canonical_name in field_index:
        return field_index[parameter.canonical_name], "canonical"
    for alias in parameter.aliases:
        if alias in field_index:
            return field_index[alias], "alias"
    return None


def _field_identifiers(field: Parameter | ResponseField) -> set[str]:
    identifiers = {field.name}
    if field.canonical_name:
        identifiers.add(field.canonical_name)
    identifiers.update(field.aliases)
    return identifiers


def _endpoint_node_id(endpoint_id: str) -> str:
    return f"endpoint:{endpoint_id}"
