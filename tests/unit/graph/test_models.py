import json

import pytest
from pydantic import ValidationError

from kg_mle.graph import GraphEdge, GraphNode, ToolGraph


def test_graph_models_round_trip_json():
    graph = ToolGraph(
        nodes=[
            GraphNode(
                node_id="endpoint:travel/search_hotels",
                type="endpoint",
                label="search_hotels",
                metadata={"domain": "travel"},
            ),
            GraphNode(
                node_id="parameter:travel/book_itinerary.hotel_id",
                type="parameter",
                label="hotel_id",
                metadata={"required": True},
            ),
        ],
        edges=[
            GraphEdge(
                source="endpoint:travel/search_hotels",
                target="parameter:travel/book_itinerary.hotel_id",
                type="output_satisfies_input",
                metadata={"field": "hotel_id"},
            ),
            GraphEdge(
                source="endpoint:travel/search_hotels",
                target="endpoint:weather/get_forecast",
                type="semantic_related",
                metadata={"score": 0.82, "backend": "mem0"},
            )
        ],
    )

    payload = json.loads(graph.model_dump_json())
    restored = ToolGraph.model_validate(payload)

    assert restored.node_count() == 2
    assert restored.node_count("endpoint") == 1
    assert restored.edge_count("output_satisfies_input") == 1
    assert restored.edge_count("semantic_related") == 1
    assert restored.get_node("endpoint:travel/search_hotels").label == "search_hotels"
    assert restored.has_edge(
        "endpoint:travel/search_hotels",
        "parameter:travel/book_itinerary.hotel_id",
        "output_satisfies_input",
    )


def test_graph_node_rejects_unknown_type():
    with pytest.raises(ValidationError):
        GraphNode(node_id="x", type="unknown", label="x")


def test_graph_edge_rejects_unknown_type():
    with pytest.raises(ValidationError):
        GraphEdge(source="a", target="b", type="unknown")
