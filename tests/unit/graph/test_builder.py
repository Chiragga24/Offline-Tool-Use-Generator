import json

from kg_mle.config import DEFAULT_INPUT_PATH
from kg_mle.graph import FakeSemanticRetriever, SemanticMatch, build_tool_graph, save_tool_graph
from kg_mle.registry import load_registry


def test_build_tool_graph_from_registry_creates_core_nodes_and_edges():
    registry = load_registry(DEFAULT_INPUT_PATH)
    graph = build_tool_graph(registry)

    assert graph.node_count("domain") == 9
    assert graph.node_count("tool") == 9
    assert graph.node_count("endpoint") == 45
    assert graph.edge_count("contains_tool") == 9
    assert graph.edge_count("exposes_endpoint") == 45
    assert graph.edge_count("requires_parameter") > 0
    assert graph.edge_count("returns_field") > 0


def test_build_tool_graph_creates_grounding_edges_from_output_to_required_input():
    registry = load_registry(DEFAULT_INPUT_PATH)
    graph = build_tool_graph(registry)

    assert graph.has_edge(
        "endpoint:travel/search_hotels",
        "endpoint:travel/get_hotel_details",
        "output_satisfies_input",
    )
    assert graph.has_edge(
        "endpoint:travel/search_hotels",
        "endpoint:travel/book_itinerary",
        "output_satisfies_input",
    )
    assert graph.has_edge(
        "endpoint:events/check_ticket_availability",
        "endpoint:events/book_tickets",
        "output_satisfies_input",
    )
    assert graph.has_edge(
        "endpoint:ai_ml/create_eval_job",
        "endpoint:ai_ml/get_eval_result",
        "output_satisfies_input",
    )


def test_build_tool_graph_adds_deterministic_semantic_edges_with_fake_retriever():
    registry = load_registry(DEFAULT_INPUT_PATH)
    retriever = FakeSemanticRetriever(
        {
            "travel/search_flights": [
                SemanticMatch("weather/get_forecast", 0.91, "travel weather planning"),
                SemanticMatch("travel/search_flights", 0.99, "self match"),
                SemanticMatch("finance/get_quote", 0.40, "below threshold"),
            ]
        }
    )

    graph = build_tool_graph(
        registry,
        semantic_retriever=retriever,
        semantic_threshold=0.72,
        semantic_top_k=5,
    )

    assert len(retriever.indexed_cards) == 45
    assert graph.has_edge(
        "endpoint:travel/search_flights",
        "endpoint:weather/get_forecast",
        "semantic_related",
    )
    assert not graph.has_edge(
        "endpoint:travel/search_flights",
        "endpoint:travel/search_flights",
        "semantic_related",
    )
    assert not graph.has_edge(
        "endpoint:travel/search_flights",
        "endpoint:finance/get_quote",
        "semantic_related",
    )


def test_build_tool_graph_does_not_duplicate_deterministic_edges_as_semantic_edges():
    registry = load_registry(DEFAULT_INPUT_PATH)
    retriever = FakeSemanticRetriever(
        {
            "travel/search_hotels": [
                SemanticMatch("travel/get_hotel_details", 0.99, "duplicate deterministic edge"),
            ]
        }
    )

    graph = build_tool_graph(
        registry,
        semantic_retriever=retriever,
        semantic_threshold=0.80,
        semantic_top_k=5,
    )

    assert graph.has_edge(
        "endpoint:travel/search_hotels",
        "endpoint:travel/get_hotel_details",
        "output_satisfies_input",
    )
    assert not graph.has_edge(
        "endpoint:travel/search_hotels",
        "endpoint:travel/get_hotel_details",
        "semantic_related",
    )


def test_save_tool_graph_round_trips_json(tmp_path):
    registry = load_registry(DEFAULT_INPUT_PATH)
    graph = build_tool_graph(registry)
    path = save_tool_graph(graph, tmp_path / "tool_graph.json")

    saved = json.loads(path.read_text(encoding="utf-8"))

    assert len(saved["nodes"]) == graph.node_count()
    assert len(saved["edges"]) == graph.edge_count()
