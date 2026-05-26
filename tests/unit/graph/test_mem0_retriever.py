from kg_mle.graph.semantic import EndpointCard, Mem0SemanticRetriever


class FakeMem0Memory:
    """Minimal fake Mem0 surface used by the retriever.

    Captures add() calls so tests can assert on `infer=False` and filter shape,
    and serves canned search() results so retrieval parsing can be exercised
    without provider credentials.
    """

    def __init__(self) -> None:
        self.added: list[dict] = []
        self.search_calls: list[dict] = []
        self._search_returns: dict[str, list[dict]] = {}

    def queue_search(self, query_text: str, results: list[dict]) -> None:
        self._search_returns[query_text] = results

    def add(self, text, *, user_id, metadata, infer):
        self.added.append(
            {"text": text, "user_id": user_id, "metadata": dict(metadata), "infer": infer}
        )

    def search(self, query, *, filters, top_k, rerank):
        self.search_calls.append(
            {"query": query, "filters": dict(filters), "top_k": top_k, "rerank": rerank}
        )
        return {"results": self._search_returns.get(query, [])}


def _build_retriever(memory: FakeMem0Memory) -> Mem0SemanticRetriever:
    return Mem0SemanticRetriever(
        embedding_provider="huggingface",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_provider="gemini",
        llm_model="gemini-2.0-flash-lite-001",
        memory=memory,
    )


def test_mem0_retriever_indexes_cards_with_infer_false_and_endpoint_metadata():
    memory = FakeMem0Memory()
    retriever = _build_retriever(memory)

    retriever.index(
        [
            EndpointCard(endpoint_id="travel/search_hotels", text="card A", metadata={"domain": "travel"}),
            EndpointCard(endpoint_id="weather/get_forecast", text="card B", metadata={"domain": "weather"}),
        ]
    )

    assert len(memory.added) == 2
    for record in memory.added:
        assert record["infer"] is False
        assert record["user_id"] == "kg_mle_tool_graph"
        assert "endpoint_id" in record["metadata"]
    assert memory.added[0]["metadata"]["endpoint_id"] == "travel/search_hotels"
    assert memory.added[0]["metadata"]["domain"] == "travel"


def test_mem0_retriever_parses_search_results_via_metadata_endpoint_id():
    memory = FakeMem0Memory()
    memory.queue_search(
        "query card",
        [
            {"metadata": {"endpoint_id": "events/book_tickets"}, "score": 0.92},
            {"metadata": {"endpoint_id": "travel/book_itinerary"}, "score": 0.81},
            {"not a dict": True},
        ],
    )
    retriever = _build_retriever(memory)

    matches = retriever.search(EndpointCard(endpoint_id="src", text="query card"), top_k=3)

    assert [m.endpoint_id for m in matches] == ["events/book_tickets", "travel/book_itinerary"]
    assert matches[0].score == 0.92
    assert matches[0].reason == "mem0"
    assert memory.search_calls[0]["filters"] == {"user_id": "kg_mle_tool_graph"}
    assert memory.search_calls[0]["rerank"] is False


def test_mem0_retriever_falls_back_to_indexed_text_when_metadata_endpoint_id_missing():
    memory = FakeMem0Memory()
    retriever = _build_retriever(memory)
    retriever.index([EndpointCard(endpoint_id="gaming/get_tournament_schedule", text="card body")])

    memory.queue_search(
        "query",
        [{"metadata": {}, "memory": "card body", "score": 0.77}],
    )

    matches = retriever.search(EndpointCard(endpoint_id="src", text="query"), top_k=5)

    assert [m.endpoint_id for m in matches] == ["gaming/get_tournament_schedule"]
    assert matches[0].score == 0.77


def test_mem0_retriever_returns_empty_when_no_endpoint_id_can_be_recovered():
    memory = FakeMem0Memory()
    memory.queue_search("q", [{"metadata": {}, "memory": "unknown text", "score": 0.9}])
    retriever = _build_retriever(memory)

    matches = retriever.search(EndpointCard(endpoint_id="src", text="q"), top_k=5)

    assert matches == []


def test_mem0_retriever_caps_results_at_top_k():
    memory = FakeMem0Memory()
    memory.queue_search(
        "q",
        [
            {"metadata": {"endpoint_id": f"d/e_{i}"}, "score": 0.9 - i * 0.01}
            for i in range(10)
        ],
    )
    retriever = _build_retriever(memory)

    matches = retriever.search(EndpointCard(endpoint_id="src", text="q"), top_k=3)

    assert len(matches) == 3
    assert memory.search_calls[0]["top_k"] == 4
