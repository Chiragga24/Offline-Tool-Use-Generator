import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sentence_transformers")

from kg_mle.graph.semantic import EndpointCard, SentenceTransformerSemanticRetriever


class TinyDeterministicModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(
        self,
        texts,
        *,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = []
        for text in texts:
            if "hotel" in text or "travel" in text:
                vector = np.array([1.0, 0.0, 0.0])
            elif "weather" in text:
                vector = np.array([0.8, 0.2, 0.0])
            else:
                vector = np.array([0.0, 1.0, 0.0])
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors)


def test_sentence_transformer_retriever_ranks_by_cosine_similarity(monkeypatch):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", TinyDeterministicModel)

    retriever = SentenceTransformerSemanticRetriever(model_name="tiny-test-model")
    cards = [
        EndpointCard("travel/search_hotels", "hotel travel city"),
        EndpointCard("weather/get_forecast", "weather city date"),
        EndpointCard("finance/get_quote", "stock market quote"),
    ]

    retriever.index(cards)
    matches = retriever.search(cards[0], top_k=3)

    assert [match.endpoint_id for match in matches] == [
        "travel/search_hotels",
        "weather/get_forecast",
        "finance/get_quote",
    ]
    assert matches[0].score >= matches[1].score >= matches[2].score
