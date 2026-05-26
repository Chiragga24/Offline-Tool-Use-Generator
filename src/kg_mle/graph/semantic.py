from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class EndpointCard:
    endpoint_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticMatch:
    endpoint_id: str
    score: float
    reason: str = ""


class SemanticRetriever(Protocol):
    def index(self, cards: list[EndpointCard]) -> None:
        """Index endpoint cards for later semantic search."""

    def search(self, card: EndpointCard, *, top_k: int) -> list[SemanticMatch]:
        """Return semantically related endpoint IDs for one card."""


class FakeSemanticRetriever:
    def __init__(self, matches: dict[str, list[SemanticMatch]] | None = None) -> None:
        self.matches = matches or {}
        self.indexed_cards: list[EndpointCard] = []

    def index(self, cards: list[EndpointCard]) -> None:
        self.indexed_cards = list(cards)

    def search(self, card: EndpointCard, *, top_k: int) -> list[SemanticMatch]:
        return self.matches.get(card.endpoint_id, [])[:top_k]


class Mem0SemanticRetriever:
    """Optional Mem0-backed retriever.

    This class is intentionally isolated so the default offline pipeline does not
    import or require Mem0. Tests should use FakeSemanticRetriever.
    """

    def __init__(
        self,
        *,
        embedding_provider: str,
        embedding_model: str,
        llm_provider: str,
        llm_model: str,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
    ) -> None:
        try:
            from mem0 import Memory  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Mem0 semantic graph expansion requires optional dependencies. "
                'Install with: uv pip install -e ".[semantic]"'
            ) from exc

        try:
            llm_config = {
                "model": llm_model,
                "temperature": 0.0,
                "max_tokens": 1000,
                "top_p": 1.0,
            }
            if llm_api_key:
                llm_config["api_key"] = llm_api_key
            if llm_base_url:
                llm_config["base_url"] = llm_base_url
                llm_config["openai_base_url"] = llm_base_url

            self._memory = Memory.from_config(
                {
                    "embedder": {
                        "provider": embedding_provider,
                        "config": {"model": embedding_model},
                    },
                    "llm": {
                        "provider": llm_provider,
                        "config": llm_config,
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": "kg_mle_tool_graph",
                            "embedding_model_dims": 384,
                            "path": ":memory:",
                        },
                    },
                }
            )
        except Exception as exc:
            raise RuntimeError(
                "Mem0 semantic graph expansion initialized embeddings but could not "
                "initialize its required LLM/vector-store configuration. Configure "
                "Mem0 fully, for example with GOOGLE_API_KEY for Gemini, "
                "or use --semantic-backend local for offline MiniLM retrieval."
            ) from exc
        self._user_id = "kg_mle_tool_graph"
        self._endpoint_ids_by_text: dict[str, str] = {}

    def index(self, cards: list[EndpointCard]) -> None:
        for card in cards:
            metadata = {"endpoint_id": card.endpoint_id, **card.metadata}
            self._endpoint_ids_by_text[card.text] = card.endpoint_id
            self._memory.add(
                card.text,
                user_id=self._user_id,
                metadata=metadata,
                infer=False,
            )

    def search(self, card: EndpointCard, *, top_k: int) -> list[SemanticMatch]:
        raw_results = self._memory.search(
            card.text,
            filters={"user_id": self._user_id},
            top_k=top_k + 1,
            rerank=False,
        )
        results = raw_results.get("results", raw_results) if isinstance(raw_results, dict) else raw_results
        matches: list[SemanticMatch] = []
        for result in results or []:
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata") or {}
            endpoint_id = metadata.get("endpoint_id")
            if not endpoint_id:
                memory_text = result.get("memory") or result.get("text") or ""
                endpoint_id = self._endpoint_ids_by_text.get(memory_text)
            if not endpoint_id:
                continue
            score = float(result.get("score") or result.get("similarity") or 0.0)
            matches.append(SemanticMatch(endpoint_id=str(endpoint_id), score=score, reason="mem0"))
        return matches[:top_k]


class SentenceTransformerSemanticRetriever:
    """Local CPU semantic retriever using sentence-transformers.

    This provides deterministic-ish cosine-similarity retrieval for offline
    graph expansion. It avoids Mem0's memory-extraction LLM path while keeping
    the same SemanticRetriever interface.
    """

    def __init__(self, *, model_name: str) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local semantic graph expansion requires sentence-transformers. "
                'Install with: uv pip install -e ".[semantic]"'
            ) from exc

        self.model_name = model_name
        self._np = np
        self._model = SentenceTransformer(model_name)
        self._cards: list[EndpointCard] = []
        self._embeddings = None

    def index(self, cards: list[EndpointCard]) -> None:
        self._cards = list(cards)
        embeddings = self._model.encode(
            [card.text for card in self._cards],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._embeddings = self._np.asarray(embeddings)

    def search(self, card: EndpointCard, *, top_k: int) -> list[SemanticMatch]:
        if self._embeddings is None:
            raise RuntimeError("Semantic retriever must be indexed before search.")
        query = self._model.encode(
            [card.text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self._embeddings @ query
        ranked_indices = sorted(
            range(len(self._cards)),
            key=lambda idx: (-float(scores[idx]), self._cards[idx].endpoint_id),
        )
        matches: list[SemanticMatch] = []
        for idx in ranked_indices:
            endpoint_id = self._cards[idx].endpoint_id
            matches.append(
                SemanticMatch(
                    endpoint_id=endpoint_id,
                    score=float(scores[idx]),
                    reason=f"sentence-transformers:{self.model_name}",
                )
            )
            if len(matches) >= top_k:
                break
        return matches
