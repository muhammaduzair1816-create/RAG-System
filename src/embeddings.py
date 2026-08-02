"""Stage 4: embedding generation.

Two interchangeable backends sit behind one interface:

``sentence-transformers``
    Runs ``all-MiniLM-L6-v2`` locally (384-d, ~80 MB). Free, offline after the
    first download, and fast enough on CPU for assignment-scale corpora.

``pinecone``
    Calls Pinecone's hosted inference API (``multilingual-e5-large``, 1024-d).
    Useful when local wheels for ``torch`` are unavailable or the machine is
    memory-constrained.

Both return **L2-normalised** vectors, so a cosine index and a dot-product
comparison agree, and scores land in an interpretable [0, 1] range.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from .config import Settings, get_settings

# Hosted models require a hint about whether the text is a document or a query.
_QUERY_PREFIXES = {"query", "passage"}


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector stays zero rather than becoming NaN
    return vectors / norms


class BaseEmbedder(ABC):
    """Common interface for every embedding backend."""

    model_name: str
    dimension: int

    @abstractmethod
    def _encode(self, texts: list[str], input_type: str) -> np.ndarray: ...

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed chunk texts for indexing. Returns shape ``(len(texts), dimension)``."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        encoded = [self._encode(batch, "passage") for batch in batches]
        return _normalise(np.vstack(encoded).astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user query. Returns shape ``(dimension,)``."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        vector = self._encode([text.strip()], "query")
        return _normalise(vector.astype(np.float32))[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embedding backend built on ``sentence-transformers``."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise EmbeddingError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt`, "
                "or set EMBEDDING_BACKEND=pinecone to use hosted embeddings instead."
            ) from exc

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001 - network / disk / model-name failures
            raise EmbeddingError(f"Could not load embedding model '{model_name}': {exc}") from exc

        self.model_name = model_name
        # Renamed in sentence-transformers 5.x; the old name still works but warns.
        get_dimension = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dimension())

    def _encode(self, texts: list[str], input_type: str) -> np.ndarray:
        del input_type  # symmetric model: queries and passages are encoded identically
        return np.asarray(self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True))


class PineconeInferenceEmbedder(BaseEmbedder):
    """Hosted embedding backend using Pinecone Inference."""

    _DIMENSIONS = {"multilingual-e5-large": 1024, "llama-text-embed-v2": 1024}

    def __init__(self, model_name: str, api_key: str) -> None:
        try:
            from pinecone import Pinecone
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError("The `pinecone` package is not installed.") from exc

        if not api_key:
            raise EmbeddingError("PINECONE_API_KEY is required for the hosted embedding backend.")

        self._client = Pinecone(api_key=api_key)
        self.model_name = model_name
        self.dimension = self._DIMENSIONS.get(model_name, 1024)

    def _encode(self, texts: list[str], input_type: str) -> np.ndarray:
        if input_type not in _QUERY_PREFIXES:
            input_type = "passage"
        try:
            response = self._client.inference.embed(
                model=self.model_name,
                inputs=texts,
                parameters={"input_type": input_type, "truncate": "END"},
            )
        except Exception as exc:  # noqa: BLE001 - network / quota failures
            raise EmbeddingError(f"Pinecone hosted embedding request failed: {exc}") from exc
        return np.asarray([item["values"] for item in response.data])


@lru_cache(maxsize=4)
def _build_embedder(backend: str, model_name: str, api_key: str) -> BaseEmbedder:
    if backend == "pinecone":
        return PineconeInferenceEmbedder(model_name, api_key)
    return SentenceTransformerEmbedder(model_name)


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Return the configured embedder, cached so the model loads only once."""
    settings = settings or get_settings()
    model_name = settings.embedding_model
    if settings.embedding_backend == "pinecone" and model_name.startswith("sentence-transformers/"):
        model_name = "multilingual-e5-large"  # sensible default when only the backend was switched
    return _build_embedder(settings.embedding_backend, model_name, settings.pinecone_api_key)
