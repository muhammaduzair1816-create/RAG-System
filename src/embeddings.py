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

    def embed_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed chunk texts for indexing. Returns shape ``(len(texts), dimension)``.

        Batches are encoded and reduced one at a time. Holding only one batch of
        activations at a time is what keeps peak memory flat on a 512 MB host,
        so the batch size deliberately defaults to a small value.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if batch_size is None:
            batch_size = get_settings().embedding_batch_size
        batch_size = max(1, batch_size)

        encoded = [
            self._encode(texts[i : i + batch_size], "passage")
            for i in range(0, len(texts), batch_size)
        ]
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


class OnnxEmbedder(BaseEmbedder):
    """Local embedding backend running the same model through ONNX Runtime.

    Produces vectors identical to :class:`SentenceTransformerEmbedder` — same
    checkpoint, same mean pooling, same L2 normalisation — but loads neither
    torch nor transformers, which together account for roughly 350 MB of
    resident memory. Existing Pinecone indexes stay valid.
    """

    # Mean pooling over the token dimension, matching the model's 1_Pooling config.
    _MAX_TOKENS = 256

    def __init__(self, model_name: str, onnx_threads: int = 1) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise EmbeddingError(
                "The ONNX embedding backend needs `onnxruntime`, `tokenizers` and "
                f"`huggingface_hub` ({exc}). Run `pip install -r requirements.txt`, or set "
                "EMBEDDING_BACKEND=sentence-transformers to use the torch backend instead."
            ) from exc

        repo = model_name
        try:
            model_path = hf_hub_download(repo, "onnx/model.onnx")
            tokenizer_path = hf_hub_download(repo, "tokenizer.json")
        except Exception as exc:  # noqa: BLE001 - network, auth, missing ONNX export
            raise EmbeddingError(
                f"Could not fetch an ONNX export of '{repo}': {exc}\n"
                "Not every checkpoint ships one. Either choose a model that does, or set "
                "EMBEDDING_BACKEND=sentence-transformers."
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, onnx_threads)
        options.inter_op_num_threads = 1
        # The CPU arena caches every allocation it ever makes and never returns
        # it. On a 512 MB host that alone pushed peak RSS from ~200 MB to ~880 MB
        # while indexing, so it stays off.
        options.enable_cpu_mem_arena = False

        self._session = ort.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {item.name for item in self._session.get_inputs()}

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=self._MAX_TOKENS)

        self.model_name = model_name
        self.dimension = int(self._session.get_outputs()[0].shape[-1])

    def _encode(self, texts: list[str], input_type: str) -> np.ndarray:
        del input_type  # symmetric model: queries and passages are encoded identically

        encodings = self._tokenizer.encode_batch(texts)
        ids = np.array([item.ids for item in encodings], dtype=np.int64)
        mask = np.array([item.attention_mask for item in encodings], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        token_embeddings = self._session.run(None, feed)[0]

        # Mean-pool over real tokens only; padding must not dilute the average.
        weights = mask[..., None].astype(np.float32)
        summed = (token_embeddings * weights).sum(axis=1)
        counts = np.clip(weights.sum(axis=1), 1e-9, None)
        return summed / counts


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
def _build_embedder(backend: str, model_name: str, api_key: str, onnx_threads: int) -> BaseEmbedder:
    """Construct an embedder, cached so a model is loaded at most once."""
    if backend == "pinecone":
        return PineconeInferenceEmbedder(model_name, api_key)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    return OnnxEmbedder(model_name, onnx_threads)


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Return the configured embedder, cached so the model loads only once.

    The model is loaded on the first call, not at import or startup — callers
    should invoke this only when an embedding is actually needed.
    """
    settings = settings or get_settings()
    model_name = settings.embedding_model
    if settings.embedding_backend == "pinecone" and model_name.startswith("sentence-transformers/"):
        model_name = "multilingual-e5-large"  # sensible default when only the backend was switched
    return _build_embedder(
        settings.embedding_backend,
        model_name,
        settings.pinecone_api_key,
        settings.onnx_threads,
    )


def reset_embedder_cache() -> None:
    """Drop the cached embedder (used by tests and after a config change)."""
    _build_embedder.cache_clear()
