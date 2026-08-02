"""Stage 5: Pinecone vector indexing.

This module is the only place that talks to Pinecone. It covers the five
capabilities the system needs:

1. **Index creation** – serverless index created on demand with the embedding
   model's dimension and the ``cosine`` metric.
2. **Namespace usage** – every corpus lives in its own namespace, which keeps
   sessions/users isolated inside a single (free-tier) index.
3. **Upserting vectors** – batched writes with retry.
4. **Querying vectors** – top-k ANN search with optional metadata filters.
5. **Metadata management** – document name, page number and chunk ID travel with
   each vector so answers stay traceable.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .config import Settings, get_settings
from .models import Chunk

logger = logging.getLogger(__name__)

try:  # the exception module moved between SDK majors
    from pinecone.exceptions import PineconeException  # type: ignore
except Exception:  # noqa: BLE001 - older SDKs expose no exceptions module

    class PineconeException(Exception):  # type: ignore[no-redef]
        """Fallback when the SDK does not expose its exception type."""


# Pinecone recommends <= 1000 vectors and <= 2 MB per upsert request.
UPSERT_BATCH_SIZE = 96
INDEX_READY_TIMEOUT_SECONDS = 90


class VectorStoreError(RuntimeError):
    """Raised for any Pinecone connection, configuration or query failure."""


def _match_to_dict(match: Any) -> dict[str, Any]:
    """Normalise one Pinecone match into a plain dict.

    Across SDK majors a match may be a dict, an OpenAPI model with ``to_dict``,
    or a simple object with attributes — accept all three.
    """
    if isinstance(match, dict):
        return match
    to_dict = getattr(match, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001 - fall through to attribute access
            pass
    return {
        "id": getattr(match, "id", ""),
        "score": getattr(match, "score", 0.0),
        "metadata": getattr(match, "metadata", {}) or {},
    }


def sanitize_namespace(raw: str) -> str:
    """Coerce arbitrary text into a namespace Pinecone will accept."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw or "").strip()).strip("-")
    return (cleaned or "default")[:48].lower()


class PineconeVectorStore:
    """Thin, well-behaved wrapper around a single Pinecone serverless index."""

    def __init__(self, settings: Settings | None = None, dimension: int | None = None) -> None:
        self.settings = settings or get_settings()
        self.dimension = dimension or self.settings.embedding_dimension
        self.index_name = self.settings.pinecone_index_name

        if not self.settings.pinecone_api_key:
            raise VectorStoreError(
                "PINECONE_API_KEY is missing. Copy .env.example to .env and add your key."
            )

        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("The `pinecone` package is not installed.") from exc

        self._ServerlessSpec = ServerlessSpec
        try:
            self._client = Pinecone(api_key=self.settings.pinecone_api_key)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not initialise the Pinecone client: {exc}") from exc

        self._index = self._ensure_index()

    # --- 1. Index creation ------------------------------------------------

    def _existing_index_names(self) -> list[str]:
        try:
            return list(self._client.list_indexes().names())
        except AttributeError:  # older/newer SDKs return a plain list of dicts
            return [item["name"] for item in self._client.list_indexes()]
        except PineconeException as exc:
            raise VectorStoreError(
                f"Could not reach Pinecone. Check your API key and network connection. ({exc})"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Unexpected Pinecone error while listing indexes: {exc}") from exc

    def _ensure_index(self):
        names = self._existing_index_names()

        if self.index_name not in names:
            logger.info("Creating Pinecone index %r (dim=%d, metric=cosine)", self.index_name, self.dimension)
            try:
                self._client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.settings.pinecone_metric,
                    spec=self._ServerlessSpec(
                        cloud=self.settings.pinecone_cloud,
                        region=self.settings.pinecone_region,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(
                    f"Failed to create Pinecone index '{self.index_name}': {exc}"
                ) from exc
            self._wait_until_ready()
        else:
            self._assert_dimension_matches()

        try:
            return self._client.Index(self.index_name)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not open Pinecone index '{self.index_name}': {exc}") from exc

    def _wait_until_ready(self) -> None:
        deadline = time.time() + INDEX_READY_TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                status = self._client.describe_index(self.index_name).status
                if status.get("ready"):
                    return
            except Exception:  # noqa: BLE001 - the index may not be visible yet
                pass
            time.sleep(2)
        raise VectorStoreError(
            f"Pinecone index '{self.index_name}' was not ready within "
            f"{INDEX_READY_TIMEOUT_SECONDS}s. Check the Pinecone console."
        )

    def _assert_dimension_matches(self) -> None:
        """A dimension mismatch produces confusing upsert errors — fail early instead."""
        try:
            described = self._client.describe_index(self.index_name)
            existing_dim = int(described.dimension)
        except Exception:  # noqa: BLE001 - non-fatal; the upsert will report it
            return
        if existing_dim != self.dimension:
            raise VectorStoreError(
                f"Index '{self.index_name}' has dimension {existing_dim} but the configured "
                f"embedding model produces {self.dimension}-d vectors. Either set "
                f"PINECONE_INDEX_NAME to a new index or delete the existing one."
            )

    # --- 3. Upserting -----------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        namespace: str,
        progress_callback=None,
    ) -> int:
        """Write ``chunks`` and their ``vectors`` into ``namespace``.

        Returns the number of vectors successfully upserted.
        """
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"Chunk/vector count mismatch: {len(chunks)} chunks vs {len(vectors)} vectors."
            )
        if not chunks:
            return 0

        namespace = sanitize_namespace(namespace)
        payload = [
            {
                "id": chunk.chunk_id,
                "values": vector.tolist(),
                "metadata": chunk.to_metadata(),
            }
            for chunk, vector in zip(chunks, vectors)
        ]

        written = 0
        for start in range(0, len(payload), UPSERT_BATCH_SIZE):
            batch = payload[start : start + UPSERT_BATCH_SIZE]
            self._upsert_batch_with_retry(batch, namespace)
            written += len(batch)
            if progress_callback:
                progress_callback(written, len(payload))
        return written

    def _upsert_batch_with_retry(self, batch: list[dict[str, Any]], namespace: str, attempts: int = 3) -> None:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self._index.upsert(vectors=batch, namespace=namespace)
                return
            except Exception as exc:  # noqa: BLE001 - transient network/rate-limit errors
                last_error = exc
                logger.warning("Upsert attempt %d/%d failed: %s", attempt, attempts, exc)
                time.sleep(1.5 * attempt)
        raise VectorStoreError(f"Failed to upsert vectors into Pinecone after {attempts} attempts: {last_error}")

    # --- 4. Querying ------------------------------------------------------

    def query(
        self,
        vector: np.ndarray,
        namespace: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a cosine similarity search and return raw Pinecone matches."""
        if top_k < 1:
            raise VectorStoreError("top_k must be at least 1.")

        try:
            response = self._index.query(
                vector=vector.tolist(),
                top_k=top_k,
                namespace=sanitize_namespace(namespace),
                include_metadata=True,
                include_values=False,
                filter=metadata_filter or None,
            )
        except PineconeException as exc:
            raise VectorStoreError(f"Pinecone query failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Unexpected error during Pinecone query: {exc}") from exc

        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
        return [_match_to_dict(match) for match in matches]

    # --- 5. Metadata / housekeeping --------------------------------------

    def describe_namespace(self, namespace: str) -> dict[str, Any]:
        """Vector count and other stats for one namespace."""
        try:
            stats = self._index.describe_index_stats()
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not read index statistics: {exc}") from exc

        raw = stats.get("namespaces", {}) if isinstance(stats, dict) else getattr(stats, "namespaces", {}) or {}
        entry = raw.get(sanitize_namespace(namespace), {}) or {}
        count = entry.get("vector_count", 0) if isinstance(entry, dict) else getattr(entry, "vector_count", 0)
        return {"namespace": sanitize_namespace(namespace), "vector_count": int(count)}

    def list_documents(self, namespace: str, sample_size: int = 1000) -> list[str]:
        """Names of the documents currently indexed in ``namespace``.

        Implemented as a wide search against a zero-ish probe vector: the free
        tier does not expose a metadata-only scan, and this is accurate enough
        for populating the UI's document filter.
        """
        probe = np.ones(self.dimension, dtype=np.float32)
        probe /= np.linalg.norm(probe)
        try:
            matches = self.query(probe, namespace=namespace, top_k=min(sample_size, 1000))
        except VectorStoreError:
            return []
        names = {str((match.get("metadata") or {}).get("document_name", "")) for match in matches}
        return sorted(name for name in names if name)

    def delete_namespace(self, namespace: str) -> None:
        """Drop every vector in ``namespace`` (used by the UI's *Clear corpus*)."""
        try:
            self._index.delete(delete_all=True, namespace=sanitize_namespace(namespace))
        except Exception as exc:  # noqa: BLE001 - deleting an empty namespace 404s on some tiers
            logger.info("delete_namespace(%s) reported: %s", namespace, exc)


def build_metadata_filter(
    document_names: Iterable[str] | None = None,
    page_range: tuple[int, int] | None = None,
    max_chunk_index: int | None = None,
) -> dict[str, Any] | None:
    """Compose a Pinecone metadata filter from the UI's selections.

    ``max_chunk_index`` restricts the search to the opening chunks of each
    document, which is how the overview fallback in :mod:`src.retriever` finds
    representative context for "what is this about?" style questions.
    """
    conditions: dict[str, Any] = {}

    names = [name for name in (document_names or []) if name]
    if names:
        conditions["document_name"] = {"$in": names}

    if page_range:
        low, high = page_range
        if low > high:
            low, high = high, low
        conditions["page_number"] = {"$gte": int(low), "$lte": int(high)}

    if max_chunk_index is not None:
        conditions["chunk_index"] = {"$lt": int(max_chunk_index)}

    return conditions or None
