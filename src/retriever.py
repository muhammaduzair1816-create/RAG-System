"""Stage 6: semantic retrieval.

Wraps the vector store with the query-time policy the assignment calls for:
top-k selection, an adjustable cosine-similarity threshold, optional metadata
filtering, and de-duplication of near-identical chunks caused by overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from .embeddings import BaseEmbedder
from .models import RetrievedChunk
from .vector_store import PineconeVectorStore, build_metadata_filter

logger = logging.getLogger(__name__)

# How many opening chunks per document count as "the overview" of it.
OVERVIEW_WINDOW = 4

# Questions about the document as an object score near zero against its own
# text, because a symmetric embedding model compares meaning and a meta-question
# shares none with the content. They need a different retrieval mode, not a
# different threshold.
_STRONG_OVERVIEW_PHRASES = (
    "summarise", "summarize", "summary", "overview", "tldr", "tl;dr",
    "main topic", "main subject", "main point", "key points", "what is it about",
    "gist", "in a nutshell",
)
# "about" is too common to trigger on alone; it needs a document-ish noun.
_WEAK_OVERVIEW_PHRASES = ("about", "cover", "discuss", "contain")
_DOCUMENT_NOUNS = ("document", "report", "paper", "pdf", "file", "text", "article", "this")


def looks_like_overview_request(query: str) -> bool:
    """True for questions asking what the corpus *is*, not what it says."""
    normalised = " ".join(query.lower().split())
    if not normalised:
        return False
    # A long question is asking something specific, even if it contains "about".
    if len(normalised.split()) > 12:
        return False
    if any(phrase in normalised for phrase in _STRONG_OVERVIEW_PHRASES):
        return True
    return any(phrase in normalised for phrase in _WEAK_OVERVIEW_PHRASES) and any(
        noun in normalised for noun in _DOCUMENT_NOUNS
    )


class RetrievalError(RuntimeError):
    """Raised when a query cannot be served."""


@dataclass
class RetrievalResult:
    """Everything the generator and the UI need to know about one retrieval."""

    query: str
    chunks: list[RetrievedChunk]
    discarded_below_threshold: int
    threshold: float
    top_k: int
    # True when context came from the document-overview fallback rather than
    # from a similarity match, so the UI and the log can say so.
    overview_fallback: bool = False

    @property
    def has_context(self) -> bool:
        return bool(self.chunks)

    @property
    def best_score(self) -> float:
        return max((chunk.score for chunk in self.chunks), default=0.0)

    @property
    def mean_score(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(chunk.score for chunk in self.chunks) / len(self.chunks)


class Retriever:
    """Turns a natural-language question into a ranked list of source chunks."""

    def __init__(self, vector_store: PineconeVectorStore, embedder: BaseEmbedder) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int = 5,
        similarity_threshold: float = 0.35,
        document_names: Iterable[str] | None = None,
        page_range: tuple[int, int] | None = None,
    ) -> RetrievalResult:
        """Retrieve the ``top_k`` most similar chunks that clear ``similarity_threshold``.

        Raises:
            RetrievalError: on an empty query or an unusable threshold.
        """
        if not query or not query.strip():
            raise RetrievalError("Please enter a question before searching.")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise RetrievalError("The similarity threshold must be between 0.0 and 1.0.")
        top_k = max(1, min(int(top_k), 20))

        query_vector = self.embedder.embed_query(query)
        metadata_filter = build_metadata_filter(document_names, page_range)

        # Over-fetch so that de-duplication cannot starve the final result set.
        matches: list[dict[str, Any]] = self.vector_store.query(
            vector=query_vector,
            namespace=namespace,
            top_k=min(top_k * 3, 60),
            metadata_filter=metadata_filter,
        )

        candidates = [RetrievedChunk.from_match(match) for match in matches]
        kept = [chunk for chunk in candidates if chunk.score >= similarity_threshold]
        discarded = len(candidates) - len(kept)

        deduped = _deduplicate(kept)[:top_k]

        # For a meta-question an empty result means the wrong retrieval mode, not
        # an absent answer, so fall back to the document's opening chunks.
        overview_fallback = False
        if not deduped and looks_like_overview_request(query):
            deduped = self._retrieve_overview(
                query_vector, namespace, top_k, document_names, page_range
            )
            overview_fallback = bool(deduped)

        logger.info(
            "Retrieved %d/%d chunks for %r (threshold=%.2f, filter=%s, overview=%s)",
            len(deduped),
            len(candidates),
            query[:60],
            similarity_threshold,
            metadata_filter,
            overview_fallback,
        )

        return RetrievalResult(
            query=query.strip(),
            chunks=deduped,
            discarded_below_threshold=discarded,
            threshold=similarity_threshold,
            top_k=top_k,
            overview_fallback=overview_fallback,
        )

    def _retrieve_overview(
        self,
        query_vector,
        namespace: str,
        top_k: int,
        document_names: Iterable[str] | None,
        page_range: tuple[int, int] | None,
    ) -> list[RetrievedChunk]:
        """Return the opening chunks of the corpus, ignoring the threshold.

        The metadata filter — not the similarity score — does the selecting
        here, so the result is deterministic and always genuinely from the
        indexed documents.
        """
        overview_filter = build_metadata_filter(
            document_names, page_range, max_chunk_index=OVERVIEW_WINDOW
        )
        try:
            matches = self.vector_store.query(
                vector=query_vector,
                namespace=namespace,
                top_k=min(max(top_k, 1), 20),
                metadata_filter=overview_filter,
            )
        except Exception as exc:  # noqa: BLE001 - the fallback must never break the query
            logger.warning("Overview fallback failed: %s", exc)
            return []

        chunks = [RetrievedChunk.from_match(match) for match in matches]
        # Present them in reading order so the model sees the document's own flow.
        chunks.sort(key=lambda chunk: (chunk.document_name, chunk.chunk_index))
        return _deduplicate(chunks)[:top_k]


def _deduplicate(chunks: list[RetrievedChunk], overlap_ratio: float = 0.80) -> list[RetrievedChunk]:
    """Drop chunks whose text is largely contained in a higher-scoring chunk.

    Chunk overlap intentionally duplicates text across neighbours; without this
    the context window fills with the same sentences repeated.
    """
    kept: list[RetrievedChunk] = []
    seen_shingles: list[set[str]] = []

    for chunk in sorted(chunks, key=lambda item: item.score, reverse=True):
        words = chunk.text.lower().split()
        shingles = {" ".join(words[i : i + 8]) for i in range(max(len(words) - 7, 1))}
        if not shingles:
            continue
        if any(len(shingles & previous) / len(shingles) >= overlap_ratio for previous in seen_shingles):
            continue
        seen_shingles.append(shingles)
        kept.append(chunk)

    return kept
