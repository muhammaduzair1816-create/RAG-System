"""Tests for the document-overview retrieval fallback.

A meta-question ("what is this document about?") scores near zero against the
document's own text, so the similarity gate refuses it even though the answer is
plainly available. These tests pin the fallback's trigger conditions and, most
importantly, prove it does not weaken refusal for genuinely absent answers.
"""

from __future__ import annotations

from src.generator import compute_confidence
from src.models import RetrievedChunk
from src.retriever import (
    OVERVIEW_WINDOW,
    RetrievalResult,
    Retriever,
    looks_like_overview_request,
)
from src.vector_store import build_metadata_filter


# --- Trigger detection ----------------------------------------------------


def test_meta_questions_are_recognised():
    for query in (
        "What is this document about?",
        "what's this report about",
        "Summarise this document",
        "Summarize the paper",
        "Give me an overview",
        "What is the main topic?",
        "TL;DR",
        "What are the key points?",
        "What does this PDF cover?",
    ):
        assert looks_like_overview_request(query), query


def test_content_questions_are_not_recognised():
    for query in (
        "By how much did nesting sites decline?",
        "What threatens the Hawksbill turtle?",
        "What is the capital of France?",
        "Which regions were surveyed in 2024?",
        "How many turtle excluder devices were deployed?",
        "What is the cosine similarity threshold?",
    ):
        assert not looks_like_overview_request(query), query


def test_long_specific_questions_never_trigger_on_a_stray_about():
    query = (
        "What does the report say about the specific decline in Hawksbill nesting "
        "sites across the Coral Triangle between 2010 and 2024 and why"
    )
    assert not looks_like_overview_request(query)


def test_empty_query_does_not_trigger():
    assert not looks_like_overview_request("")
    assert not looks_like_overview_request("   ")


# --- Filter construction --------------------------------------------------


def test_overview_filter_limits_to_opening_chunks():
    built = build_metadata_filter(None, None, max_chunk_index=OVERVIEW_WINDOW)
    assert built == {"chunk_index": {"$lt": OVERVIEW_WINDOW}}


def test_overview_filter_composes_with_user_filters():
    built = build_metadata_filter(["a.pdf"], (2, 6), max_chunk_index=4)
    assert built["document_name"] == {"$in": ["a.pdf"]}
    assert built["page_number"] == {"$gte": 2, "$lte": 6}
    assert built["chunk_index"] == {"$lt": 4}


def test_filter_is_unchanged_when_no_overview_requested():
    assert build_metadata_filter(None, None) is None


# --- Retrieval behaviour ---------------------------------------------------


class StubVectorStore:
    """Returns nothing for the semantic pass, opening chunks for the filtered one."""

    def __init__(self, opening_chunks: list[dict]):
        self.opening_chunks = opening_chunks
        self.filters_seen: list[dict | None] = []

    def query(self, vector, namespace, top_k=5, metadata_filter=None):
        self.filters_seen.append(metadata_filter)
        if metadata_filter and "chunk_index" in metadata_filter:
            return self.opening_chunks
        return []


class StubEmbedder:
    dimension = 3
    model_name = "stub"

    def embed_query(self, text):
        import numpy as np

        return np.array([1.0, 0.0, 0.0])

    def embed_documents(self, texts, batch_size=64):  # pragma: no cover - unused
        import numpy as np

        return np.zeros((len(texts), 3))


def _match(chunk_index: int, score: float = 0.08) -> dict:
    return {
        "id": f"doc::c{chunk_index}",
        "score": score,
        "metadata": {
            "chunk_id": f"doc::c{chunk_index}",
            "text": f"Opening section number {chunk_index} of the marine conservation report.",
            "document_name": "report.pdf",
            "page_number": 1,
            "chunk_index": chunk_index,
        },
    }


def test_overview_question_falls_back_to_opening_chunks():
    store = StubVectorStore([_match(2), _match(0), _match(1)])
    retriever = Retriever(store, StubEmbedder())

    result = retriever.retrieve("What is this document about?", namespace="ns", top_k=5)

    assert result.overview_fallback
    assert result.has_context
    # Reading order, not score order.
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2]
    assert any(
        f and "chunk_index" in f for f in store.filters_seen
    ), "the fallback must use a metadata-filtered query"


def test_unanswerable_question_is_still_refused():
    """The critical guarantee: the fallback must not become a hallucination hole."""
    store = StubVectorStore([_match(0), _match(1)])
    retriever = Retriever(store, StubEmbedder())

    result = retriever.retrieve("What is the capital of France?", namespace="ns", top_k=5)

    assert not result.overview_fallback
    assert not result.has_context  # still refused


def test_fallback_does_not_run_when_similarity_already_matched():
    class MatchingStore(StubVectorStore):
        def query(self, vector, namespace, top_k=5, metadata_filter=None):
            self.filters_seen.append(metadata_filter)
            return [_match(7, score=0.82)]

    store = MatchingStore([])
    retriever = Retriever(store, StubEmbedder())

    result = retriever.retrieve("Summarise this document", namespace="ns", top_k=5)

    assert not result.overview_fallback
    assert result.chunks[0].score == 0.82


def test_fallback_failure_degrades_to_refusal():
    class BrokenStore(StubVectorStore):
        def query(self, vector, namespace, top_k=5, metadata_filter=None):
            if metadata_filter and "chunk_index" in metadata_filter:
                raise RuntimeError("pinecone down")
            return []

    retriever = Retriever(BrokenStore([]), StubEmbedder())
    result = retriever.retrieve("What is this document about?", namespace="ns", top_k=5)

    assert not result.overview_fallback
    assert not result.has_context


# --- Confidence scoring ----------------------------------------------------


def _overview_result(chunks: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(
        query="What is this document about?",
        chunks=chunks,
        discarded_below_threshold=0,
        threshold=0.35,
        top_k=5,
        overview_fallback=True,
    )


def test_overview_confidence_ignores_meaningless_similarity():
    chunk = RetrievedChunk("id", "text", "doc.pdf", 1, 0, score=0.08)
    confidence = compute_confidence(_overview_result([chunk]), cited_fraction=1.0)

    # Not dragged down to ~5% by the irrelevant 0.08 similarity...
    assert confidence > 0.5
    # ...but never presented as a high-confidence pinpoint match.
    assert confidence < 0.70


def test_overview_confidence_still_rewards_citation():
    chunk = RetrievedChunk("id", "text", "doc.pdf", 1, 0, score=0.08)
    cited = compute_confidence(_overview_result([chunk]), cited_fraction=1.0)
    uncited = compute_confidence(_overview_result([chunk]), cited_fraction=0.0)
    assert cited > uncited


def test_normal_confidence_path_is_unchanged():
    chunk = RetrievedChunk("id", "text", "doc.pdf", 1, 0, score=0.90)
    normal = RetrievalResult("q", [chunk], 0, 0.35, 5)
    assert compute_confidence(normal, 1.0) == round(0.55 * 0.90 + 0.25 * 0.90 + 0.20, 3)
