"""Unit tests for retrieval policy, metadata filters and confidence scoring."""

from __future__ import annotations

from src.generator import INSUFFICIENT_CONTEXT_MESSAGE, compute_confidence, format_context, is_refusal
from src.models import RetrievedChunk
from src.retriever import RetrievalResult, _deduplicate
from src.vector_store import _match_to_dict, build_metadata_filter, sanitize_namespace


def make_chunk(text: str, score: float, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc::p{page}::c0",
        text=text,
        document_name="doc.pdf",
        page_number=page,
        chunk_index=0,
        score=score,
    )


def test_from_match_reads_pinecone_payload():
    chunk = RetrievedChunk.from_match(
        {
            "id": "doc::p4::c9",
            "score": 0.8123,
            "metadata": {
                "chunk_id": "doc::p4::c9",
                "text": "Some retrieved text.",
                "document_name": "doc.pdf",
                "page_number": 4.0,  # Pinecone returns numeric metadata as floats
                "chunk_index": 9.0,
            },
        }
    )
    assert chunk.page_number == 4 and chunk.chunk_index == 9
    assert chunk.score == 0.8123


def test_deduplicate_drops_near_identical_chunks():
    body = "The system stores embeddings in Pinecone using cosine similarity for retrieval. " * 3
    chunks = [make_chunk(body, 0.9), make_chunk(body + " Extra tail.", 0.8), make_chunk("A totally different topic about weather patterns and rainfall.", 0.7)]

    kept = _deduplicate(chunks)
    assert len(kept) == 2
    assert kept[0].score == 0.9


def test_deduplicate_preserves_score_ordering():
    kept = _deduplicate([make_chunk("alpha beta gamma delta epsilon zeta eta theta", 0.4),
                         make_chunk("one two three four five six seven eight", 0.9)])
    assert [chunk.score for chunk in kept] == [0.9, 0.4]


def test_retrieval_result_aggregates():
    result = RetrievalResult(
        query="q", chunks=[make_chunk("a", 0.9), make_chunk("b", 0.5)],
        discarded_below_threshold=3, threshold=0.35, top_k=5,
    )
    assert result.has_context
    assert result.best_score == 0.9
    assert result.mean_score == 0.7


def test_empty_result_reports_no_context():
    result = RetrievalResult(query="q", chunks=[], discarded_below_threshold=4, threshold=0.6, top_k=5)
    assert not result.has_context
    assert result.best_score == 0.0 and result.mean_score == 0.0


def test_confidence_is_zero_without_context():
    result = RetrievalResult(query="q", chunks=[], discarded_below_threshold=0, threshold=0.35, top_k=5)
    assert compute_confidence(result, 1.0) == 0.0


def test_confidence_rewards_strong_and_well_cited_retrieval():
    strong = RetrievalResult(query="q", chunks=[make_chunk("a", 0.9), make_chunk("b", 0.85)],
                             discarded_below_threshold=0, threshold=0.35, top_k=5)
    weak = RetrievalResult(query="q", chunks=[make_chunk("a", 0.4), make_chunk("b", 0.36)],
                           discarded_below_threshold=0, threshold=0.35, top_k=5)

    assert compute_confidence(strong, 1.0) > compute_confidence(weak, 1.0)
    assert compute_confidence(strong, 1.0) > compute_confidence(strong, 0.0)
    assert 0.0 <= compute_confidence(strong, 1.0) <= 1.0


def test_context_blocks_are_numbered_and_attributed():
    context = format_context([make_chunk("First fact.", 0.91, page=2), make_chunk("Second fact.", 0.72, page=7)])
    assert "[S1]" in context and "[S2]" in context
    assert "page: 2" in context and "page: 7" in context
    assert "First fact." in context


def test_bare_refusal_is_detected():
    assert is_refusal(INSUFFICIENT_CONTEXT_MESSAGE)
    assert is_refusal(f"  {INSUFFICIENT_CONTEXT_MESSAGE}  ")
    assert is_refusal(INSUFFICIENT_CONTEXT_MESSAGE.upper())


def test_partial_answer_quoting_the_refusal_is_not_a_refusal():
    partial = (
        "The document states that the index uses the cosine metric [S1] and stores the page "
        "number in metadata [S2]. It does not describe the hardware used for benchmarking; for "
        f"that part, {INSUFFICIENT_CONTEXT_MESSAGE}"
    )
    assert not is_refusal(partial)


def test_grounded_answer_is_not_a_refusal():
    assert not is_refusal("Pinecone is configured with a cosine metric [S1].")


def test_metadata_filter_composition():
    assert build_metadata_filter(None, None) is None
    assert build_metadata_filter(["a.pdf"], None) == {"document_name": {"$in": ["a.pdf"]}}

    combined = build_metadata_filter(["a.pdf", "b.pdf"], (10, 4))
    assert combined["page_number"] == {"$gte": 4, "$lte": 10}  # reversed input is normalised
    assert combined["document_name"] == {"$in": ["a.pdf", "b.pdf"]}


def test_match_normalisation_accepts_every_sdk_shape():
    payload = {"id": "a", "score": 0.5, "metadata": {"text": "t"}}

    class WithToDict:
        def to_dict(self):
            return payload

    class WithAttributes:
        id, score, metadata = "a", 0.5, {"text": "t"}

    assert _match_to_dict(payload) == payload
    assert _match_to_dict(WithToDict()) == payload
    assert _match_to_dict(WithAttributes()) == payload


def test_namespace_sanitisation():
    assert sanitize_namespace("My Corpus #1") == "my-corpus-1"
    assert sanitize_namespace("   ") == "default"
    assert len(sanitize_namespace("x" * 200)) == 48
