"""Unit tests for chunking — no API keys or network required."""

from __future__ import annotations

import pytest

from src.chunker import ChunkingError, chunk_document
from src.models import Document, PageText


def make_document(pages: list[str], name: str = "sample.pdf") -> Document:
    return Document(
        document_name=name,
        pages=[PageText(page_number=i, text=text) for i, text in enumerate(pages, start=1)],
        total_pages=len(pages),
    )


def test_chunks_respect_page_boundaries():
    document = make_document(["Alpha " * 200, "Beta " * 200])
    chunks = chunk_document(document, chunk_size=300, chunk_overlap=0)

    pages = {chunk.page_number for chunk in chunks}
    assert pages == {1, 2}
    for chunk in chunks:
        expected = "Alpha" if chunk.page_number == 1 else "Beta"
        assert expected in chunk.text


def test_chunk_ids_are_unique_and_carry_metadata():
    document = make_document(["Sentence one. " * 120])
    chunks = chunk_document(document, chunk_size=250, chunk_overlap=50)

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        metadata = chunk.to_metadata()
        assert metadata["document_name"] == "sample.pdf"
        assert metadata["page_number"] == 1
        assert metadata["text"]


def test_overlap_carries_context_forward():
    document = make_document(["First idea here. Second idea follows. Third idea closes. " * 30])
    with_overlap = chunk_document(document, chunk_size=400, chunk_overlap=150)
    without_overlap = chunk_document(document, chunk_size=400, chunk_overlap=0)

    assert len(with_overlap) == len(without_overlap)
    # Overlapped chunks after the first carry extra leading text.
    assert len(with_overlap[1].text) > len(without_overlap[1].text)


def test_overlap_never_starts_mid_word():
    document = make_document(["The quick brown fox jumps over the lazy dog every single morning. " * 12])
    chunks = chunk_document(document, chunk_size=300, chunk_overlap=90)

    words = set(document.pages[0].text.split())
    for chunk in chunks[1:]:
        assert chunk.text.split()[0] in words, f"chunk begins mid-word: {chunk.text[:40]!r}"


def test_overlap_applies_when_tail_ends_on_a_sentence_boundary():
    # The tail slice ends exactly at ". ", which must not collapse the overlap away.
    document = make_document(["Sentence about alpha ends here. Sentence about beta follows on. " * 10])
    overlapped = chunk_document(document, chunk_size=320, chunk_overlap=80)
    plain = chunk_document(document, chunk_size=320, chunk_overlap=0)

    assert len(overlapped[1].text) > len(plain[1].text)


def test_sentence_punctuation_is_preserved_across_splits():
    document = make_document(["First fact stated plainly. Second fact stated plainly. " * 15])
    chunks = chunk_document(document, chunk_size=260, chunk_overlap=0)

    joined = " ".join(chunk.text for chunk in chunks)
    assert "plainly First" not in joined  # the period must survive the split
    assert joined.count(".") >= 25


def test_splitter_prefers_paragraph_boundaries():
    text = "\n\n".join(["Paragraph number {} content.".format(i) * 4 for i in range(6)])
    chunks = chunk_document(make_document([text]), chunk_size=200, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(chunk.text.strip() for chunk in chunks)


def test_oversized_token_is_hard_split():
    document = make_document(["x" * 1000])
    chunks = chunk_document(document, chunk_size=200, chunk_overlap=0)

    assert len(chunks) >= 5
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_invalid_parameters_are_rejected():
    document = make_document(["some text " * 50])

    with pytest.raises(ChunkingError):
        chunk_document(document, chunk_size=500, chunk_overlap=500)
    with pytest.raises(ChunkingError):
        chunk_document(document, chunk_size=50, chunk_overlap=0)


def test_namespace_salt_changes_ids():
    document = make_document(["Repeatable content. " * 60])
    a = chunk_document(document, 300, 0, namespace_salt="ns-a")
    b = chunk_document(document, 300, 0, namespace_salt="ns-b")

    assert [chunk.chunk_id for chunk in a] != [chunk.chunk_id for chunk in b]
