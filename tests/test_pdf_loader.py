"""Unit tests for PDF validation and text cleaning."""

from __future__ import annotations

import pytest

from src.pdf_loader import PDFProcessingError, clean_text, validate_pdf_bytes


def test_hyphenated_line_breaks_are_rejoined():
    assert "retrieval" in clean_text("retrie-\nval augmented generation")


def test_soft_wraps_become_spaces_but_paragraphs_survive():
    cleaned = clean_text("This sentence wraps\nacross lines.\n\nA new paragraph starts here.")
    assert "wraps across lines" in cleaned
    assert "\n\n" in cleaned


def test_standalone_page_numbers_are_dropped():
    cleaned = clean_text("Body of the page.\n\n12\n\nPage 3 of 40\n\nMore body text.")
    assert "12" not in cleaned.split()
    assert "Page 3 of 40" not in cleaned
    assert "More body text." in cleaned


def test_ligatures_and_control_characters_are_normalised():
    cleaned = clean_text("efﬁcient workﬂow\x00 here")
    assert "efficient" in cleaned
    assert "workflow" in cleaned
    assert "\x00" not in cleaned


def test_excess_whitespace_collapses():
    assert clean_text("too     many      spaces") == "too many spaces"


def test_non_pdf_extension_is_rejected():
    with pytest.raises(PDFProcessingError, match="not a PDF"):
        validate_pdf_bytes(b"%PDF-1.4 data", "notes.txt")


def test_empty_upload_is_rejected():
    with pytest.raises(PDFProcessingError, match="empty"):
        validate_pdf_bytes(b"", "empty.pdf")


def test_corrupt_header_is_rejected():
    with pytest.raises(PDFProcessingError, match="corrupted"):
        validate_pdf_bytes(b"this is not a pdf at all", "broken.pdf")


def test_oversized_file_is_rejected():
    oversized = b"%PDF-1.4" + b"0" * (21 * 1024 * 1024)
    with pytest.raises(PDFProcessingError, match="limit"):
        validate_pdf_bytes(oversized, "huge.pdf")
