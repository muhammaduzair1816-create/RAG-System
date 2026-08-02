"""Unit tests for the OCR fallback.

These never invoke Tesseract: the detection, merging and degradation logic is
pure Python, and the OCR call itself is stubbed where a test needs it.
"""

from __future__ import annotations

import pytest

from src import ocr as ocr_module
from src.config import Settings
from src.models import Document, PageText
from src.ocr import (
    OCRAvailability,
    OCRUnavailableError,
    merge_page_text,
    needs_ocr,
)


# --- Page-level detection -------------------------------------------------


def test_blank_page_needs_ocr():
    assert needs_ocr("", 80)
    assert needs_ocr("   \n  \t ", 80)


def test_page_with_stray_watermark_text_still_needs_ocr():
    # Scans often extract a few characters from a stamp or watermark layer.
    assert needs_ocr("CONFIDENTIAL", 80)


def test_full_text_page_does_not_need_ocr():
    page = "This page carries a full paragraph of genuine extractable text. " * 3
    assert not needs_ocr(page, 80)


def test_threshold_boundary_is_exclusive():
    assert needs_ocr("x" * 79, 80)
    assert not needs_ocr("x" * 80, 80)


def test_zero_threshold_only_catches_empty_pages():
    assert not needs_ocr("a", 0)
    assert not needs_ocr("", 0)


# --- Merging native and OCR text ------------------------------------------


def test_ocr_only_page_uses_ocr_text():
    assert merge_page_text("", "Recovered by OCR.") == "Recovered by OCR."


def test_page_without_ocr_keeps_native_text():
    assert merge_page_text("Native text.", "") == "Native text."


def test_ocr_superset_replaces_native_instead_of_duplicating():
    native = "Quarterly revenue increased sharply"
    ocr = "Quarterly revenue increased sharply across every regional market segment"

    merged = merge_page_text(native, ocr)

    assert merged == ocr
    assert merged.count("Quarterly") == 1  # no duplication into the chunker


def test_genuinely_different_content_is_concatenated():
    native = "The paragraph typeset as real text in the PDF body."
    ocr = "Figure 4 caption rendered only inside a scanned bitmap."

    merged = merge_page_text(native, ocr)

    assert native in merged and ocr in merged


def test_merge_is_whitespace_safe():
    assert merge_page_text("  ", "  OCR text  ") == "OCR text"


# --- Degradation when the toolchain is absent ------------------------------


def test_ocr_raises_unavailable_when_toolchain_missing(monkeypatch):
    monkeypatch.setattr(
        ocr_module,
        "check_availability",
        lambda settings=None: OCRAvailability(available=False, reason="Tesseract not installed."),
    )
    with pytest.raises(OCRUnavailableError, match="Tesseract not installed"):
        ocr_module.ocr_page_numbers(b"%PDF-1.4", [1], Settings())


def test_no_pages_requested_short_circuits(monkeypatch):
    monkeypatch.setattr(
        ocr_module,
        "check_availability",
        lambda settings=None: OCRAvailability(available=True, tesseract_version="5.3.0"),
    )
    assert ocr_module.ocr_page_numbers(b"%PDF-1.4", [], Settings()) == {}


def test_availability_reports_disabled_state():
    ocr_module.reset_availability_cache()
    availability = ocr_module.check_availability(Settings(ocr_enabled=False))
    assert not availability.available
    assert "disabled" in availability.reason.lower()


def test_availability_never_raises_and_has_a_summary():
    ocr_module.reset_availability_cache()
    availability = ocr_module.check_availability(Settings())
    assert isinstance(availability.available, bool)
    assert availability.summary  # always renderable in the sidebar


def test_install_hint_names_both_binaries():
    assert "Tesseract" in ocr_module.INSTALL_HINT
    assert "Poppler" in ocr_module.INSTALL_HINT


# --- Document-level OCR bookkeeping ---------------------------------------


def test_document_tracks_ocr_pages():
    document = Document(
        document_name="scan.pdf",
        pages=[PageText(page_number=1, text="from ocr", source="ocr")],
        total_pages=1,
        ocr_pages=[1],
        ocr_attempted=True,
    )
    assert document.ocr_used
    assert not document.is_empty


def test_document_defaults_report_no_ocr():
    document = Document(
        document_name="text.pdf",
        pages=[PageText(page_number=1, text="native text")],
        total_pages=1,
    )
    assert not document.ocr_used
    assert document.ocr_pages == []
    assert document.ocr_attempted is False
    assert document.ocr_warning is None
    assert document.pages[0].source == "native"  # unchanged default for text PDFs
