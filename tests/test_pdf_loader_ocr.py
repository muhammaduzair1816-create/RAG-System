"""Integration tests for the OCR fallback inside the loader.

Tesseract is never called: `ocr_page_numbers` is stubbed so these run anywhere.
What they verify is the wiring — when OCR is invoked, when it is skipped, how
its output is merged, and how failures degrade.
"""

from __future__ import annotations

import pytest

from src import ocr as ocr_module
from src.config import Settings
from src.ocr import OCRAvailability, OCRUnavailableError
from src.pdf_loader import PDFProcessingError, load_pdf

from .pdf_fixtures import build_scanned_pdf, build_text_pdf

TEXT_PAGES = [
    [
        "Retrieval Augmented Generation Overview",
        "RAG systems combine a retriever with a generative language model so that",
        "answers stay anchored to a trusted corpus instead of model memory.",
    ],
    [
        "Pinecone Configuration",
        "The index is serverless and uses the cosine metric with 384 dimensions.",
        "Each vector carries the document name, page number and chunk identifier.",
    ],
]


@pytest.fixture
def available(monkeypatch):
    """Pretend the OCR toolchain is installed and working."""
    monkeypatch.setattr(
        ocr_module,
        "check_availability",
        lambda settings=None: OCRAvailability(available=True, tesseract_version="5.3.0"),
    )


@pytest.fixture
def unavailable(monkeypatch):
    """Pretend Tesseract is not installed."""
    monkeypatch.setattr(
        ocr_module,
        "check_availability",
        lambda settings=None: OCRAvailability(
            available=False, reason="The Tesseract OCR program was not found on PATH."
        ),
    )


# --- The existing text path must be untouched -----------------------------


def test_text_pdf_never_invokes_ocr(monkeypatch, available):
    def explode(*args, **kwargs):
        raise AssertionError("OCR must not run for a text PDF")

    monkeypatch.setattr(ocr_module, "ocr_page_numbers", explode)

    document = load_pdf(build_text_pdf(TEXT_PAGES), "text.pdf", Settings())

    assert len(document.pages) == 2
    assert not document.ocr_used
    assert document.ocr_attempted is False
    assert all(page.source == "native" for page in document.pages)
    assert "Retrieval Augmented Generation" in document.pages[0].text


def test_text_pdf_cleaning_still_applies(available):
    document = load_pdf(build_text_pdf(TEXT_PAGES), "text.pdf", Settings())
    assert document.total_pages == 2
    assert document.pages[1].page_number == 2
    assert "cosine metric" in document.pages[1].text


# --- The scanned path ------------------------------------------------------


def test_scanned_pdf_is_detected_and_ocr_supplies_the_text(monkeypatch, available):
    seen: dict[str, object] = {}

    def fake_ocr(data, page_numbers, settings=None, progress_callback=None):
        seen["pages"] = list(page_numbers)
        return {number: f"Recovered text for page {number}. " * 6 for number in page_numbers}

    monkeypatch.setattr(ocr_module, "ocr_page_numbers", fake_ocr)

    scanned = build_scanned_pdf([["Annual Safety Report"], ["Appendix"]])
    document = load_pdf(scanned, "scan.pdf", Settings())

    assert seen["pages"] == [1, 2]  # both pages had no extractable text
    assert document.ocr_used
    assert document.ocr_pages == [1, 2]
    assert document.ocr_attempted is True
    assert document.ocr_warning is None
    assert all(page.source == "ocr" for page in document.pages)
    assert "Recovered text for page 1." in document.pages[0].text


def test_ocr_progress_callback_is_forwarded(monkeypatch, available):
    def fake_ocr(data, page_numbers, settings=None, progress_callback=None):
        assert progress_callback is not None
        progress_callback(1, len(page_numbers))
        return {number: "Recovered text. " * 10 for number in page_numbers}

    monkeypatch.setattr(ocr_module, "ocr_page_numbers", fake_ocr)

    calls: list[tuple[int, int]] = []
    load_pdf(
        build_scanned_pdf([["Scanned"]]),
        "scan.pdf",
        Settings(),
        ocr_progress_callback=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(1, 1)]


def test_mixed_document_ocrs_only_the_image_pages(monkeypatch, available):
    """A text PDF whose second page is blank: only that page should go to OCR."""
    requested: list[int] = []

    def fake_ocr(data, page_numbers, settings=None, progress_callback=None):
        requested.extend(page_numbers)
        return {number: "Text recovered from the scanned page. " * 5 for number in page_numbers}

    monkeypatch.setattr(ocr_module, "ocr_page_numbers", fake_ocr)

    mixed = build_text_pdf([TEXT_PAGES[0], [""]])
    document = load_pdf(mixed, "mixed.pdf", Settings())

    assert requested == [2]
    assert document.ocr_pages == [2]
    assert document.pages[0].source == "native"
    assert document.pages[1].source == "ocr"


# --- Degradation -----------------------------------------------------------


def test_scanned_pdf_without_tesseract_gives_installation_guidance(unavailable):
    with pytest.raises(PDFProcessingError) as info:
        load_pdf(build_scanned_pdf([["Annual Safety Report"]]), "scan.pdf", Settings())

    message = str(info.value)
    assert "Tesseract" in message
    assert "scanned image PDF" in message
    assert "poppler" in message.lower()  # both binaries are named


def test_ocr_failure_does_not_lose_the_native_pages(monkeypatch, available):
    def failing_ocr(*args, **kwargs):
        raise OCRUnavailableError("Poppler is not installed.")

    monkeypatch.setattr(ocr_module, "ocr_page_numbers", failing_ocr)

    mixed = build_text_pdf([TEXT_PAGES[0], [""]])
    document = load_pdf(mixed, "mixed.pdf", Settings())

    # Page 1 survives even though OCR blew up on page 2.
    assert len(document.pages) == 1
    assert document.pages[0].source == "native"
    assert not document.ocr_used
    assert "Poppler is not installed." in (document.ocr_warning or "")


def test_ocr_can_be_disabled_by_configuration(monkeypatch, available):
    def explode(*args, **kwargs):
        raise AssertionError("OCR must not run when OCR_ENABLED is false")

    monkeypatch.setattr(ocr_module, "ocr_page_numbers", explode)

    with pytest.raises(PDFProcessingError):
        load_pdf(build_scanned_pdf([["Scanned"]]), "scan.pdf", Settings(ocr_enabled=False))


def test_ocr_that_recovers_nothing_reports_a_scan_quality_problem(monkeypatch, available):
    monkeypatch.setattr(
        ocr_module, "ocr_page_numbers", lambda *args, **kwargs: {}
    )

    with pytest.raises(PDFProcessingError, match="even with OCR"):
        load_pdf(build_scanned_pdf([["Unreadable"]]), "scan.pdf", Settings())
