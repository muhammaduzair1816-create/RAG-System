"""Stage 1–2: PDF ingestion and text cleaning.

Responsibilities:
  * validate the uploaded file (extension, size, parseability)
  * extract per-page text with ``pypdf``
  * fall back to OCR for pages that carry no extractable text (scans)
  * strip the formatting artifacts that PDF extraction reliably produces
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Callable

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from . import ocr as ocr_module
from .config import Settings, get_settings
from .models import Document, PageText

logger = logging.getLogger(__name__)


class PDFProcessingError(ValueError):
    """Raised for any user-fixable problem with the supplied PDF."""


# --- Cleaning rules -------------------------------------------------------

# "hyphen-\nated" line breaks introduced by justified typesetting
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
# A single newline inside a sentence is a wrap, not a paragraph break
_SOFT_WRAP = re.compile(r"(?<![\n.!?:;])\n(?![\n•\-\*\d])")
# Three or more blank lines collapse to one paragraph break
_EXCESS_BLANKS = re.compile(r"\n{3,}")
# Repeated spaces / tabs / non-breaking spaces
_EXCESS_SPACES = re.compile(r"[ \t ​]{2,}")
# Control characters and the replacement glyph that broken fonts emit
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")
# Lines that are only page furniture, e.g. "12", "- 12 -", "Page 12 of 40"
_PAGE_FURNITURE = re.compile(r"^\s*(?:page\s+)?[-–—]?\s*\d+\s*(?:of\s+\d+)?\s*[-–—]?\s*$", re.IGNORECASE)
# Ligatures that pypdf sometimes leaves as single glyphs
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean_text(raw: str) -> str:
    """Normalise raw PDF text into clean prose suitable for chunking."""
    if not raw:
        return ""

    text = raw
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)

    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # Drop stand-alone page numbers before newlines get rewritten.
    kept_lines = [line for line in text.split("\n") if not _PAGE_FURNITURE.match(line)]
    text = "\n".join(kept_lines)

    text = _SOFT_WRAP.sub(" ", text)
    text = _EXCESS_SPACES.sub(" ", text)
    text = _EXCESS_BLANKS.sub("\n\n", text)

    return "\n".join(line.strip() for line in text.split("\n")).strip()


# --- Validation -----------------------------------------------------------


def validate_pdf_bytes(data: bytes, filename: str, settings: Settings | None = None) -> None:
    """Raise :class:`PDFProcessingError` if the payload is not a usable PDF."""
    settings = settings or get_settings()

    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError(f"'{filename}' is not a PDF file. Only .pdf uploads are supported.")
    if not data:
        raise PDFProcessingError(f"'{filename}' is empty (0 bytes).")
    if len(data) > settings.max_pdf_size_bytes:
        size_mb = len(data) / (1024 * 1024)
        raise PDFProcessingError(
            f"'{filename}' is {size_mb:.1f} MB, above the {settings.max_pdf_size_mb} MB limit."
        )
    if not data.lstrip()[:5].startswith(b"%PDF-"):
        raise PDFProcessingError(f"'{filename}' does not have a valid PDF header and appears corrupted.")


# --- Extraction -----------------------------------------------------------


def _open_reader(data: bytes, filename: str) -> PdfReader:
    """Open ``data`` with pypdf, translating library errors into user-facing ones."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise PDFProcessingError(f"'{filename}' could not be parsed as a PDF: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - pypdf raises assorted low-level errors
        raise PDFProcessingError(f"Unexpected error while opening '{filename}': {exc}") from exc

    if reader.is_encrypted:
        # An empty user password is common for "print-protected" files; try it.
        try:
            if reader.decrypt("") == 0:
                raise PDFProcessingError(
                    f"'{filename}' is password-protected. Please upload an unlocked copy."
                )
        except PDFProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PDFProcessingError(f"'{filename}' is encrypted and could not be opened: {exc}") from exc

    return reader


def _extract_native_text(reader: PdfReader) -> dict[int, str]:
    """Cleaned embedded text for every page, keyed by 1-indexed page number."""
    native: dict[int, str] = {}
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one broken page must not abort the document
            logger.warning("Text extraction failed on page %d: %s", index, exc)
            raw = ""
        native[index] = clean_text(raw)
    return native


def load_pdf(
    data: bytes,
    filename: str,
    settings: Settings | None = None,
    ocr_progress_callback: Callable[[int, int], None] | None = None,
) -> Document:
    """Extract and clean the text of every page of ``data``.

    Pages that yield little or no embedded text are automatically passed to OCR
    when the toolchain is available (see :mod:`src.ocr`); the recovered text is
    merged with whatever was extracted natively. Text PDFs never touch OCR, so
    their behaviour is exactly as before.

    Raises:
        PDFProcessingError: invalid, encrypted, corrupt, or text-free PDF.
    """
    settings = settings or get_settings()
    validate_pdf_bytes(data, filename, settings)

    reader = _open_reader(data, filename)
    total_pages = len(reader.pages)
    native = _extract_native_text(reader)

    # --- Decide which pages need OCR ------------------------------------
    ocr_candidates = [
        number for number, text in native.items() if ocr_module.needs_ocr(text, settings.ocr_min_chars)
    ]

    ocr_text: dict[int, str] = {}
    ocr_attempted = False
    ocr_warning: str | None = None

    if ocr_candidates and settings.ocr_enabled:
        availability = ocr_module.check_availability(settings)
        if availability.available:
            ocr_attempted = True
            logger.info(
                "'%s': %d of %d page(s) have no extractable text; attempting OCR",
                filename,
                len(ocr_candidates),
                total_pages,
            )
            try:
                ocr_text = ocr_module.ocr_page_numbers(
                    data, ocr_candidates, settings, progress_callback=ocr_progress_callback
                )
            except ocr_module.OCRError as exc:
                # OCR is a fallback: record why it failed and carry on with
                # whatever native text exists rather than losing the upload.
                ocr_warning = str(exc)
                logger.warning("OCR failed for '%s': %s", filename, exc)
        else:
            ocr_warning = availability.reason
            logger.info("OCR skipped for '%s': %s", filename, availability.reason)
    elif ocr_candidates and not settings.ocr_enabled:
        ocr_warning = "OCR is disabled (OCR_ENABLED=false), so image-only pages were skipped."

    # --- Merge and assemble ---------------------------------------------
    pages: list[PageText] = []
    ocr_pages: list[int] = []

    for number in sorted(native):
        native_text = native[number]
        recovered = clean_text(ocr_text.get(number, ""))

        if recovered:
            combined = ocr_module.merge_page_text(native_text, recovered)
            source = "merged" if native_text.strip() else "ocr"
            ocr_pages.append(number)
        else:
            combined, source = native_text, "native"

        if combined.strip():
            pages.append(PageText(page_number=number, text=combined, source=source))

    document = Document(
        document_name=Path(filename).name,
        pages=pages,
        total_pages=total_pages,
        ocr_pages=ocr_pages,
        ocr_attempted=ocr_attempted,
        ocr_warning=ocr_warning,
    )

    if document.is_empty:
        raise PDFProcessingError(_empty_document_message(filename, ocr_attempted, ocr_warning))

    if ocr_pages:
        logger.info("'%s': OCR supplied text for page(s) %s", filename, ocr_pages)

    return document


def _empty_document_message(filename: str, ocr_attempted: bool, ocr_warning: str | None) -> str:
    """Explain an empty extraction in terms of what was actually tried."""
    if ocr_attempted:
        return (
            f"No text could be extracted from '{filename}', even with OCR. The scan may be "
            "too low-resolution or too noisy to read. Try a higher-quality scan, or raise "
            "OCR_DPI in your .env."
        )
    if ocr_warning:
        return (
            f"No selectable text found in '{filename}' — it appears to be a scanned image PDF, "
            f"and OCR could not run.\n\n{ocr_warning}\n\n{ocr_module.INSTALL_HINT}"
        )
    return (
        f"No selectable text found in '{filename}'. It is most likely a scanned "
        "image PDF — install the OCR toolchain (see the README) and re-upload."
    )
