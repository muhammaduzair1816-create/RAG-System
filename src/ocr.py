"""Optional OCR stage for scanned / image-only PDFs.

Supplies text only for pages where ``pypdf`` found little or nothing to extract.
Everything downstream receives an ordinary :class:`~src.models.Document`.

The stack is ``pdf2image`` (rasterise a page via Poppler) plus ``pytesseract``
(read the raster via the Tesseract binary). Both need system binaries that pip
cannot install, so every entry point here degrades to an actionable message
instead of raising: OCR is an enhancement, never a hard dependency.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class OCRError(RuntimeError):
    """Raised when OCR was attempted and failed for a reason worth surfacing."""


class OCRUnavailableError(OCRError):
    """Raised when the OCR toolchain is not installed or not reachable."""


# Default install locations the Windows/macOS/Linux packages use, checked before
# giving up so a standard install works with no configuration at all.
_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"{localappdata}\Programs\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)

INSTALL_HINT = (
    "OCR needs two system programs that pip cannot install:\n"
    "  • Tesseract OCR — Windows: https://github.com/UB-Mannheim/tesseract/wiki · "
    "macOS: `brew install tesseract` · Debian/Ubuntu: `sudo apt install tesseract-ocr`\n"
    "  • Poppler (used to rasterise PDF pages) — Windows: "
    "https://github.com/oschwartz10612/poppler-windows/releases · "
    "macOS: `brew install poppler` · Debian/Ubuntu: `sudo apt install poppler-utils`\n"
    "On Windows, either add both to PATH or set TESSERACT_CMD and POPPLER_PATH in your .env."
)


@dataclass(frozen=True)
class OCRAvailability:
    """Result of probing the local OCR toolchain."""

    available: bool
    tesseract_version: str = ""
    tesseract_path: str = ""
    poppler_path: str = ""
    reason: str = ""

    @property
    def summary(self) -> str:
        if self.available:
            return f"Tesseract {self.tesseract_version} ready"
        return self.reason or "OCR unavailable"


# --- Toolchain discovery --------------------------------------------------


def _resolve_tesseract(settings: Settings) -> str:
    """Return a usable tesseract executable path, or "" if none is found."""
    if settings.tesseract_cmd:
        # An explicit setting is authoritative — report it even if it is wrong,
        # so the user sees their own value in the error message.
        return settings.tesseract_cmd

    found = shutil.which("tesseract")
    if found:
        return found

    localappdata = os.environ.get("LOCALAPPDATA", "")
    for candidate in _TESSERACT_CANDIDATES:
        path = candidate.format(localappdata=localappdata) if "{" in candidate else candidate
        if path and Path(path).is_file():
            return path
    return ""


def _resolve_poppler(settings: Settings) -> str:
    """Return the Poppler ``bin`` directory, or "" to mean "already on PATH"."""
    if settings.poppler_path:
        return settings.poppler_path
    return ""  # pdf2image falls back to PATH, which is the normal Unix case


def _poppler_on_path(poppler_path: str) -> bool:
    if poppler_path:
        binary = Path(poppler_path) / ("pdftoppm.exe" if os.name == "nt" else "pdftoppm")
        return binary.is_file() or Path(poppler_path).is_dir()
    return shutil.which("pdftoppm") is not None


@lru_cache(maxsize=4)
def _probe(tesseract_cmd: str, poppler_path: str, enabled: bool) -> OCRAvailability:
    """Cached probe. Keyed on settings so a config change re-probes."""
    if not enabled:
        return OCRAvailability(available=False, reason="OCR is disabled (OCR_ENABLED=false).")

    # Probe with find_spec, never a real import: importing pytesseract and
    # pdf2image costs ~70 MB and this runs on every page render.
    missing = [name for name in ("pytesseract", "pdf2image") if find_spec(name) is None]
    if missing:
        return OCRAvailability(
            available=False,
            reason=(
                f"Python OCR packages are missing ({', '.join(missing)}). "
                "Run `pip install -r requirements.txt`."
            ),
        )

    if not tesseract_cmd:
        return OCRAvailability(
            available=False,
            reason="The Tesseract OCR program was not found on PATH or in the usual install locations.",
        )

    # Ask the binary directly rather than via pytesseract, for the same reason.
    try:
        completed = subprocess.run(
            [tesseract_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except Exception as exc:  # noqa: BLE001 - missing binary, bad path, permissions, timeout
        return OCRAvailability(
            available=False,
            tesseract_path=tesseract_cmd,
            reason=f"Tesseract was found at '{tesseract_cmd}' but could not be run: {exc}",
        )

    # First token of the first line, e.g. "tesseract 5.5.0" -> "5.5.0".
    first_line = (completed.stdout or completed.stderr or "").splitlines()
    parts = first_line[0].split() if first_line else []
    version = parts[1] if len(parts) > 1 else "unknown"

    if not _poppler_on_path(poppler_path):
        return OCRAvailability(
            available=False,
            tesseract_version=version,
            tesseract_path=tesseract_cmd,
            reason=(
                "Tesseract is installed but Poppler (pdftoppm) was not found. "
                "pdf2image needs Poppler to turn PDF pages into images."
            ),
        )

    logger.info("OCR toolchain ready: tesseract %s at %s", version, tesseract_cmd)
    return OCRAvailability(
        available=True,
        tesseract_version=version,
        tesseract_path=tesseract_cmd,
        poppler_path=poppler_path,
    )


def check_availability(settings: Settings | None = None) -> OCRAvailability:
    """Probe the OCR toolchain. Never raises — inspect the returned object."""
    settings = settings or get_settings()
    return _probe(
        _resolve_tesseract(settings),
        _resolve_poppler(settings),
        settings.ocr_enabled,
    )


def reset_availability_cache() -> None:
    """Forget the cached probe (used by tests and after a config change)."""
    _probe.cache_clear()


# --- Page-level decision ---------------------------------------------------


def needs_ocr(page_text: str, min_chars: int) -> bool:
    """True when a page's extracted text is too thin to be real content.

    A scanned page usually extracts to nothing at all, but some produce a few
    stray characters from a header stamp or a watermark, so this is a threshold
    rather than an emptiness test.
    """
    return len(page_text.strip()) < max(min_chars, 0)


def _word_coverage(candidate: str, reference: str) -> float:
    """Fraction of ``reference``'s words that also appear in ``candidate``."""
    reference_words = [word for word in reference.lower().split() if len(word) > 2]
    if not reference_words:
        return 1.0
    candidate_words = set(candidate.lower().split())
    hits = sum(1 for word in reference_words if word in candidate_words)
    return hits / len(reference_words)


def merge_page_text(native: str, ocr: str, coverage_threshold: float = 0.80) -> str:
    """Combine the embedded page text with the OCR reading of the same page.

    OCR re-reads the whole rendered page, so blind concatenation would duplicate
    the native text. When the OCR output already covers it, keep only the OCR
    version; otherwise the two are different content and both are kept.
    """
    native, ocr = native.strip(), ocr.strip()
    if not ocr:
        return native
    if not native:
        return ocr
    if _word_coverage(ocr, native) >= coverage_threshold:
        return ocr
    return f"{native}\n\n{ocr}"


# --- OCR execution ---------------------------------------------------------


def ocr_page_numbers(
    data: bytes,
    page_numbers: list[int],
    settings: Settings | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, str]:
    """OCR the given 1-indexed pages of ``data``.

    Returns a ``{page_number: raw_text}`` mapping containing only pages that
    actually produced text. Individual page failures are logged and skipped so
    one bad page cannot lose the rest of the document.

    Raises:
        OCRUnavailableError: the toolchain is not installed or not runnable.
    """
    settings = settings or get_settings()
    availability = check_availability(settings)
    if not availability.available:
        raise OCRUnavailableError(availability.reason)

    if not page_numbers:
        return {}

    import pytesseract
    from pdf2image import convert_from_bytes

    pytesseract.pytesseract.tesseract_cmd = availability.tesseract_path

    targets = sorted(set(page_numbers))[: max(settings.ocr_max_pages, 0)]
    convert_kwargs = {"dpi": settings.ocr_dpi, "fmt": "png"}
    if availability.poppler_path:
        convert_kwargs["poppler_path"] = availability.poppler_path

    results: dict[int, str] = {}
    logger.info("Running OCR on %d page(s) at %d DPI", len(targets), settings.ocr_dpi)

    for position, page_number in enumerate(targets, start=1):
        try:
            # Render one page at a time: a 300 DPI A4 raster is ~25 MB, so
            # converting a whole scanned document at once would exhaust memory.
            images = convert_from_bytes(
                data,
                first_page=page_number,
                last_page=page_number,
                **convert_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - Poppler failures are environmental
            logger.warning("Could not rasterise page %d for OCR: %s", page_number, exc)
            if position == 1:
                # Failing on the very first page means the toolchain is broken,
                # not that this one page is odd.
                raise OCRUnavailableError(
                    f"Could not convert PDF pages to images: {exc}\n\n{INSTALL_HINT}"
                ) from exc
            continue

        if not images:
            logger.warning("Page %d rasterised to no image; skipping", page_number)
            continue

        try:
            # Greyscale before recognition: Tesseract binarises internally and
            # feeding it one channel is both faster and slightly more accurate.
            text = pytesseract.image_to_string(
                images[0].convert("L"), lang=settings.ocr_language
            )
        except Exception as exc:  # noqa: BLE001 - bad language pack, corrupt raster
            logger.warning("OCR failed on page %d: %s", page_number, exc)
            continue
        finally:
            for image in images:
                image.close()

        if text and text.strip():
            results[page_number] = text
            logger.debug("OCR page %d recovered %d characters", page_number, len(text.strip()))
        else:
            logger.debug("OCR page %d produced no text", page_number)

        if progress_callback:
            progress_callback(position, len(targets))

    logger.info("OCR recovered text for %d of %d page(s)", len(results), len(targets))
    return results
