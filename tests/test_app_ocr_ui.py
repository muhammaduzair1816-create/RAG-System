"""UI tests for the OCR status panel, driven headlessly by Streamlit's AppTest.

Runs without a browser, so the sidebar is covered in CI as well as locally.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src import ocr as ocr_module
from src.ocr import OCRAvailability

AppTest = pytest.importorskip("streamlit.testing.v1", reason="Streamlit not installed").AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _script(available: bool) -> str:
    """A tiny app that renders only the OCR status panel, with a stubbed probe.

    The stub is applied *after* ``import app`` on purpose: pytest runs every
    AppTest in one process, so the second import of ``app`` is a cache hit and
    an import-time patch would leak the first test's stub into the second.
    """
    return textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})

        from src.ocr import OCRAvailability
        import app

        app.check_availability = lambda settings=None: OCRAvailability(
            available={available},
            tesseract_version="5.3.0",
            reason="" if {available} else "The Tesseract OCR program was not found on PATH.",
        )
        app.render_ocr_status()
        """
    )


def test_sidebar_reports_ocr_ready(tmp_path):
    script = tmp_path / "ready_app.py"
    script.write_text(_script(True), encoding="utf-8")

    at = AppTest.from_file(str(script), default_timeout=60).run()

    assert not at.exception
    assert any("OCR ready" in element.value for element in at.sidebar.success)
    assert "Tesseract 5.3.0" in at.sidebar.success[0].value
    assert not at.sidebar.warning


def test_sidebar_warns_and_explains_when_ocr_missing(tmp_path):
    script = tmp_path / "missing_app.py"
    script.write_text(_script(False), encoding="utf-8")

    at = AppTest.from_file(str(script), default_timeout=60).run()

    assert not at.exception
    # Warned, not errored: a missing OCR install must never look fatal.
    assert any("OCR unavailable" in element.value for element in at.sidebar.warning)
    assert not at.sidebar.error

    captions = " ".join(element.value for element in at.sidebar.caption)
    assert "Tesseract OCR program was not found" in captions
    assert "Text-based PDFs are unaffected" in captions


def test_install_guidance_is_offered_when_ocr_missing(tmp_path):
    script = tmp_path / "missing_app2.py"
    script.write_text(_script(False), encoding="utf-8")

    at = AppTest.from_file(str(script), default_timeout=60).run()

    markdown = " ".join(element.value for element in at.sidebar.markdown)
    assert "Tesseract OCR" in markdown
    assert "Poppler" in markdown
    assert "TESSERACT_CMD" in markdown


def test_availability_summary_is_always_renderable():
    ocr_module.reset_availability_cache()
    for availability in (
        OCRAvailability(available=True, tesseract_version="5.3.0"),
        OCRAvailability(available=False, reason="missing"),
        OCRAvailability(available=False),
    ):
        assert isinstance(availability.summary, str) and availability.summary
