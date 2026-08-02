"""UI tests for the speech-input panel, driven headlessly by Streamlit's AppTest.

Runs without a browser or a microphone, so the panel is covered in CI too.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1", reason="Streamlit not installed").AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _script(body: str) -> str:
    """Wrap a snippet in the preamble every speech-panel test needs.

    Two details matter, both caused by pytest running each AppTest in one
    process. ``app`` is only executed on its first import, so ``init_state()``
    is called explicitly here — the real entry script reruns it every time.
    And stubs must go through ``patch.object`` rather than plain assignment,
    because ``app.stt`` *is* the shared :mod:`src.stt` module and a bare
    assignment would leak into every later test.
    """
    return textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})

        from unittest.mock import patch
        from src import stt as stt_module
        from src.stt import EmptyRecordingError, STTAvailability, STTError
        from src.stt import STTUnavailableError, Transcription
        import app

        app.init_state()

        {textwrap.indent(textwrap.dedent(body), " " * 8).strip()}
        """
    )


def _run(tmp_path, name: str, body: str):
    script = tmp_path / f"{name}.py"
    script.write_text(_script(body), encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=60).run()


def _available(backend: str = "faster-whisper") -> str:
    return (
        f'patch.object(stt_module, "check_availability", lambda settings=None: '
        f'STTAvailability(available=True, backend="{backend}", model="base.en"))'
    )


def _unavailable(reason: str = "No Whisper backend is installed.") -> str:
    return (
        f'patch.object(stt_module, "check_availability", lambda settings=None: '
        f'STTAvailability(available=False, reason="{reason}"))'
    )


# --- Backend unavailable ---------------------------------------------------


def test_panel_explains_how_to_enable_voice_when_no_backend(tmp_path):
    at = _run(
        tmp_path,
        "unavailable",
        f"""
        with {_unavailable()}:
            app.render_speech_input(app.st.container())
        """,
    )

    assert not at.exception
    captions = " ".join(element.value for element in at.caption)
    assert "Unavailable" in captions
    assert "No Whisper backend is installed." in captions

    markdown = " ".join(element.value for element in at.markdown)
    assert "faster-whisper" in markdown  # install guidance is offered


def test_unavailable_backend_is_not_shown_as_an_error(tmp_path):
    """Missing speech input must never look fatal — typing still works."""
    at = _run(
        tmp_path,
        "unavailable_not_error",
        f"""
        with {_unavailable()}:
            app.render_speech_input(app.st.container())
        """,
    )

    assert not at.error
    assert not at.exception


# --- Ready state -----------------------------------------------------------


def test_ready_state_shows_recording_instructions(tmp_path):
    at = _run(
        tmp_path,
        "ready",
        f"""
        with {_available()}:
            app.render_speech_input(app.st.container())
        """,
    )

    assert not at.exception
    captions = " ".join(element.value for element in at.caption)
    assert "Ready" in captions
    assert "click to record" in captions.lower()
    # Permission recovery guidance is offered up front, not only after a failure.
    assert "address bar" in captions.lower()


def test_ready_state_renders_the_microphone_widget(tmp_path):
    at = _run(
        tmp_path,
        "ready_widget",
        f"""
        with {_available()}:
            app.render_speech_input(app.st.container())
        """,
    )

    assert not at.exception
    assert len(at.get("audio_input")) == 1  # the recorder is a real widget in the tree


# --- Status rendering ------------------------------------------------------


def test_success_status_is_rendered(tmp_path):
    at = _run(
        tmp_path,
        "status_success",
        """
        app.st.session_state[app.STT_STATUS_KEY] = ("success", "Transcribed - 5 words")
        app._render_stt_status(app.st.container())
        """,
    )

    assert not at.exception
    assert any("Transcribed" in element.value for element in at.success)


def test_warning_status_adds_microphone_guidance(tmp_path):
    at = _run(
        tmp_path,
        "status_warning",
        """
        app.st.session_state[app.STT_STATUS_KEY] = ("warning", "No speech was detected.")
        app._render_stt_status(app.st.container())
        """,
    )

    assert not at.exception
    assert any("No speech was detected." in element.value for element in at.warning)
    assert "microphone" in " ".join(e.value for e in at.caption).lower()


def test_error_status_is_rendered(tmp_path):
    at = _run(
        tmp_path,
        "status_error",
        """
        app.st.session_state[app.STT_STATUS_KEY] = ("error", "Speech recognition failed: boom")
        app._render_stt_status(app.st.container())
        """,
    )

    assert not at.exception
    assert any("boom" in element.value for element in at.error)


# --- Transcript reaches the question box -----------------------------------


def test_transcript_is_written_into_the_question_state(tmp_path):
    at = _run(
        tmp_path,
        "transcribe_ok",
        """
        stub = lambda data, settings=None: Transcription(
            text="what evaluation metrics does the paper report",
            language="en", language_probability=0.98, duration_seconds=2.5,
        )
        from streamlit.runtime.scriptrunner_utils.exceptions import RerunException

        with patch.object(stt_module, "transcribe", stub):
            try:
                app._transcribe_into_question(app.st.container(), b"x" * 4096)
            except RerunException:
                pass  # the helper reruns so the text area picks up the transcript
        app.st.write("QUESTION::" + app.st.session_state[app.QUESTION_KEY])
        """,
    )

    assert not at.exception
    written = " ".join(element.value for element in at.markdown)
    assert "QUESTION::what evaluation metrics does the paper report" in written


def test_empty_transcript_warns_and_leaves_the_question_untouched(tmp_path):
    at = _run(
        tmp_path,
        "transcribe_silent",
        """
        app.st.session_state[app.QUESTION_KEY] = "typed by hand"
        stub = lambda data, settings=None: Transcription(text="   ")
        with patch.object(stt_module, "transcribe", stub):
            app._transcribe_into_question(app.st.container(), b"x" * 4096)
        app._render_stt_status(app.st.container())
        app.st.write("QUESTION::" + app.st.session_state[app.QUESTION_KEY])
        """,
    )

    assert not at.exception
    assert any("No speech was detected" in element.value for element in at.warning)
    assert "QUESTION::typed by hand" in " ".join(e.value for e in at.markdown)


def test_failed_transcription_preserves_the_typed_question(tmp_path):
    at = _run(
        tmp_path,
        "transcribe_fails",
        """
        app.st.session_state[app.QUESTION_KEY] = "typed by hand"

        def boom(data, settings=None):
            raise STTError("Speech recognition failed: decoder error")

        with patch.object(stt_module, "transcribe", boom):
            app._transcribe_into_question(app.st.container(), b"x" * 4096)
        app._render_stt_status(app.st.container())
        app.st.write("QUESTION::" + app.st.session_state[app.QUESTION_KEY])
        """,
    )

    assert not at.exception
    assert any("decoder error" in element.value for element in at.error)
    assert "QUESTION::typed by hand" in " ".join(e.value for e in at.markdown)


def test_empty_recording_is_reported_as_a_warning(tmp_path):
    at = _run(
        tmp_path,
        "transcribe_empty_recording",
        """
        def boom(data, settings=None):
            raise EmptyRecordingError("That recording was too short to contain speech.")

        with patch.object(stt_module, "transcribe", boom):
            app._transcribe_into_question(app.st.container(), b"x" * 4096)
        app._render_stt_status(app.st.container())
        """,
    )

    assert not at.exception
    assert any("too short" in element.value for element in at.warning)
    assert not at.error  # an empty clip is user error, not a failure


def test_missing_backend_at_transcribe_time_offers_install_hint(tmp_path):
    at = _run(
        tmp_path,
        "transcribe_unavailable",
        """
        def boom(data, settings=None):
            raise STTUnavailableError("No Whisper backend is installed.")

        with patch.object(stt_module, "transcribe", boom):
            app._transcribe_into_question(app.st.container(), b"x" * 4096)
        app._render_stt_status(app.st.container())
        """,
    )

    assert not at.exception
    errors = " ".join(element.value for element in at.error)
    assert "No Whisper backend is installed." in errors
    assert "pip install faster-whisper" in errors
