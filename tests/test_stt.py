"""Unit tests for the speech-to-text module.

Whisper is never invoked: validation, backend selection and error handling are
pure Python, and the transcription call is stubbed where a test needs it.
"""

from __future__ import annotations

import pytest

from src import stt as stt_module
from src.config import Settings
from src.stt import (
    MIN_AUDIO_BYTES,
    EmptyRecordingError,
    STTAvailability,
    STTError,
    STTUnavailableError,
    Transcription,
    audio_fingerprint,
    transcribe,
    validate_audio,
)


def _audio(size: int = MIN_AUDIO_BYTES * 4) -> bytes:
    return b"RIFF" + b"\x00" * (size - 4)


@pytest.fixture
def available(monkeypatch):
    monkeypatch.setattr(
        stt_module,
        "check_availability",
        lambda settings=None: STTAvailability(
            available=True, backend="faster-whisper", model="base.en"
        ),
    )


# --- Recording validation --------------------------------------------------


def test_missing_audio_is_rejected():
    with pytest.raises(EmptyRecordingError, match="No audio"):
        validate_audio(b"", Settings())


def test_too_short_recording_is_rejected():
    with pytest.raises(EmptyRecordingError, match="too short"):
        validate_audio(b"\x00" * 100, Settings())


def test_recording_at_the_threshold_is_accepted():
    validate_audio(b"\x00" * MIN_AUDIO_BYTES, Settings())


def test_oversized_recording_is_rejected():
    oversized = b"\x00" * (2 * 1024 * 1024)
    with pytest.raises(STTError, match="above the 1 MB limit"):
        validate_audio(oversized, Settings(stt_max_audio_mb=1))


def test_max_audio_bytes_conversion():
    assert Settings(stt_max_audio_mb=3).stt_max_audio_bytes == 3 * 1024 * 1024


# --- Fingerprinting (stops the same clip transcribing twice) ---------------


def test_fingerprint_is_stable_and_distinct():
    assert audio_fingerprint(b"abc") == audio_fingerprint(b"abc")
    assert audio_fingerprint(b"abc") != audio_fingerprint(b"abd")
    assert len(audio_fingerprint(b"abc")) == 16


# --- Availability ----------------------------------------------------------

def test_availability_reports_disabled_state():
    stt_module.reset_availability_cache()
    availability = stt_module.check_availability(Settings(stt_enabled=False))
    assert not availability.available
    assert "disabled" in availability.reason.lower()


def test_availability_never_raises_and_always_summarises():
    stt_module.reset_availability_cache()
    availability = stt_module.check_availability(Settings())
    assert isinstance(availability.available, bool)
    assert availability.summary


def test_faster_whisper_is_preferred_when_present():
    stt_module.reset_availability_cache()
    availability = stt_module.check_availability(Settings())
    # faster-whisper is a declared dependency, so it must win the probe.
    assert availability.backend == "faster-whisper"


def test_install_hint_names_both_backends():
    assert "faster-whisper" in stt_module.INSTALL_HINT
    assert "openai-whisper" in stt_module.INSTALL_HINT


def test_mic_help_explains_permission_recovery():
    assert "microphone" in stt_module.MIC_HELP.lower()
    assert "reload" in stt_module.MIC_HELP.lower()


# --- Transcription result object -------------------------------------------


def test_transcription_reports_emptiness_and_word_count():
    assert Transcription(text="").is_empty
    assert Transcription(text="   ").is_empty
    spoken = Transcription(text="what is this document about")
    assert not spoken.is_empty
    assert spoken.word_count == 5


def test_transcription_summary_is_human_readable():
    summary = Transcription(
        text="two words", language="en", language_probability=0.99, duration_seconds=3.25
    ).summary
    assert "2 words" in summary
    assert "3.2s audio" in summary
    assert "language en" in summary


def test_transcription_summary_handles_single_word_and_missing_metadata():
    assert Transcription(text="hello").summary == "1 word"


# --- Transcription flow ----------------------------------------------------


def test_transcribe_delegates_to_faster_whisper(monkeypatch, available):
    captured = {}

    def fake(data, settings):
        captured["bytes"] = len(data)
        return Transcription(text="what is this document about", backend="faster-whisper")

    monkeypatch.setattr(stt_module, "_transcribe_faster_whisper", fake)

    result = transcribe(_audio(), Settings())

    assert result.text == "what is this document about"
    assert result.backend == "faster-whisper"
    assert captured["bytes"] > MIN_AUDIO_BYTES


def test_transcribe_uses_openai_whisper_when_that_is_the_backend(monkeypatch):
    monkeypatch.setattr(
        stt_module,
        "check_availability",
        lambda settings=None: STTAvailability(
            available=True, backend="openai-whisper", model="base.en"
        ),
    )
    monkeypatch.setattr(
        stt_module,
        "_transcribe_openai_whisper",
        lambda data, settings: Transcription(text="fallback path", backend="openai-whisper"),
    )

    assert transcribe(_audio(), Settings()).backend == "openai-whisper"


def test_transcribe_without_a_backend_raises_unavailable(monkeypatch):
    monkeypatch.setattr(
        stt_module,
        "check_availability",
        lambda settings=None: STTAvailability(available=False, reason="No Whisper backend."),
    )
    with pytest.raises(STTUnavailableError, match="No Whisper backend"):
        transcribe(_audio(), Settings())


def test_backend_failure_is_wrapped_in_stt_error(monkeypatch, available):
    def boom(data, settings):
        raise RuntimeError("ctranslate2 exploded")

    monkeypatch.setattr(stt_module, "_transcribe_faster_whisper", boom)

    with pytest.raises(STTError, match="Speech recognition failed: ctranslate2 exploded"):
        transcribe(_audio(), Settings())


def test_empty_recording_is_not_swallowed_by_the_generic_handler(monkeypatch, available):
    # Validation runs before the backend, so this must stay an EmptyRecordingError.
    with pytest.raises(EmptyRecordingError):
        transcribe(b"", Settings())


def test_silent_audio_transcribes_to_an_empty_string(monkeypatch, available):
    monkeypatch.setattr(
        stt_module,
        "_transcribe_faster_whisper",
        lambda data, settings: Transcription(text="", backend="faster-whisper"),
    )

    result = transcribe(_audio(), Settings())

    # Silence is not an error — the caller decides how to report it.
    assert result.is_empty
