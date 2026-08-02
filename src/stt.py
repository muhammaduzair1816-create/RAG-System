"""Optional speech-to-text stage for spoken questions.

Turns a browser recording into text for the Ask tab's question box. Nothing
downstream is aware of it: the transcript is ordinary text that the user can
edit before submitting, and the retrieval and generation path is unchanged.

Two backends are supported, tried in this order:

``faster-whisper``
    Preferred. A CTranslate2 reimplementation of Whisper — several times faster
    than the reference model on CPU, with int8 quantisation and no ffmpeg
    dependency for WAV input.

``openai-whisper``
    Fallback for environments where CTranslate2 has no wheel. Needs the
    ``ffmpeg`` binary on PATH.

Like :mod:`src.ocr`, every entry point degrades to an actionable message rather
than raising at import time: speech input is an enhancement, never a hard
dependency.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class STTError(RuntimeError):
    """Raised when transcription was attempted and failed."""


class STTUnavailableError(STTError):
    """Raised when no speech-to-text backend is installed."""


class EmptyRecordingError(STTError):
    """Raised when the recording is too short or too quiet to contain speech."""


# A WAV header alone is ~44 bytes; anything this small cannot hold speech.
MIN_AUDIO_BYTES = 2048

INSTALL_HINT = (
    "Speech-to-text needs a Whisper backend:\n"
    "  • Preferred — `pip install faster-whisper` (CPU-friendly, no ffmpeg needed)\n"
    "  • Fallback  — `pip install openai-whisper`, which also needs the ffmpeg binary\n"
    "Both are already listed in requirements.txt; run `pip install -r requirements.txt`.\n"
    "The model itself (~75 MB for base.en) downloads automatically on first use."
)

MIC_HELP = (
    "If the microphone button does nothing, the browser has blocked recording. "
    "Click the padlock (or camera/mic icon) in the address bar, allow microphone "
    "access for this site, then reload the page. Browsers only permit recording "
    "on `localhost` or over HTTPS."
)


@dataclass(frozen=True)
class STTAvailability:
    """Result of probing the local speech-to-text backends."""

    available: bool
    backend: str = ""
    model: str = ""
    reason: str = ""

    @property
    def summary(self) -> str:
        if self.available:
            return f"{self.backend} ready ({self.model})"
        return self.reason or "Speech-to-text unavailable"


@dataclass(frozen=True)
class Transcription:
    """Recognised speech plus what the backend reported about it."""

    text: str
    language: str = ""
    language_probability: float = 0.0
    duration_seconds: float = 0.0
    backend: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def summary(self) -> str:
        """One-line description for the UI status area."""
        parts = [f"{self.word_count} word{'s' if self.word_count != 1 else ''}"]
        if self.duration_seconds:
            parts.append(f"{self.duration_seconds:.1f}s audio")
        if self.language:
            confidence = (
                f" {self.language_probability:.0%}" if self.language_probability else ""
            )
            parts.append(f"language {self.language}{confidence}")
        return " · ".join(parts)


# --- Backend discovery ----------------------------------------------------


@lru_cache(maxsize=4)
def _probe(model: str, enabled: bool) -> STTAvailability:
    """Cached probe. Keyed on settings so a config change re-probes."""
    if not enabled:
        return STTAvailability(available=False, reason="Speech input is disabled (STT_ENABLED=false).")

    try:
        import faster_whisper  # noqa: F401 - import is the availability check

        return STTAvailability(available=True, backend="faster-whisper", model=model)
    except ImportError:
        logger.info("faster-whisper not installed; trying openai-whisper")

    try:
        import whisper  # noqa: F401 - import is the availability check

        return STTAvailability(available=True, backend="openai-whisper", model=model)
    except ImportError:
        pass

    return STTAvailability(
        available=False,
        reason="No Whisper backend is installed (tried faster-whisper, then openai-whisper).",
    )


def check_availability(settings: Settings | None = None) -> STTAvailability:
    """Probe the speech-to-text backends. Never raises — inspect the result."""
    settings = settings or get_settings()
    return _probe(settings.stt_model, settings.stt_enabled)


def reset_availability_cache() -> None:
    """Forget the cached probe (used by tests and after a config change)."""
    _probe.cache_clear()


# --- Model loading ---------------------------------------------------------


@lru_cache(maxsize=2)
def _load_faster_whisper(model: str, device: str, compute_type: str):
    """Load and cache a faster-whisper model; the first call downloads it."""
    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper model %r (device=%s, compute=%s)", model, device, compute_type)
    return WhisperModel(model, device=device, compute_type=compute_type)


@lru_cache(maxsize=2)
def _load_openai_whisper(model: str):
    """Load and cache a reference-implementation Whisper model."""
    import whisper

    logger.info("Loading openai-whisper model %r", model)
    return whisper.load_model(model)


# --- Validation ------------------------------------------------------------


def audio_fingerprint(data: bytes) -> str:
    """Stable ID for a recording, used to avoid re-transcribing the same audio."""
    return hashlib.sha1(data).hexdigest()[:16]


def validate_audio(data: bytes, settings: Settings | None = None) -> None:
    """Raise if ``data`` cannot plausibly be a usable recording.

    Raises:
        EmptyRecordingError: nothing was recorded, or it was far too short.
        STTError: the recording is larger than the configured limit.
    """
    settings = settings or get_settings()

    if not data:
        raise EmptyRecordingError("No audio was recorded. Hold the microphone button and speak.")
    if len(data) < MIN_AUDIO_BYTES:
        raise EmptyRecordingError(
            "That recording was too short to contain speech. Try again and speak for "
            "at least a second."
        )
    if len(data) > settings.stt_max_audio_bytes:
        size_mb = len(data) / (1024 * 1024)
        raise STTError(
            f"The recording is {size_mb:.1f} MB, above the {settings.stt_max_audio_mb} MB limit. "
            "Record a shorter question."
        )


# --- Transcription ---------------------------------------------------------


def _transcribe_faster_whisper(data: bytes, settings: Settings) -> Transcription:
    model = _load_faster_whisper(settings.stt_model, settings.stt_device, settings.stt_compute_type)

    # faster-whisper decodes file-like input directly, so no temp file is needed.
    segments, info = model.transcribe(
        io.BytesIO(data),
        beam_size=settings.stt_beam_size,
        language=settings.stt_language or None,
        # Voice-activity filtering trims silence, which both speeds up decoding
        # and stops Whisper hallucinating text over near-silent audio.
        vad_filter=True,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()

    return Transcription(
        text=text,
        language=getattr(info, "language", "") or "",
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration_seconds=float(getattr(info, "duration", 0.0) or 0.0),
        backend="faster-whisper",
    )


def _transcribe_openai_whisper(data: bytes, settings: Settings) -> Transcription:
    model = _load_openai_whisper(settings.stt_model)

    # The reference implementation reads from a path via ffmpeg, so the recording
    # has to land on disk first.
    handle, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(handle, "wb") as audio_file:
            audio_file.write(data)
        result = model.transcribe(path, language=settings.stt_language or None)
    finally:
        try:
            os.unlink(path)
        except OSError:  # pragma: no cover - the temp file may already be gone
            logger.debug("Could not remove temporary audio file %s", path)

    return Transcription(
        text=str(result.get("text", "")).strip(),
        language=str(result.get("language", "") or ""),
        backend="openai-whisper",
    )


def transcribe(data: bytes, settings: Settings | None = None) -> Transcription:
    """Convert recorded audio into text.

    Args:
        data: raw bytes of the browser recording (WAV from ``st.audio_input``).
        settings: overrides the process-wide configuration, for tests.

    Returns:
        A :class:`Transcription`. Callers should check ``is_empty`` — silent
        audio transcribes successfully to an empty string rather than failing.

    Raises:
        EmptyRecordingError: the recording was absent or too short.
        STTUnavailableError: no Whisper backend is installed.
        STTError: the backend failed while decoding or transcribing.
    """
    settings = settings or get_settings()
    validate_audio(data, settings)

    availability = check_availability(settings)
    if not availability.available:
        raise STTUnavailableError(availability.reason)

    logger.info(
        "Transcribing %.1f KB of audio with %s (%s)",
        len(data) / 1024,
        availability.backend,
        settings.stt_model,
    )

    try:
        if availability.backend == "faster-whisper":
            transcription = _transcribe_faster_whisper(data, settings)
        else:
            transcription = _transcribe_openai_whisper(data, settings)
    except (EmptyRecordingError, STTUnavailableError):
        raise
    except Exception as exc:  # noqa: BLE001 - decoder, model download and runtime errors
        raise STTError(f"Speech recognition failed: {exc}") from exc

    logger.info("Transcribed %d word(s): %r", transcription.word_count, transcription.text[:80])
    return transcription
