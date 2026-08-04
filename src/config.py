"""Centralised configuration.

Every secret and tunable lives here so that no other module ever touches
``os.environ`` directly. Values come from environment variables (loaded from a
local ``.env`` file when present), each with a sensible default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key, default)
    return value.strip() if value else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {key}={raw!r} is not a valid integer.") from exc


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Environment variable {key}={raw!r} is not a valid boolean.")


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {key}={raw!r} is not a valid number.") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the application configuration."""

    # --- Pinecone ---
    pinecone_api_key: str = field(default_factory=lambda: _env_str("PINECONE_API_KEY", ""))
    pinecone_index_name: str = field(default_factory=lambda: _env_str("PINECONE_INDEX_NAME", "rag-pdf-index"))
    pinecone_cloud: str = field(default_factory=lambda: _env_str("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: _env_str("PINECONE_REGION", "us-east-1"))
    pinecone_metric: str = "cosine"

    # --- LLM ---
    groq_api_key: str = field(default_factory=lambda: _env_str("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _env_str("GROQ_MODEL", "llama-3.3-70b-versatile"))

    # --- Embeddings ---
    # "onnx" runs all-MiniLM-L6-v2 through ONNX Runtime and produces vectors
    # identical to the torch backend for ~350 MB less resident memory, so it is
    # the default. "sentence-transformers" keeps the original torch path.
    embedding_backend: str = field(default_factory=lambda: _env_str("EMBEDDING_BACKEND", "onnx"))
    embedding_model: str = field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_dimension: int = field(default_factory=lambda: _env_int("EMBEDDING_DIMENSION", 384))
    # Chunks are encoded one batch at a time; only one batch of activations is
    # ever resident, so a small batch keeps peak memory flat on a 512 MB host.
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 16))
    # ONNX Runtime spawns a thread pool per session; one thread is plenty for
    # this model size and avoids per-thread arenas.
    onnx_threads: int = field(default_factory=lambda: _env_int("ONNX_THREADS", 1))

    # --- Retrieval defaults (the UI may override these per request) ---
    default_chunk_size: int = field(default_factory=lambda: _env_int("DEFAULT_CHUNK_SIZE", 800))
    default_chunk_overlap: int = field(default_factory=lambda: _env_int("DEFAULT_CHUNK_OVERLAP", 150))
    default_top_k: int = field(default_factory=lambda: _env_int("DEFAULT_TOP_K", 5))
    default_similarity_threshold: float = field(
        default_factory=lambda: _env_float("DEFAULT_SIMILARITY_THRESHOLD", 0.35)
    )

    # --- OCR fallback for scanned / image-only PDFs ---
    ocr_enabled: bool = field(default_factory=lambda: _env_bool("OCR_ENABLED", True))
    ocr_language: str = field(default_factory=lambda: _env_str("OCR_LANGUAGE", "eng"))
    ocr_dpi: int = field(default_factory=lambda: _env_int("OCR_DPI", 300))
    # A page yielding fewer than this many characters is treated as un-extractable
    # and handed to OCR.
    ocr_min_chars: int = field(default_factory=lambda: _env_int("OCR_MIN_CHARS", 80))
    # Upper bound on pages sent to OCR per document, so one huge scan cannot hang the UI.
    ocr_max_pages: int = field(default_factory=lambda: _env_int("OCR_MAX_PAGES", 50))
    # Optional explicit binary locations; empty means "discover automatically / use PATH".
    tesseract_cmd: str = field(default_factory=lambda: _env_str("TESSERACT_CMD", ""))
    poppler_path: str = field(default_factory=lambda: _env_str("POPPLER_PATH", ""))

    # --- Speech-to-text for spoken questions ---
    stt_enabled: bool = field(default_factory=lambda: _env_bool("STT_ENABLED", True))
    # Whisper checkpoint name; "base.en" balances accuracy against a ~75 MB download.
    stt_model: str = field(default_factory=lambda: _env_str("STT_MODEL", "base.en"))
    stt_device: str = field(default_factory=lambda: _env_str("STT_DEVICE", "cpu"))
    stt_compute_type: str = field(default_factory=lambda: _env_str("STT_COMPUTE_TYPE", "int8"))
    # Empty means "let Whisper detect the language".
    stt_language: str = field(default_factory=lambda: _env_str("STT_LANGUAGE", ""))
    stt_beam_size: int = field(default_factory=lambda: _env_int("STT_BEAM_SIZE", 5))
    stt_max_audio_mb: int = field(default_factory=lambda: _env_int("STT_MAX_AUDIO_MB", 25))
    stt_cpu_threads: int = field(default_factory=lambda: _env_int("STT_CPU_THREADS", 1))
    # Keeping the model resident makes repeat questions instant but holds ~90 MB.
    # Set false on a memory-constrained host to release it after each transcript.
    stt_keep_model_loaded: bool = field(
        default_factory=lambda: _env_bool("STT_KEEP_MODEL_LOADED", True)
    )

    # --- Limits ---
    max_pdf_size_mb: int = field(default_factory=lambda: _env_int("MAX_PDF_SIZE_MB", 20))

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

    @property
    def stt_max_audio_bytes(self) -> int:
        return self.stt_max_audio_mb * 1024 * 1024

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems (empty == OK)."""
        problems: list[str] = []

        if not self.pinecone_api_key:
            problems.append("PINECONE_API_KEY is not set. Add it to your .env file.")
        if not self.groq_api_key:
            problems.append("GROQ_API_KEY is not set. Add it to your .env file.")
        if self.embedding_backend not in {"onnx", "sentence-transformers", "pinecone"}:
            problems.append(
                f"EMBEDDING_BACKEND={self.embedding_backend!r} is invalid; "
                "use 'onnx', 'sentence-transformers' or 'pinecone'."
            )
        if self.embedding_dimension <= 0:
            problems.append("EMBEDDING_DIMENSION must be a positive integer.")
        if self.embedding_batch_size < 1:
            problems.append("EMBEDDING_BATCH_SIZE must be at least 1.")
        if self.onnx_threads < 1:
            problems.append("ONNX_THREADS must be at least 1.")
        if self.default_chunk_overlap >= self.default_chunk_size:
            problems.append("DEFAULT_CHUNK_OVERLAP must be smaller than DEFAULT_CHUNK_SIZE.")
        if not 0.0 <= self.default_similarity_threshold <= 1.0:
            problems.append("DEFAULT_SIMILARITY_THRESHOLD must be between 0.0 and 1.0.")
        # OCR is an optional enhancement: bad values are worth reporting, but a
        # missing Tesseract install must never block startup.
        if self.ocr_dpi < 72 or self.ocr_dpi > 600:
            problems.append("OCR_DPI must be between 72 and 600.")
        if self.ocr_min_chars < 0:
            problems.append("OCR_MIN_CHARS cannot be negative.")
        # Speech-to-text is optional in the same way: validate the numbers, but a
        # missing Whisper install must never block startup.
        if self.stt_beam_size < 1:
            problems.append("STT_BEAM_SIZE must be at least 1.")
        if self.stt_max_audio_mb < 1:
            problems.append("STT_MAX_AUDIO_MB must be at least 1.")

        return problems


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return the process-wide settings singleton."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
