"""Download the models at image build time.

Render's free tier has an ephemeral filesystem: anything written at runtime is
lost when the instance spins down. Without this step the first request after
every cold start would re-download ~130 MB of weights, which is slow and looks
broken to a user.

Run as part of the Docker build, after requirements are installed. Reads the
same settings as the app, so changing STT_MODEL in the build environment fetches
the matching checkpoint.
"""

from __future__ import annotations

import os
import sys

# Import the app's own configuration so the build and the runtime always agree.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings  # noqa: E402

# Systran publishes the CTranslate2 conversions faster-whisper expects.
_WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def prefetch_embedding_model(repo: str) -> None:
    """Fetch the ONNX export and tokenizer used by the default embedder."""
    from huggingface_hub import hf_hub_download

    for filename in ("onnx/model.onnx", "tokenizer.json"):
        path = hf_hub_download(repo, filename)
        print(f"  cached {filename} -> {path}")


def prefetch_speech_model(model: str) -> None:
    """Fetch the Whisper checkpoint faster-whisper will load at runtime."""
    from huggingface_hub import snapshot_download

    repo = _WHISPER_REPOS.get(model)
    if repo is None:
        print(f"  '{model}' is not a known Whisper checkpoint; skipping prefetch")
        return

    # snapshot_download pulls the weights without instantiating the model, so
    # the build never pays the memory cost of loading it.
    path = snapshot_download(repo)
    print(f"  cached {repo} -> {path}")


def main() -> int:
    settings = get_settings()
    print(f"HF_HOME = {os.environ.get('HF_HOME', '(default)')}")

    failures: list[str] = []

    print(f"\nEmbedding model ({settings.embedding_backend}): {settings.embedding_model}")
    if settings.embedding_backend == "pinecone":
        print("  hosted backend — nothing to download")
    else:
        try:
            prefetch_embedding_model(settings.embedding_model)
        except Exception as exc:  # noqa: BLE001 - network or a repo without an ONNX export
            failures.append(f"embedding model: {exc}")
            print(f"  FAILED: {exc}")

    print(f"\nSpeech model: {settings.stt_model}")
    if not settings.stt_enabled:
        print("  speech input disabled — nothing to download")
    else:
        try:
            prefetch_speech_model(settings.stt_model)
        except Exception as exc:  # noqa: BLE001 - network failures
            failures.append(f"speech model: {exc}")
            print(f"  FAILED: {exc}")

    if failures:
        # Fail the build loudly: a missing model here becomes a slow, confusing
        # download on the user's first request instead.
        print("\nPrefetch failed:\n  - " + "\n  - ".join(failures))
        return 1

    print("\nAll models cached into the image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
