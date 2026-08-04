"""Guards for the memory behaviour that keeps the app inside a 512 MB host.

These are structural assertions, not benchmarks: they pin *when* heavy modules
are imported and *when* models are constructed. A regression here is what caused
the original out-of-memory crash, and it is invisible to every other test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.config import Settings
from src.pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Importing any of these costs 70-460 MB, so none may load before it is needed.
HEAVY_MODULES = ("torch", "sentence_transformers", "transformers", "faster_whisper",
                 "ctranslate2", "pytesseract", "pdf2image")


# Appended at column 0 to whatever the test runs; prints any heavy module that
# ended up in sys.modules.
REPORT = (
    f"for _name in {HEAVY_MODULES!r}:\n"
    "    if _name in sys.modules:\n"
    "        print(_name)\n"
)


def _run_in_subprocess(body: str) -> str:
    """Execute a snippet in a clean interpreter and return its stdout."""
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        + textwrap.dedent(body).strip()
        + "\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    assert completed.returncode == 0, f"subprocess failed:\n{completed.stderr}"
    return completed.stdout


def _run_probe(body: str) -> list[str]:
    """Run ``body``, then report which heavy modules it caused to be imported."""
    stdout = _run_in_subprocess(textwrap.dedent(body).strip() + "\n" + REPORT)
    return [name for name in stdout.strip().splitlines() if name]


# --- Import-time behaviour -------------------------------------------------


def test_importing_src_modules_loads_nothing_heavy():
    """Importing the package must not drag in torch, whisper or the OCR wrappers."""
    assert _run_probe(
        """
        from src.config import get_settings
        from src.pipeline import RAGPipeline
        from src import ocr, stt, embeddings, pdf_loader, retriever, generator
        """
    ) == []


def test_constructing_the_pipeline_loads_nothing_heavy():
    """The interface builds the pipeline during the first render, before any work."""
    assert _run_probe(
        """
        import os
        os.environ.setdefault("PINECONE_API_KEY", "probe")
        os.environ.setdefault("GROQ_API_KEY", "probe")
        from src.config import get_settings
        from src.pipeline import RAGPipeline
        RAGPipeline(get_settings())
        """
    ) == []


def test_availability_probes_load_nothing_heavy():
    """The OCR and STT probes run on every page render, so they must stay cheap."""
    assert _run_probe(
        """
        from src.config import get_settings
        from src import ocr, stt
        settings = get_settings()
        ocr.check_availability(settings)
        stt.check_availability(settings)
        """
    ) == []


def test_probes_still_report_correctly_without_importing():
    """Cheap probing must not cost accuracy — faster-whisper is installed here."""
    stdout = _run_in_subprocess(
        """
        from src.config import get_settings
        from src import stt
        a = stt.check_availability(get_settings())
        print(f"{a.available}|{a.backend}")
        """
    )
    available, backend = stdout.strip().split("|")
    assert available == "True"
    assert backend == "faster-whisper"


# --- Lazy component construction -------------------------------------------


def _pipeline() -> RAGPipeline:
    return RAGPipeline(Settings(pinecone_api_key="probe", groq_api_key="probe"))


def test_pipeline_starts_with_no_components_built():
    pipeline = _pipeline()
    assert pipeline._embedder is None
    assert pipeline._vector_store is None
    assert pipeline._retriever is None
    assert pipeline._generator is None


def test_embedder_is_cached_after_first_access(monkeypatch):
    calls = []

    class FakeEmbedder:
        model_name, dimension = "fake", 384

    import src.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "get_embedder", lambda settings: (calls.append(1), FakeEmbedder())[1]
    )

    pipeline = _pipeline()
    first, second = pipeline.embedder, pipeline.embedder

    assert first is second
    assert len(calls) == 1  # loaded once, then reused


def test_embedder_dimension_mismatch_is_reported_clearly(monkeypatch):
    class WrongDimension:
        model_name, dimension = "wrong", 1024

    import src.pipeline as pipeline_module
    from src.embeddings import EmbeddingError

    monkeypatch.setattr(pipeline_module, "get_embedder", lambda settings: WrongDimension())

    with pytest.raises(EmbeddingError, match="EMBEDDING_DIMENSION=1024"):
        _ = _pipeline().embedder


def test_reading_index_stats_does_not_build_the_embedder(monkeypatch):
    """Rendering the Ask tab reads stats; that must not load the model."""
    import src.pipeline as pipeline_module

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def describe_namespace(self, namespace):
            return {"namespace": namespace, "vector_count": 0}

    monkeypatch.setattr(pipeline_module, "PineconeVectorStore", FakeStore)
    monkeypatch.setattr(
        pipeline_module, "get_embedder", lambda settings: pytest.fail("embedder was built")
    )

    pipeline = _pipeline()
    assert pipeline.namespace_stats("ns")["vector_count"] == 0
    assert pipeline._embedder is None


# --- Batching --------------------------------------------------------------


def test_embed_documents_defaults_to_the_configured_batch_size():
    from src.embeddings import BaseEmbedder
    import numpy as np

    seen: list[int] = []

    class Recorder(BaseEmbedder):
        model_name, dimension = "recorder", 4

        def _encode(self, texts, input_type):
            seen.append(len(texts))
            return np.ones((len(texts), 4), dtype=np.float32)

    Recorder().embed_documents(["text"] * 40, batch_size=16)

    assert seen == [16, 16, 8]  # one batch resident at a time, never all 40


def test_embed_documents_handles_an_empty_corpus():
    from src.embeddings import BaseEmbedder

    class Recorder(BaseEmbedder):
        model_name, dimension = "recorder", 4

        def _encode(self, texts, input_type):  # pragma: no cover - must not run
            raise AssertionError("should not encode an empty list")

    assert Recorder().embed_documents([]).shape == (0, 4)
