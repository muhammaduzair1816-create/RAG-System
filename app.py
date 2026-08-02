"""Streamlit interface for the Pinecone-backed RAG system.

This module owns presentation only: every piece of domain logic lives in
``src/`` and is reached through :class:`src.pipeline.RAGPipeline`.
"""

from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st

from src.chunker import ChunkingError
from src.config import get_settings
from src.embeddings import EmbeddingError
from src.generator import GenerationError
from src.models import Answer
from src.ocr import INSTALL_HINT, check_availability
from src.pdf_loader import PDFProcessingError
from src import stt
from src.pipeline import RAGPipeline
from src.query_logger import configure_logging, read_query_log
from src.retriever import RetrievalError, RetrievalResult
from src.vector_store import VectorStoreError, sanitize_namespace

st.set_page_config(page_title="RAG over PDFs · Pinecone", page_icon="📘", layout="wide")

configure_logging(logging.INFO)
SETTINGS = get_settings()


# ---------------------------------------------------------------------------
# Resources and session state
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model and connecting to Pinecone…")
def load_pipeline() -> RAGPipeline:
    """Build the pipeline once per server process."""
    return RAGPipeline()


# Widget/session keys for the question box and its speech input.
QUESTION_KEY = "question_text"
AUDIO_KEY = "question_audio"
STT_STATUS_KEY = "stt_status"          # (level, message) shown under the microphone
STT_LAST_AUDIO_KEY = "stt_last_audio"  # fingerprint of the last transcribed clip


def init_state() -> None:
    defaults = {
        "namespace": f"session-{datetime.now():%Y%m%d-%H%M%S}",
        "history": [],          # query history (session memory)
        "indexed": [],          # IngestionReport summaries for this session
        "last_answer": None,
        "last_result": None,
        QUESTION_KEY: "",
        STT_STATUS_KEY: None,
        STT_LAST_AUDIO_KEY: None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


# ---------------------------------------------------------------------------
# Sidebar: configuration and corpus management
# ---------------------------------------------------------------------------


def render_ocr_status() -> None:
    """Show whether scanned PDFs can be read, without blocking anything if not."""
    availability = check_availability(SETTINGS)

    st.sidebar.subheader("Scanned PDFs (OCR)")
    if availability.available:
        st.sidebar.success(f"OCR ready — Tesseract {availability.tesseract_version}")
        st.sidebar.caption(
            f"Pages with under {SETTINGS.ocr_min_chars} characters of extractable text are "
            f"read automatically at {SETTINGS.ocr_dpi} DPI (`{SETTINGS.ocr_language}`)."
        )
    else:
        st.sidebar.warning("OCR unavailable — scanned PDFs cannot be read")
        st.sidebar.caption(availability.reason)
        with st.sidebar.expander("How to enable OCR"):
            st.markdown(INSTALL_HINT)
        st.sidebar.caption("Text-based PDFs are unaffected and work normally.")


def render_sidebar() -> dict:
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.caption(
        f"**Index** `{SETTINGS.pinecone_index_name}` · **Embeddings** "
        f"`{SETTINGS.embedding_model.split('/')[-1]}` · **LLM** `{SETTINGS.groq_model}`"
    )

    namespace = st.sidebar.text_input(
        "Pinecone namespace",
        value=st.session_state["namespace"],
        help="Namespaces isolate corpora inside one index. Reuse a name to query a previous corpus.",
    )
    st.session_state["namespace"] = sanitize_namespace(namespace)
    if st.session_state["namespace"] != namespace:
        st.sidebar.caption(f"Normalised to `{st.session_state['namespace']}`")

    st.sidebar.subheader("Indexing")
    chunk_size = st.sidebar.slider(
        "Chunk size (characters)", 200, 2000, SETTINGS.default_chunk_size, step=100,
        help="Smaller chunks give sharper retrieval; larger chunks preserve more context.",
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk overlap (characters)", 0, 600, SETTINGS.default_chunk_overlap, step=25,
        help="Text carried across chunk boundaries so sentences are not cut mid-thought.",
    )
    if chunk_overlap >= chunk_size:
        st.sidebar.error("Overlap must be smaller than chunk size.")

    render_ocr_status()

    st.sidebar.subheader("Retrieval")
    top_k = st.sidebar.slider("Top-k chunks", 1, 15, SETTINGS.default_top_k)
    threshold = st.sidebar.slider(
        "Similarity threshold (cosine)", 0.0, 1.0, SETTINGS.default_similarity_threshold, step=0.05,
        help="Chunks scoring below this are discarded. Raise it to force stricter grounding.",
    )

    return {
        "namespace": st.session_state["namespace"],
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_answer(answer: Answer, result: RetrievalResult) -> None:
    """Show the answer, its confidence, and full source attribution."""
    if not answer.grounded:
        st.warning(answer.text)
        st.caption(
            f"{result.discarded_below_threshold} chunk(s) were retrieved but scored below the "
            f"{result.threshold:.2f} threshold. Lower the threshold or rephrase the question."
        )
        return

    st.markdown("#### Answer")
    st.markdown(answer.text)

    if result.overview_fallback:
        st.caption(
            "ℹ️ This reads as a question *about* the document rather than about its contents, "
            "which no similarity search can match. The answer was built from the opening "
            "sections of the indexed document(s) instead."
        )

    left, middle, right = st.columns(3)
    left.metric("Confidence", f"{answer.confidence:.0%}", answer.confidence_label)
    middle.metric("Top similarity", f"{result.best_score:.3f}")
    right.metric("Response time", f"{answer.latency_seconds:.2f} s")

    st.markdown("#### Sources")
    for position, source in enumerate(answer.sources, start=1):
        label = (
            f"[S{position}] {source.document_name} — page {source.page_number} "
            f"· similarity {source.score:.3f}"
        )
        with st.expander(label):
            st.progress(min(max(source.score, 0.0), 1.0), text=f"Cosine similarity {source.score:.3f}")
            st.markdown(f"> {source.excerpt(600)}")
            st.caption(f"Chunk ID `{source.chunk_id}`")


def render_speech_input(container) -> None:
    """Draw the microphone, transcribe new recordings into the question box.

    Runs *before* the question ``text_area`` is created so that a fresh
    transcript can be written into ``st.session_state`` — Streamlit only allows
    a widget's value to be set before that widget is instantiated.
    """
    availability = stt.check_availability(SETTINGS)

    container.markdown("**Ask by voice**")
    if not availability.available:
        container.caption(f"🎙️ Unavailable — {availability.reason}")
        with container.expander("How to enable voice input"):
            st.markdown(stt.INSTALL_HINT)
        return

    recording = container.audio_input(
        "Record your question",
        key=AUDIO_KEY,
        label_visibility="collapsed",
        help="Click to record, click again to stop. The transcript lands in the question box.",
    )

    if recording is not None:
        audio_bytes = recording.getvalue()
        fingerprint = stt.audio_fingerprint(audio_bytes)
        # The widget replays the same clip on every rerun; only transcribe new audio.
        if fingerprint != st.session_state.get(STT_LAST_AUDIO_KEY):
            st.session_state[STT_LAST_AUDIO_KEY] = fingerprint
            _transcribe_into_question(container, audio_bytes)

    _render_stt_status(container)


def _transcribe_into_question(container, audio_bytes: bytes) -> None:
    """Transcribe a recording and place the text in the question box."""
    try:
        with container.status("⏳ Processing… transcribing your question", expanded=False):
            transcription = stt.transcribe(audio_bytes, SETTINGS)
    except stt.EmptyRecordingError as exc:
        st.session_state[STT_STATUS_KEY] = ("warning", str(exc))
        return
    except stt.STTUnavailableError as exc:
        st.session_state[STT_STATUS_KEY] = ("error", f"{exc}\n\n{stt.INSTALL_HINT}")
        return
    except stt.STTError as exc:
        st.session_state[STT_STATUS_KEY] = ("error", str(exc))
        return

    if transcription.is_empty:
        st.session_state[STT_STATUS_KEY] = (
            "warning",
            "No speech was detected in that recording. Check your microphone level and try again.",
        )
        return

    st.session_state[QUESTION_KEY] = transcription.text
    st.session_state[STT_STATUS_KEY] = (
        "success",
        f"Transcribed — {transcription.summary}. Edit it below if needed.",
    )
    # Rerun so the question box is rebuilt with the transcript already in place.
    st.rerun()


def _render_stt_status(container) -> None:
    """Show the current speech-input state under the microphone."""
    status = st.session_state.get(STT_STATUS_KEY)
    if status is None:
        container.caption("🎙️ Ready — click to record, then click again to stop.")
        container.caption(stt.MIC_HELP)
        return

    level, message = status
    renderer = {
        "success": container.success,
        "warning": container.warning,
        "error": container.error,
    }.get(level, container.info)
    renderer(message)
    if level in {"warning", "error"}:
        container.caption(stt.MIC_HELP)


def record_history(question: str, answer: Answer, result: RetrievalResult, controls: dict) -> None:
    st.session_state["history"].insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "question": question,
            "answer": answer.text,
            "confidence": answer.confidence,
            "grounded": answer.grounded,
            "top_k": controls["top_k"],
            "threshold": controls["threshold"],
            "sources": [(source.document_name, source.page_number, source.score) for source in answer.sources],
        },
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def tab_upload(pipeline: RAGPipeline, controls: dict) -> None:
    st.subheader("1 · Upload and index PDFs")
    st.caption(
        f"Multiple files are supported. Maximum {SETTINGS.max_pdf_size_mb} MB per file. "
        "Scanned pages are detected and read with OCR automatically — nothing to switch on."
    )

    uploads = st.file_uploader(
        "Choose one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploads and st.button("Index documents", type="primary"):
        if controls["chunk_overlap"] >= controls["chunk_size"]:
            st.error("Fix the chunk overlap in the sidebar before indexing.")
            return

        progress = st.progress(0.0, text="Starting…")
        for number, upload in enumerate(uploads, start=1):
            prefix = f"[{number}/{len(uploads)}] "
            try:
                report = pipeline.ingest_pdf(
                    data=upload.getvalue(),
                    filename=upload.name,
                    namespace=controls["namespace"],
                    chunk_size=controls["chunk_size"],
                    chunk_overlap=controls["chunk_overlap"],
                    progress_callback=lambda stage, fraction, prefix=prefix: progress.progress(
                        min(fraction, 1.0), text=prefix + stage
                    ),
                )
            except PDFProcessingError as exc:
                st.error(f"**{upload.name}** — {exc}")
                continue
            except (ChunkingError, ValueError) as exc:
                st.error(f"**{upload.name}** — {exc}")
                continue
            except EmbeddingError as exc:
                st.error(f"Embedding failure on **{upload.name}**: {exc}")
                continue
            except VectorStoreError as exc:
                st.error(f"Pinecone failure on **{upload.name}**: {exc}")
                continue

            st.success(
                f"**{report.document_name}** — {report.vectors_upserted} vectors from "
                f"{report.chunks_created} chunks across {report.pages_with_text}/"
                f"{report.total_pages} pages → namespace `{report.namespace}`"
            )
            if report.ocr_used:
                st.info(f"🔍 {report.ocr_summary}")
            for warning in report.warnings:
                st.warning(warning)
            st.session_state["indexed"].append(report)

        progress.empty()
        st.cache_data.clear()

    if st.session_state["indexed"]:
        st.divider()
        st.markdown("**Indexed this session**")
        st.dataframe(
            [
                {
                    "Document": report.document_name,
                    "Pages": f"{report.pages_with_text}/{report.total_pages}",
                    "Chunks": report.chunks_created,
                    "Vectors": report.vectors_upserted,
                    "OCR pages": len(report.ocr_pages),
                    "Namespace": report.namespace,
                }
                for report in st.session_state["indexed"]
            ],
            use_container_width=True,
            hide_index=True,
        )


@st.cache_data(show_spinner=False, ttl=60)
def cached_documents(namespace: str, _pipeline: RAGPipeline) -> list[str]:
    try:
        return _pipeline.list_documents(namespace)
    except VectorStoreError:
        return []


def tab_ask(pipeline: RAGPipeline, controls: dict) -> None:
    st.subheader("2 · Ask a question")

    try:
        stats = pipeline.namespace_stats(controls["namespace"])
        vector_count = stats["vector_count"]
    except VectorStoreError as exc:
        st.error(f"Could not reach Pinecone: {exc}")
        return

    if vector_count == 0:
        st.info(
            f"Namespace `{controls['namespace']}` is empty. Upload a PDF on the **Upload** tab first. "
            "Newly indexed vectors can take a few seconds to become queryable."
        )

    available = cached_documents(controls["namespace"], pipeline)

    filter_column, page_column = st.columns([2, 1])
    selected_documents = filter_column.multiselect(
        "Restrict to documents (optional)", options=available, default=[],
        help="Metadata filter on `document_name`. Leave empty to search the whole namespace.",
    )
    use_page_filter = page_column.checkbox("Filter by page range")
    page_range = None
    if use_page_filter:
        low, high = page_column.columns(2)
        first = low.number_input("From page", min_value=1, value=1, step=1)
        last = high.number_input("To page", min_value=1, value=50, step=1)
        page_range = (int(first), int(last))

    question_column, mic_column = st.columns([3, 1])
    # The microphone runs first so a transcript can be injected into the
    # question box below, even though it is laid out beside it.
    render_speech_input(mic_column)
    question = question_column.text_area(
        "Your question",
        key=QUESTION_KEY,
        placeholder="e.g. What evaluation metrics does the paper report?",
        height=160,
    )

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Please type a question first.")
            return

        try:
            with st.spinner("Retrieving context and generating an answer…"):
                answer, result = pipeline.ask(
                    question=question,
                    namespace=controls["namespace"],
                    top_k=controls["top_k"],
                    similarity_threshold=controls["threshold"],
                    document_names=selected_documents or None,
                    page_range=page_range,
                )
        except RetrievalError as exc:
            st.warning(str(exc))
            return
        except EmbeddingError as exc:
            st.error(f"Embedding failure: {exc}")
            return
        except VectorStoreError as exc:
            st.error(f"Pinecone failure: {exc}")
            return
        except GenerationError as exc:
            st.error(f"Answer generation failed: {exc}")
            return

        st.session_state["last_answer"] = answer
        st.session_state["last_result"] = result
        record_history(question, answer, result, controls)

    if st.session_state["last_answer"] is not None:
        st.divider()
        render_answer(st.session_state["last_answer"], st.session_state["last_result"])


def tab_history() -> None:
    st.subheader("3 · Query history")
    history = st.session_state["history"]

    if not history:
        st.info("No questions asked yet in this session.")
        return

    if st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()

    for entry in history:
        badge = "✅" if entry["grounded"] else "🚫"
        with st.expander(f"{badge} {entry['time']} — {entry['question'][:80]}"):
            st.markdown(entry["answer"])
            st.caption(
                f"confidence {entry['confidence']:.0%} · top-k {entry['top_k']} · "
                f"threshold {entry['threshold']:.2f}"
            )
            if entry["sources"]:
                st.caption(
                    "Sources: "
                    + ", ".join(f"{name} p.{page} ({score:.2f})" for name, page, score in entry["sources"])
                )


def tab_logs(pipeline: RAGPipeline, controls: dict) -> None:
    st.subheader("4 · Corpus and logs")

    try:
        stats = pipeline.namespace_stats(controls["namespace"])
        st.metric(f"Vectors in namespace `{controls['namespace']}`", stats["vector_count"])
    except VectorStoreError as exc:
        st.error(f"Could not read index statistics: {exc}")

    documents = cached_documents(controls["namespace"], pipeline)
    if documents:
        st.markdown("**Documents in this namespace**")
        st.write("\n".join(f"- {name}" for name in documents))

    with st.expander("⚠️ Danger zone"):
        st.caption("Deletes every vector in the current namespace. This cannot be undone.")
        if st.checkbox("I understand this permanently deletes the indexed corpus"):
            if st.button("Delete all vectors in this namespace"):
                try:
                    pipeline.clear_namespace(controls["namespace"])
                    st.cache_data.clear()
                    st.session_state["indexed"] = []
                    st.success(f"Namespace `{controls['namespace']}` cleared.")
                except VectorStoreError as exc:
                    st.error(str(exc))

    st.divider()
    st.markdown("**Persistent query log** (`logs/queries.jsonl`)")
    records = read_query_log(limit=100)
    if not records:
        st.info("No queries logged yet.")
        return

    st.dataframe(
        [
            {
                "Time (UTC)": record["timestamp"].replace("T", " "),
                "Query": record["query"][:70],
                "Chunks": record["chunks_retrieved"],
                "Best score": record["best_score"],
                "Confidence": record["confidence"],
                "Grounded": record["grounded"],
                "Latency (s)": record["latency_seconds"],
            }
            for record in records
        ],
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("📘 RAG over PDFs — Pinecone Vector Database")
    st.caption(
        "Answers are generated strictly from the uploaded documents, with page-level "
        "source attribution and similarity scores."
    )

    problems = SETTINGS.validate()
    if problems:
        st.error("**Configuration incomplete**\n\n" + "\n".join(f"- {problem}" for problem in problems))
        st.info("Copy `.env.example` to `.env`, fill in your API keys, then restart the app.")
        st.stop()

    controls = render_sidebar()

    try:
        pipeline = load_pipeline()
    except (VectorStoreError, EmbeddingError, GenerationError) as exc:
        st.error(f"**Startup failed** — {exc}")
        st.stop()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    upload, ask, history, logs = st.tabs(["📤 Upload", "💬 Ask", "🕘 History", "📊 Corpus & Logs"])
    with upload:
        tab_upload(pipeline, controls)
    with ask:
        tab_ask(pipeline, controls)
    with history:
        tab_history()
    with logs:
        tab_logs(pipeline, controls)


if __name__ == "__main__":
    main()
