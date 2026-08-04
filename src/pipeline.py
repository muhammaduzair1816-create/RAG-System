"""Orchestration layer.

:class:`RAGPipeline` wires the stages together and is the single entry point the
interface talks to. No Streamlit import appears anywhere below this line, so the
same pipeline can be driven from a CLI, a FastAPI route, or a test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .chunker import chunk_document
from .config import Settings, get_settings
from .embeddings import BaseEmbedder, EmbeddingError, get_embedder
from .generator import AnswerGenerator
from .models import Answer, Document
from .pdf_loader import load_pdf
from .query_logger import log_query
from .retriever import Retriever, RetrievalResult
from .vector_store import PineconeVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    """Outcome of indexing one PDF."""

    document_name: str
    pages_with_text: int
    total_pages: int
    chunks_created: int
    vectors_upserted: int
    namespace: str
    warnings: list[str] = field(default_factory=list)
    # Pages whose text came wholly or partly from OCR, in reading order.
    ocr_pages: list[int] = field(default_factory=list)

    @property
    def ocr_used(self) -> bool:
        return bool(self.ocr_pages)

    @property
    def ocr_summary(self) -> str:
        """One-line description of the OCR contribution, for the UI."""
        if not self.ocr_pages:
            return "No OCR needed — text extracted directly."
        pages = ", ".join(str(number) for number in self.ocr_pages[:12])
        if len(self.ocr_pages) > 12:
            pages += f", … (+{len(self.ocr_pages) - 12} more)"
        noun = "page" if len(self.ocr_pages) == 1 else "pages"
        return f"OCR used on {len(self.ocr_pages)} {noun} of {self.total_pages}: {pages}"


class RAGPipeline:
    """PDF ➜ chunks ➜ embeddings ➜ Pinecone ➜ retrieval ➜ grounded answer."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        problems = self.settings.validate()
        if problems:
            raise RuntimeError("Configuration problems:\n  - " + "\n  - ".join(problems))

        # Nothing heavy is built here. Constructing the pipeline must stay cheap
        # because the interface builds it during the first page render, long
        # before anyone uploads a document or asks a question. Each component
        # below is created on first use and then cached for the process.
        self._embedder: BaseEmbedder | None = None
        self._vector_store: PineconeVectorStore | None = None
        self._retriever: Retriever | None = None
        self._generator: AnswerGenerator | None = None

        logger.info(
            "Pipeline configured (embedder=%s/%s, index=%s, llm=%s) — components load on demand",
            self.settings.embedding_backend,
            self.settings.embedding_model,
            self.settings.pinecone_index_name,
            self.settings.groq_model,
        )

    # --- Lazily constructed components ------------------------------------

    @property
    def embedder(self) -> BaseEmbedder:
        """The embedding model, loaded on first use."""
        if self._embedder is None:
            self._embedder = get_embedder(self.settings)
            if self._embedder.dimension != self.settings.embedding_dimension:
                raise EmbeddingError(
                    f"'{self._embedder.model_name}' produces {self._embedder.dimension}-d vectors "
                    f"but EMBEDDING_DIMENSION is {self.settings.embedding_dimension}. Set "
                    f"EMBEDDING_DIMENSION={self._embedder.dimension} in your .env and use an index "
                    "of that dimension."
                )
            logger.info(
                "Embedder ready (%s, %d-d)", self._embedder.model_name, self._embedder.dimension
            )
        return self._embedder

    @property
    def vector_store(self) -> PineconeVectorStore:
        """The Pinecone client, connected on first use.

        Uses the configured dimension rather than the embedder's, so that
        reading index stats never drags the embedding model into memory.
        """
        if self._vector_store is None:
            self._vector_store = PineconeVectorStore(
                self.settings, dimension=self.settings.embedding_dimension
            )
        return self._vector_store

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.vector_store, self.embedder)
        return self._retriever

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = AnswerGenerator(self.settings)
        return self._generator

    # --- Ingestion --------------------------------------------------------

    def ingest_pdf(
        self,
        data: bytes,
        filename: str,
        namespace: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> IngestionReport:
        """Run the full indexing path for a single PDF.

        ``progress_callback(stage_label, fraction_complete)`` is invoked as the
        stages advance so the UI can render a progress bar.
        """
        chunk_size = chunk_size or self.settings.default_chunk_size
        chunk_overlap = chunk_overlap if chunk_overlap is not None else self.settings.default_chunk_overlap

        def report(stage: str, fraction: float) -> None:
            if progress_callback:
                progress_callback(stage, fraction)

        report(f"Extracting text from {filename}", 0.10)
        document: Document = load_pdf(
            data,
            filename,
            self.settings,
            # OCR is by far the slowest stage, so give it its own progress span.
            ocr_progress_callback=lambda done, total: report(
                f"Running OCR on scanned pages ({done}/{total})", 0.10 + 0.15 * done / max(total, 1)
            ),
        )

        report("Chunking text", 0.30)
        chunks = chunk_document(document, chunk_size, chunk_overlap, namespace_salt=namespace)
        if not chunks:
            raise ValueError(
                f"'{filename}' produced no usable chunks. Try lowering the chunk size."
            )

        report(f"Generating {len(chunks)} embeddings", 0.45)
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])

        report("Upserting vectors into Pinecone", 0.75)
        upserted = self.vector_store.upsert_chunks(
            chunks,
            vectors,
            namespace=namespace,
            progress_callback=lambda done, total: report(
                f"Upserting vectors into Pinecone ({done}/{total})", 0.75 + 0.25 * done / max(total, 1)
            ),
        )

        report("Done", 1.0)

        warnings: list[str] = []
        skipped_pages = document.total_pages - len(document.pages)
        if skipped_pages > 0:
            warnings.append(
                f"{skipped_pages} of {document.total_pages} pages contained no extractable text "
                "(likely images or scans) and were skipped."
            )
        if document.ocr_warning:
            warnings.append(f"OCR was not applied: {document.ocr_warning}")

        logger.info(
            "Indexed %s: %d chunks -> namespace %r (OCR pages: %s)",
            filename,
            upserted,
            namespace,
            document.ocr_pages or "none",
        )

        return IngestionReport(
            document_name=document.document_name,
            pages_with_text=len(document.pages),
            total_pages=document.total_pages,
            chunks_created=len(chunks),
            vectors_upserted=upserted,
            namespace=namespace,
            warnings=warnings,
            ocr_pages=document.ocr_pages,
        )

    # --- Question answering ----------------------------------------------

    def ask(
        self,
        question: str,
        namespace: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        document_names: Iterable[str] | None = None,
        page_range: tuple[int, int] | None = None,
    ) -> tuple[Answer, RetrievalResult]:
        """Retrieve context and generate a grounded answer, logging the exchange."""
        top_k = top_k or self.settings.default_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.settings.default_similarity_threshold
        )

        result = self.retriever.retrieve(
            query=question,
            namespace=namespace,
            top_k=top_k,
            similarity_threshold=threshold,
            document_names=document_names,
            page_range=page_range,
        )
        answer = self.generator.generate(result)

        log_query(result, answer, namespace=namespace, extra={"embedding_model": self.embedder.model_name})
        return answer, result

    # --- Corpus management ------------------------------------------------

    def list_documents(self, namespace: str) -> list[str]:
        return self.vector_store.list_documents(namespace)

    def namespace_stats(self, namespace: str) -> dict:
        return self.vector_store.describe_namespace(namespace)

    def clear_namespace(self, namespace: str) -> None:
        self.vector_store.delete_namespace(namespace)
