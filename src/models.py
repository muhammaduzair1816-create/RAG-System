"""Plain data objects shared across the pipeline stages.

Keeping these in one module means the loader, chunker, retriever and generator
depend on a common vocabulary rather than on each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageText:
    """Cleaned text belonging to a single page of a source PDF."""

    page_number: int  # 1-indexed, as a human would cite it
    text: str
    # How the text was obtained: "native" (embedded PDF text), "ocr" (rasterised
    # and read by Tesseract), or "merged" (both sources combined).
    source: str = "native"


@dataclass
class Document:
    """A parsed PDF, ready for chunking."""

    document_name: str
    pages: list[PageText]
    total_pages: int
    # Pages whose text came wholly or partly from OCR, in reading order.
    ocr_pages: list[int] = field(default_factory=list)
    # True when OCR was attempted, whether or not it recovered any text.
    ocr_attempted: bool = False
    # Set when OCR was needed but could not run (e.g. Tesseract not installed).
    ocr_warning: str | None = None

    @property
    def character_count(self) -> int:
        return sum(len(page.text) for page in self.pages)

    @property
    def is_empty(self) -> bool:
        return self.character_count == 0

    @property
    def ocr_used(self) -> bool:
        return bool(self.ocr_pages)


@dataclass
class Chunk:
    """A retrievable unit of text plus the metadata stored alongside its vector."""

    chunk_id: str
    text: str
    document_name: str
    page_number: int
    chunk_index: int

    def to_metadata(self) -> dict[str, Any]:
        """Metadata payload persisted in Pinecone next to the embedding."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity search, with its score."""

    chunk_id: str
    text: str
    document_name: str
    page_number: int
    chunk_index: int
    score: float  # cosine similarity in [-1, 1]; in practice [0, 1] for these models

    @classmethod
    def from_match(cls, match: dict[str, Any]) -> "RetrievedChunk":
        metadata = match.get("metadata") or {}
        return cls(
            chunk_id=str(metadata.get("chunk_id", match.get("id", ""))),
            text=str(metadata.get("text", "")),
            document_name=str(metadata.get("document_name", "unknown")),
            page_number=int(float(metadata.get("page_number", 0))),
            chunk_index=int(float(metadata.get("chunk_index", 0))),
            score=float(match.get("score", 0.0)),
        )

    def excerpt(self, max_chars: int = 320) -> str:
        """Short preview used for source attribution in the UI."""
        flat = " ".join(self.text.split())
        if len(flat) <= max_chars:
            return flat
        return flat[:max_chars].rsplit(" ", 1)[0] + " …"


@dataclass
class Answer:
    """The final result handed back to the interface layer."""

    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = True  # False when the model had to refuse for lack of context
    model: str = ""
    latency_seconds: float = 0.0

    @property
    def confidence_label(self) -> str:
        if not self.grounded:
            return "No answer"
        if self.confidence >= 0.70:
            return "High"
        if self.confidence >= 0.50:
            return "Medium"
        return "Low"
