"""Stage 3: intelligent text chunking.

Chunking is page-scoped so that every chunk keeps an unambiguous page number for
source attribution. Within a page the splitter is recursive: it tries the
largest natural boundary first (paragraph → sentence → word) and only falls back
to a hard character cut when a single token is longer than the window.
"""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document

# Ordered from coarsest to finest natural boundary.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkingError(ValueError):
    """Raised when chunking parameters are inconsistent."""


def _validate_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size < 100:
        raise ChunkingError("chunk_size must be at least 100 characters.")
    if chunk_overlap < 0:
        raise ChunkingError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})."
        )


def _split_recursive(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Split ``text`` into pieces no longer than ``chunk_size`` where possible."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # No natural boundary left: hard-cut on the character window.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *rest = separators
    raw_parts = text.split(separator)
    # Re-attach the separator to the part it followed, so splitting on ". " does
    # not silently delete sentence-ending punctuation.
    parts = [part + separator for part in raw_parts[:-1]] + [raw_parts[-1]]

    pieces: list[str] = []
    buffer = ""
    for part in parts:
        candidate = buffer + part
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer)
        if len(part) > chunk_size:
            pieces.extend(_split_recursive(part, chunk_size, tuple(rest)))
            buffer = ""
        else:
            buffer = part
    if buffer:
        pieces.append(buffer)

    return [piece for piece in pieces if piece.strip()]


def _apply_overlap(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Prepend a tail of the previous piece to each piece, on a sentence boundary."""
    if chunk_overlap == 0 or len(pieces) < 2:
        return pieces

    merged: list[str] = [pieces[0]]
    for previous, current in zip(pieces, pieces[1:]):
        tail = previous[-chunk_overlap:]
        # Start the overlap on a sentence boundary, else a word boundary, so it
        # never begins mid-word. Blanks are filtered: a tail ending exactly on a
        # boundary yields a trailing "".
        sentences = [sentence for sentence in _SENTENCE_END.split(tail) if sentence.strip()]
        if len(sentences) > 1:
            tail = " ".join(sentences[1:])
        elif " " in tail.strip():
            tail = tail.strip().split(" ", 1)[1]
        tail = tail.strip()
        merged.append(f"{tail} {current}".strip() if tail else current)
    return merged


def chunk_document(
    document: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    namespace_salt: str = "",
) -> list[Chunk]:
    """Turn a :class:`Document` into overlapping, page-tagged chunks.

    Args:
        document: parsed PDF from :mod:`src.pdf_loader`.
        chunk_size: target characters per chunk.
        chunk_overlap: characters of context carried over between chunks.
        namespace_salt: extra string mixed into chunk IDs so the same document
            uploaded into two namespaces never collides.

    Returns:
        Chunks in reading order, each carrying a deterministic, unique ID.
    """
    _validate_params(chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    running_index = 0
    # Pinecone caps vector IDs at 512 bytes, so keep the readable prefix short.
    id_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", document.document_name)[:64]

    for page in document.pages:
        pieces = _split_recursive(page.text, chunk_size, _SEPARATORS)
        pieces = _apply_overlap(pieces, chunk_size, chunk_overlap)

        for piece in pieces:
            text = piece.strip()
            if len(text) < 30:  # noise: page headers, stray captions
                continue
            fingerprint = hashlib.sha1(
                f"{namespace_salt}|{document.document_name}|{page.page_number}|{running_index}|{text}".encode()
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=f"{id_stem}::p{page.page_number}::c{running_index}::{fingerprint}",
                    text=text,
                    document_name=document.document_name,
                    page_number=page.page_number,
                    chunk_index=running_index,
                )
            )
            running_index += 1

    return chunks
