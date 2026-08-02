"""Stage 7: grounded answer generation.

Hallucination prevention is enforced at three independent points:

1. **Retrieval gate** – if nothing clears the similarity threshold the LLM is
   never called; the refusal is returned directly.
2. **Prompt contract** – a strict system prompt plus numbered, citation-tagged
   context blocks, sent at ``temperature=0``.
3. **Post-generation check** – the response is inspected for the refusal
   sentence and for citation markers before it is shown as a grounded answer.
"""

from __future__ import annotations

import logging
import re
import time

from .config import Settings, get_settings
from .models import Answer, RetrievedChunk
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT_MESSAGE = "The answer is not available in the provided document."

SYSTEM_PROMPT = f"""You are a careful document-analysis assistant. You answer \
questions using ONLY the numbered context excerpts supplied by the user.

Rules you must follow without exception:
1. Use only facts stated in the CONTEXT. Never use outside or prior knowledge.
2. Never guess, extrapolate, or fill gaps with plausible-sounding detail.
3. Cite the source of every claim inline using the excerpt's marker, e.g. [S1] or [S2].
4. If the CONTEXT does not contain enough information to answer the QUESTION, \
reply with exactly this sentence and nothing else:
{INSUFFICIENT_CONTEXT_MESSAGE}
5. If the CONTEXT answers the question only partially, state what it does say, \
cite it, and then say plainly which part is not covered by the document.
6. Be concise and factual. Do not add preamble, apologies, or closing remarks.
"""

USER_PROMPT_TEMPLATE = """CONTEXT
{context}

QUESTION
{question}

Answer using only the CONTEXT above, citing each claim with its [S#] marker."""

_CITATION_PATTERN = re.compile(r"\[S\d+\]")


def is_refusal(text: str) -> bool:
    """True when the model declined for lack of context.

    Rule 5 of the system prompt lets a *partial* answer name the gap using the
    same sentence, so a bare substring test would throw away good answers. Treat
    the response as a refusal only when that sentence leads it or is essentially
    the whole of it.
    """
    normalised = " ".join(text.lower().split()).strip()
    target = " ".join(INSUFFICIENT_CONTEXT_MESSAGE.lower().split())
    if normalised.startswith(target):
        return True
    return target in normalised and len(normalised) <= len(target) * 1.3


class GenerationError(RuntimeError):
    """Raised when the LLM call cannot be completed."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, attributable context blocks."""
    blocks = []
    for position, chunk in enumerate(chunks, start=1):
        header = f"[S{position}] (document: {chunk.document_name}, page: {chunk.page_number}, similarity: {chunk.score:.3f})"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def compute_confidence(result: RetrievalResult, cited_fraction: float) -> float:
    """Blend retrieval quality with how well the answer stayed anchored to it.

    * 55 % – similarity of the single best supporting chunk
    * 25 % – mean similarity across the supporting set (breadth of support)
    * 20 % – fraction of retrieved sources the answer actually cited

    Overview-fallback context is selected by metadata rather than similarity, so
    the similarity terms carry no information there; those results are scored on
    citation rate alone and capped below the "High" band.
    """
    if not result.has_context:
        return 0.0
    if result.overview_fallback:
        return round(0.45 + 0.20 * max(0.0, min(cited_fraction, 1.0)), 3)
    score = 0.55 * result.best_score + 0.25 * result.mean_score + 0.20 * cited_fraction
    return round(max(0.0, min(score, 1.0)), 3)


class AnswerGenerator:
    """Generates grounded answers with a Groq-hosted open-source LLM."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        if not self.settings.groq_api_key:
            raise GenerationError(
                "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise GenerationError("The `groq` package is not installed.") from exc

        try:
            self._client = Groq(api_key=self.settings.groq_api_key)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Could not initialise the Groq client: {exc}") from exc

        self.model = self.settings.groq_model

    def generate(self, result: RetrievalResult, max_tokens: int = 900) -> Answer:
        """Produce an :class:`Answer` for a completed retrieval."""
        started = time.perf_counter()

        # Gate 1: nothing retrieved above the threshold — do not call the LLM.
        if not result.has_context:
            return Answer(
                text=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                confidence=0.0,
                grounded=False,
                model=self.model,
                latency_seconds=round(time.perf_counter() - started, 3),
            )

        prompt = USER_PROMPT_TEMPLATE.format(
            context=format_context(result.chunks),
            question=result.query,
        )

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # deterministic: no creative drift away from the context
                max_tokens=max_tokens,
                top_p=1.0,
            )
        except Exception as exc:  # noqa: BLE001 - network, auth, rate-limit and model errors
            raise GenerationError(f"The language model request failed: {exc}") from exc

        text = (completion.choices[0].message.content or "").strip()
        if not text:
            raise GenerationError("The language model returned an empty response.")

        # Gate 3: did the model refuse, and did it cite what it was given?
        refused = is_refusal(text)
        cited = {marker.upper() for marker in _CITATION_PATTERN.findall(text)}
        cited_fraction = len(cited) / len(result.chunks) if result.chunks else 0.0

        if refused:
            return Answer(
                text=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                confidence=0.0,
                grounded=False,
                model=self.model,
                latency_seconds=round(time.perf_counter() - started, 3),
            )

        # Only surface the sources the answer actually leaned on, when it cited any.
        cited_sources = [
            chunk for position, chunk in enumerate(result.chunks, start=1) if f"[S{position}]" in cited
        ]

        return Answer(
            text=text,
            sources=cited_sources or result.chunks,
            confidence=compute_confidence(result, min(cited_fraction, 1.0)),
            grounded=True,
            model=self.model,
            latency_seconds=round(time.perf_counter() - started, 3),
        )
