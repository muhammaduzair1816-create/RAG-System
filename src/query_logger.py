"""Query logging (intermediate-level enhancement).

Every question is appended to ``logs/queries.jsonl`` as one JSON object per
line, which makes the log trivially loadable into pandas for the performance
analysis in the technical report.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LOG_DIR
from .models import Answer
from .retriever import RetrievalResult

QUERY_LOG_PATH = LOG_DIR / "queries.jsonl"
APP_LOG_PATH = LOG_DIR / "app.log"

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Set up file + console logging once per process."""
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(APP_LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    _configured = True


def log_query(
    result: RetrievalResult,
    answer: Answer,
    namespace: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one structured record describing a completed question."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "namespace": namespace,
        "query": result.query,
        "top_k": result.top_k,
        "similarity_threshold": result.threshold,
        "chunks_retrieved": len(result.chunks),
        "chunks_below_threshold": result.discarded_below_threshold,
        "overview_fallback": result.overview_fallback,
        "best_score": round(result.best_score, 4),
        "mean_score": round(result.mean_score, 4),
        "grounded": answer.grounded,
        "confidence": answer.confidence,
        "model": answer.model,
        "latency_seconds": answer.latency_seconds,
        "sources": [
            {"document": source.document_name, "page": source.page_number, "score": round(source.score, 4)}
            for source in answer.sources
        ],
    }
    if extra:
        record.update(extra)

    try:
        with QUERY_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:  # logging must never break the user-facing flow
        logging.getLogger(__name__).warning("Could not write query log: %s", exc)


def read_query_log(limit: int = 200) -> list[dict[str, Any]]:
    """Return the most recent log records, newest first."""
    path = Path(QUERY_LOG_PATH)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:][::-1]
