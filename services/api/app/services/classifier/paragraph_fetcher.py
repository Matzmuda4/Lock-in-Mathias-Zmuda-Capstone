"""
Paragraph text window fetcher for the intervention LLM.

Given a document_id and a chunk_index (the integer emitted by the frontend
as ``current_chunk_index`` in every telemetry batch), this module returns a
sliding window of consecutive DocumentChunk texts centred on the current
reading position.

Why chunk_index, not the string current_paragraph_id?
  The frontend sets data-paragraph-id = "chunk-{chunk.id}" (the DB primary
  key, not the sequential chunk_index) on each rendered paragraph element.
  It also sets data-chunk-index = chunkIndex (the sequential DocumentChunk
  .chunk_index from the ordered render array).  Using the integer
  current_chunk_index avoids any string parsing and maps directly to
  DocumentChunk.chunk_index in a single WHERE clause.

Usage at inference time
-----------------------
_run_classification (drift.py) calls fetch_text_window after every full-window
RF classification and stores the result inside intervention_context JSONB on
the session_attentional_states row.  The intervention LLM therefore receives
the paragraph text without any additional joins at prompt assembly time.

Usage at training time
----------------------
build_intervention_dataset.py queries activity_events to recover the
current_chunk_index that was active at the end of each supervised.jsonl
window, then calls the standalone dict-based helper for the same logic.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk

log = logging.getLogger(__name__)


async def fetch_text_window(
    document_id: int | None,
    chunk_index: int | None,
    db: AsyncSession,
    window_size: int = 3,
) -> list[str]:
    """
    Return up to ``window_size`` text paragraphs centred on ``chunk_index``
    for document ``document_id``.

    The window is symmetric: for window_size=3 it returns the paragraph
    before, the current paragraph, and the paragraph after (subject to
    document boundaries).  Image and table chunks (where text is empty) are
    automatically excluded so the LLM always receives readable prose.

    Parameters
    ----------
    document_id  : Document.id from the active session
    chunk_index  : DocumentChunk.chunk_index of the current paragraph
                   (the integer ``current_chunk_index`` from telemetry)
    db           : active async SQLAlchemy session
    window_size  : number of text paragraphs to return (default 3)

    Returns
    -------
    List of stripped paragraph strings.  May be shorter than window_size if
    the user is near the document start/end, or if chunk_index is None.
    Returns [] when document_id or chunk_index is None or on DB error.
    """
    if document_id is None or chunk_index is None:
        return []

    half = window_size // 2
    start_idx = max(0, chunk_index - half)
    # Fetch a slightly larger slice to account for image/table gaps, then
    # trim to window_size after filtering.
    fetch_count = window_size + 4

    try:
        result = await db.execute(
            select(DocumentChunk.chunk_index, DocumentChunk.text)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_index >= start_idx,
                DocumentChunk.chunk_index < start_idx + fetch_count,
            )
            .order_by(DocumentChunk.chunk_index)
        )
        rows = result.all()
    except Exception as exc:
        log.warning(
            "[paragraph_fetcher] DB query failed (doc=%s, chunk=%s): %s",
            document_id, chunk_index, exc,
        )
        return []

    texts: list[str] = []
    for row in rows:
        text = (row.text or "").strip()
        if text:  # skip image / table placeholders which have empty text
            texts.append(text)
        if len(texts) >= window_size:
            break

    return texts


def text_window_from_dict(
    chunks: dict[int, str],
    chunk_index: int,
    window_size: int = 3,
) -> list[str]:
    """
    Synchronous variant for the dataset builder that works against an
    in-memory ``{chunk_index: text}`` dict (pre-loaded from the DB).

    Returns up to ``window_size`` non-empty text strings centred on
    ``chunk_index``.  Falls back gracefully at document boundaries.
    """
    half = window_size // 2
    start = max(0, chunk_index - half)
    texts: list[str] = []

    # Scan forward from start until we have enough text paragraphs
    idx = start
    while len(texts) < window_size and idx < start + window_size + 4:
        text = (chunks.get(idx) or "").strip()
        if text:
            texts.append(text)
        idx += 1

    return texts
