"""
In-memory classification cache.

Stores the latest ClassificationResult per session_id.
No DB writes — this is the transient live state used by the API endpoint.
Once DB tables are added later, results will also be persisted there.

Thread-safety: asyncio single-threaded — plain dict is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .base import ClassificationResult

_MAX_SESSIONS: int = 500  # cap to prevent unbounded memory growth


@dataclass
class CachedClassification:
    result: ClassificationResult
    session_id: int
    packet_seq: int
    classified_at: datetime


class ClassificationCache:
    """Latest-per-session in-memory store."""

    def __init__(self) -> None:
        self._store: dict[int, CachedClassification] = {}

    def put(
        self,
        session_id: int,
        packet_seq: int,
        result: ClassificationResult,
    ) -> None:
        if len(self._store) >= _MAX_SESSIONS and session_id not in self._store:
            # Evict the oldest entry
            oldest_key = min(
                self._store, key=lambda k: self._store[k].classified_at
            )
            del self._store[oldest_key]

        self._store[session_id] = CachedClassification(
            result=result,
            session_id=session_id,
            packet_seq=packet_seq,
            classified_at=datetime.now(timezone.utc),
        )

    def get(self, session_id: int) -> Optional[CachedClassification]:
        return self._store.get(session_id)

    def evict(self, session_id: int) -> None:
        self._store.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._store)
