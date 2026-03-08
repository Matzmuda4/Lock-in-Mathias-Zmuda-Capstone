"""
Rolling telemetry window query — Phase 7.

Fetches the last 30 seconds of telemetry_batch events (and progress_marker
events) for a session from the TimescaleDB hypertable.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_WINDOW_SECONDS: int = 30

_WINDOW_SQL = text(
    "SELECT payload FROM activity_events "
    "WHERE session_id = :sid "
    "  AND event_type = 'telemetry_batch' "
    "  AND created_at >= NOW() - INTERVAL '30 seconds' "
    "ORDER BY created_at ASC"
)

_MARKERS_SQL = text(
    "SELECT COUNT(*) FROM activity_events "
    "WHERE session_id = :sid "
    "  AND event_type = 'progress_marker' "
    "  AND created_at >= NOW() - INTERVAL '30 seconds'"
)


async def fetch_window(session_id: int, db: AsyncSession) -> list[dict[str, Any]]:
    """
    Return an ordered list of telemetry_batch payload dicts for the last
    30 seconds of the given session.
    """
    result = await db.execute(_WINDOW_SQL, {"sid": session_id})
    return [row[0] for row in result.fetchall()]


async def fetch_progress_markers_count(session_id: int, db: AsyncSession) -> int:
    """
    Return the number of 'progress_marker' events in the last 30 seconds
    for this session. These are emitted when the user explicitly advances
    (e.g., Load More / Next Section) and represent strong engagement signals.
    """
    result = await db.execute(_MARKERS_SQL, {"sid": session_id})
    row = result.fetchone()
    return int(row[0]) if row else 0
