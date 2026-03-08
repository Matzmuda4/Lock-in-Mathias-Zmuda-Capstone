"""
Rolling telemetry window query — Phase 7.

Fetches the last 30 seconds of telemetry_batch events for a session
from the TimescaleDB hypertable.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_WINDOW_SECONDS: int = 30

# SQL uses an interval literal constructed server-side to avoid parameter
# binding limitations with INTERVAL expressions in asyncpg.
_WINDOW_SQL = text(
    "SELECT payload FROM activity_events "
    "WHERE session_id = :sid "
    "  AND event_type = 'telemetry_batch' "
    "  AND created_at >= NOW() - INTERVAL '30 seconds' "
    "ORDER BY created_at ASC"
)


async def fetch_window(session_id: int, db: AsyncSession) -> list[dict[str, Any]]:
    """
    Return an ordered list of telemetry_batch payload dicts for the last
    30 seconds of the given session.

    Uses a raw SQL query for performance — avoids loading full ORM objects.
    """
    result = await db.execute(_WINDOW_SQL, {"sid": session_id})
    return [row[0] for row in result.fetchall()]
