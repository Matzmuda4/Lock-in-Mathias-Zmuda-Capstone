"""
Persistence helpers for drift state — Phase 7.

The session_drift_states table is a simple key-value store keyed by session_id.
We upsert on every drift recompute.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SessionDriftHistory, SessionDriftState
from app.services.drift.types import DriftResult

# Append to history every N upserts (~10 seconds at 2s batches = every 5th update)
_HISTORY_EVERY_N: int = 5
_upsert_counters: dict[int, int] = {}


async def upsert_drift_state(
    session_id: int,
    result: DriftResult,
    db: AsyncSession,
) -> SessionDriftState:
    """
    Insert or update the drift state row for a session (latest snapshot).

    Also appends a history row to session_drift_history every ~10 seconds
    (every _HISTORY_EVERY_N calls) so the full drift trajectory is preserved.
    """
    now = datetime.now(timezone.utc)

    existing_q = await db.execute(
        select(SessionDriftState).where(SessionDriftState.session_id == session_id)
    )
    row = existing_q.scalar_one_or_none()

    if row is None:
        row = SessionDriftState(
            session_id=session_id,
            drift_level=result.drift_level,
            drift_ema=result.drift_ema,
            disruption_score=result.disruption_score,
            engagement_score=result.engagement_score,
            beta_effective=result.beta_effective,
            beta_ema=result.beta_ema,
            attention_score=result.attention_score,
            drift_score=result.drift_score,
            confidence=result.confidence,
            last_window_ends_at=now,
            updated_at=now,
        )
        db.add(row)
    else:
        row.drift_level = result.drift_level
        row.drift_ema = result.drift_ema
        row.disruption_score = result.disruption_score
        row.engagement_score = result.engagement_score
        row.beta_effective = result.beta_effective
        row.beta_ema = result.beta_ema
        row.attention_score = result.attention_score
        row.drift_score = result.drift_score
        row.confidence = result.confidence
        row.last_window_ends_at = now
        row.updated_at = now

    # ── Append to drift history every ~10 seconds ─────────────────────────
    _upsert_counters[session_id] = _upsert_counters.get(session_id, 0) + 1
    if _upsert_counters[session_id] % _HISTORY_EVERY_N == 1:
        db.add(SessionDriftHistory(
            session_id=session_id,
            created_at=now,
            drift_ema=result.drift_ema,
            beta_ema=result.beta_ema,
            disruption_score=result.disruption_score,
            engagement_score=result.engagement_score,
            confidence=result.confidence,
            pace_ratio=result.pace_ratio,
        ))

    await db.flush()
    return row


async def get_drift_state(
    session_id: int,
    db: AsyncSession,
) -> SessionDriftState | None:
    """Return the current drift state row or None if it does not exist yet."""
    result = await db.execute(
        select(SessionDriftState).where(SessionDriftState.session_id == session_id)
    )
    return result.scalar_one_or_none()
