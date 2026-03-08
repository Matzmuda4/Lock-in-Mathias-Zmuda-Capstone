"""
Persistence helpers for drift state — Phase 7.

The session_drift_states table is a simple key-value store keyed by session_id.
We upsert on every drift recompute.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SessionDriftState
from app.services.drift.types import DriftResult


async def upsert_drift_state(
    session_id: int,
    result: DriftResult,
    db: AsyncSession,
) -> SessionDriftState:
    """Insert or update the drift state row for a session."""
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
