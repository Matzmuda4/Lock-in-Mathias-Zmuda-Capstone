"""
Persistence helpers for drift state — Phase 7.

The session_drift_states table is a simple key-value store keyed by session_id.
We upsert on every drift recompute.

session_state_packets is an append-only hypertable storing full state packets
every ~10 s (same cadence as session_drift_history).  These packets are the
primary training-data source exported by the classify pipeline.

Each packet is self-contained: it includes the embedded baseline_snapshot so
it can be used for training without a separate baseline join.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    SessionDriftHistory,
    SessionDriftState,
    SessionStatePacket,
    UserBaseline,
)
from app.services.drift.types import DriftResult

# Append to history every N upserts (~10 seconds at 2s batches = every 5th update)
_HISTORY_EVERY_N: int = 5
_upsert_counters: dict[int, int] = {}


async def upsert_drift_state(
    session_id: int,
    result: DriftResult,
    db: AsyncSession,
    baseline_row: Optional[UserBaseline] = None,
    window_end_at: Optional[datetime] = None,
) -> SessionDriftState:
    """
    Insert or update the drift state row for a session (latest snapshot).

    Also appends rows to session_drift_history and session_state_packets every
    ~10 seconds (_HISTORY_EVERY_N calls) to preserve the full trajectory.

    Parameters
    ----------
    baseline_row : optional UserBaseline ORM row for the session's user.
        When provided, a baseline_snapshot is embedded in the state packet for
        self-contained training data (no separate baseline join required).
    window_end_at : optional timestamp of the newest telemetry batch in the
        30 s window that produced this drift result.  Used to compute
        window_start_at = window_end_at - 30 s.  Falls back to now() if absent.
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

    # ── Append to history and state-packet tables every ~10 seconds ──────
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

        # Compute packet_seq (next monotonic counter for this session)
        seq_result = await db.execute(
            select(func.coalesce(func.max(SessionStatePacket.packet_seq), -1) + 1).where(
                SessionStatePacket.session_id == session_id
            )
        )
        next_seq: int = seq_result.scalar() or 0

        # Window bounds
        w_end = window_end_at or now
        w_start = w_end - timedelta(seconds=30)

        # Store full state packet for training-data export
        db.add(SessionStatePacket(
            session_id=session_id,
            created_at=now,
            packet_seq=next_seq,
            window_start_at=w_start,
            window_end_at=w_end,
            packet_json=_build_packet_json(result, baseline_row),
        ))

    await db.flush()
    return row


def _build_packet_json(
    result: DriftResult,
    baseline_row: Optional[UserBaseline] = None,
) -> dict:
    """
    Serialise a DriftResult into a self-contained dict for storage.

    Embeds a baseline_snapshot so each packet can be used for training
    without a separate join to user_baselines.
    """
    baseline_json: dict[str, Any] = baseline_row.baseline_json if baseline_row else {}
    wpm_eff = baseline_json.get("wpm_effective") or baseline_json.get("wpm_gross") or 0.0
    baseline_valid = (
        baseline_row is not None
        and wpm_eff > 0
        and baseline_json.get("calibration_duration_seconds", 0) >= 60
    )
    baseline_snapshot: dict[str, Any] = {
        "baseline_json": baseline_json or None,
        "baseline_updated_at": (
            baseline_row.updated_at.isoformat() if baseline_row and baseline_row.updated_at else None
        ),
        "baseline_valid": baseline_valid,
    }

    return {
        "drift": {
            "drift_level": result.drift_level,
            "drift_ema": result.drift_ema,
            "beta_effective": result.beta_effective,
            "beta_ema": result.beta_ema,
            "disruption_score": result.disruption_score,
            "engagement_score": result.engagement_score,
            "confidence": result.confidence,
            "pace_ratio": result.pace_ratio,
            "pace_available": result.pace_available,
        },
        "features": dataclasses.asdict(result.features),
        "z_scores": dataclasses.asdict(result.z_scores),
        "debug": result.beta_components,
        "baseline_snapshot": baseline_snapshot,
    }


async def get_drift_state(
    session_id: int,
    db: AsyncSession,
) -> SessionDriftState | None:
    """Return the current drift state row or None if it does not exist yet."""
    result = await db.execute(
        select(SessionDriftState).where(SessionDriftState.session_id == session_id)
    )
    return result.scalar_one_or_none()
