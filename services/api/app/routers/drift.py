"""
Phase 7 — Drift endpoints.

GET /sessions/{id}/drift         → current drift state (recomputes if stale)
GET /sessions/{id}/drift/debug   → full explainability dump (DEBUG=true only)

_recompute_and_save is also imported by activity.py to update drift on
every telemetry batch.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user
from app.db.models import DocumentChunk, Session, SessionDriftState, User, UserBaseline
from app.db.session import get_db
from app.services.drift.features import extract_features
from app.services.drift.model import BETA0, compute_drift_result, elapsed_minutes_from_seconds
from app.services.drift.store import get_drift_state, upsert_drift_state
from app.services.drift.windowing import fetch_window

router = APIRouter(tags=["drift"])
log = logging.getLogger(__name__)

_STALE_THRESHOLD_S: int = 10


# ── Response schemas ──────────────────────────────────────────────────────────


class DriftStateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: int
    beta_effective: float
    attention_score: float
    drift_score: float
    drift_ema: float
    confidence: float
    baseline_used: bool = False
    updated_at: datetime


class DriftDebugResponse(BaseModel):
    """Full explainability payload — returned only when settings.debug=True."""
    session_id: int
    user_id: int
    baseline_used: bool
    # Computed state
    beta_effective: float
    beta_ema: float
    beta_raw: float
    attention_score: float
    drift_score: float
    drift_ema: float
    confidence: float
    # Window metadata
    n_batches_in_window: int
    elapsed_minutes: float
    # Per-term breakdown
    beta_components: dict[str, float]
    z_scores: dict[str, float]
    # Extracted window features
    features: dict[str, Any]
    # Baseline values actually used (for verification)
    baseline_snapshot: dict[str, Any]
    # Pace details
    pace_ratio: float
    pace_dev: float
    pace_available: bool
    window_wpm_effective: float


# ── Internal recompute helper ─────────────────────────────────────────────────


async def _recompute_and_save(
    session: Session,
    db: AsyncSession,
) -> SessionDriftState:
    """
    Fetch the rolling window, extract features, run the drift model,
    and upsert session_drift_states.  Returns the updated row.

    This is called:
    - From activity.py after every POST /activity/batch (fast path).
    - From the drift GET endpoint when the row is stale.
    """
    # ── Baseline ─────────────────────────────────────────────────────────────
    baseline_row = (
        await db.execute(
            select(UserBaseline).where(UserBaseline.user_id == session.user_id)
        )
    ).scalar_one_or_none()
    baseline: dict = baseline_row.baseline_json if baseline_row else {}
    baseline_used = baseline_row is not None

    if settings.debug:
        log.debug(
            "[drift] session=%d user=%d baseline_used=%s wpm_effective=%s",
            session.id,
            session.user_id,
            baseline_used,
            baseline.get("wpm_effective"),
        )

    # ── Paragraph word counts ─────────────────────────────────────────────────
    chunks_result = await db.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == session.document_id)
    )
    chunks = chunks_result.scalars().all()
    paragraph_word_counts: dict[str, int] = {}
    for c in chunks:
        wc = (c.meta or {}).get("word_count") or len(c.text.split())
        paragraph_word_counts[f"chunk-{c.id}"] = wc
        paragraph_word_counts[f"calib-{c.chunk_index}"] = wc
        paragraph_word_counts[f"chunk-{c.chunk_index}"] = wc

    # ── Rolling window ────────────────────────────────────────────────────────
    batches = await fetch_window(session.id, db)

    # ── Elapsed time (wall clock minutes from started_at) ─────────────────────
    now = datetime.now(timezone.utc)
    started = session.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed_s = max(0.0, (now - started).total_seconds())
    elapsed_min = elapsed_minutes_from_seconds(elapsed_s)

    # ── Previous state (EMA continuity) ───────────────────────────────────────
    prev_row = await get_drift_state(session.id, db)
    prev_ema = prev_row.drift_ema if prev_row else 0.0
    prev_beta_ema = prev_row.beta_ema if prev_row else BETA0

    # ── Feature extraction + model ────────────────────────────────────────────
    baseline_wpm = baseline.get("wpm_effective") or baseline.get("wpm_gross") or 0.0
    features = extract_features(batches, paragraph_word_counts, baseline_wpm)
    result = compute_drift_result(
        features, baseline, elapsed_min, prev_ema, prev_beta_ema
    )

    return await upsert_drift_state(session.id, result, db)


# ── Shared session ownership check ───────────────────────────────────────────


async def _get_owned_session(
    session_id: int, user_id: int, db: AsyncSession
) -> Session:
    row = (
        await db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return row


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/sessions/{session_id}/drift", response_model=DriftStateResponse)
async def get_drift(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DriftStateResponse:
    """
    Return the drift state for a session.

    If the state is missing or older than 10 seconds, recompute it
    on demand before responding.
    """
    session = await _get_owned_session(session_id, current_user.id, db)

    existing = await get_drift_state(session_id, db)
    now = datetime.now(timezone.utc)

    needs_recompute = (
        existing is None
        or (
            session.status == "active"
            and (now - existing.updated_at.replace(tzinfo=timezone.utc))
            > timedelta(seconds=_STALE_THRESHOLD_S)
        )
    )

    if needs_recompute and session.status == "active":
        row = await _recompute_and_save(session, db)
        await db.commit()
    else:
        row = existing
        if row is None:
            row = SessionDriftState(
                session_id=session_id,
                beta_effective=BETA0,
                beta_ema=BETA0,
                attention_score=1.0,
                drift_score=0.0,
                drift_ema=0.0,
                confidence=0.0,
                updated_at=now,
            )

    # Determine whether a baseline exists for the current user
    baseline_row = (
        await db.execute(
            select(UserBaseline).where(UserBaseline.user_id == current_user.id)
        )
    ).scalar_one_or_none()

    return DriftStateResponse(
        session_id=session_id,
        beta_effective=row.beta_effective,
        attention_score=row.attention_score,
        drift_score=row.drift_score,
        drift_ema=row.drift_ema,
        confidence=row.confidence,
        baseline_used=baseline_row is not None,
        updated_at=row.updated_at,
    )


@router.get("/sessions/{session_id}/drift/debug")
async def get_drift_debug(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DriftDebugResponse:
    """
    Full debug view: features, z-scores, beta components, baseline snapshot.
    Only available when settings.debug is True.

    This endpoint is the primary tool for diagnosing why drift is high or low.
    Check beta_components to see which term is the biggest contributor.
    """
    if not settings.debug:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Debug endpoint not available in production",
        )

    session = await _get_owned_session(session_id, current_user.id, db)

    baseline_row = (
        await db.execute(
            select(UserBaseline).where(UserBaseline.user_id == session.user_id)
        )
    ).scalar_one_or_none()
    baseline: dict = baseline_row.baseline_json if baseline_row else {}
    baseline_used = baseline_row is not None

    chunks_result = await db.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == session.document_id)
    )
    chunks = chunks_result.scalars().all()
    paragraph_word_counts: dict[str, int] = {}
    for c in chunks:
        wc = (c.meta or {}).get("word_count") or len(c.text.split())
        paragraph_word_counts[f"chunk-{c.id}"] = wc
        paragraph_word_counts[f"calib-{c.chunk_index}"] = wc
        paragraph_word_counts[f"chunk-{c.chunk_index}"] = wc

    batches = await fetch_window(session.id, db)

    now = datetime.now(timezone.utc)
    started = session.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed_s = max(0.0, (now - started).total_seconds())
    elapsed_min = elapsed_minutes_from_seconds(elapsed_s)

    prev_row = await get_drift_state(session.id, db)
    prev_ema = prev_row.drift_ema if prev_row else 0.0
    prev_beta_ema = prev_row.beta_ema if prev_row else BETA0

    baseline_wpm = baseline.get("wpm_effective") or baseline.get("wpm_gross") or 0.0
    features = extract_features(batches, paragraph_word_counts, baseline_wpm)
    result = compute_drift_result(
        features, baseline, elapsed_min, prev_ema, prev_beta_ema
    )

    # Build a human-readable baseline snapshot so the caller can verify
    # which values were actually used in normalization.
    baseline_snapshot: dict[str, Any] = {
        "baseline_used": baseline_used,
        "wpm_effective": baseline.get("wpm_effective"),
        "wpm_gross": baseline.get("wpm_gross"),
        "idle_ratio_mean": baseline.get("idle_ratio_mean", 0.05),
        "idle_ratio_std": baseline.get("idle_ratio_std", 0.08),
        "idle_seconds_mean": baseline.get("idle_seconds_mean", 1.0),
        "idle_seconds_std": baseline.get("idle_seconds_std", 1.0),
        "scroll_jitter_mean": baseline.get("scroll_jitter_mean", 0.10),
        "scroll_jitter_std": baseline.get("scroll_jitter_std", 0.10),
        "regress_rate_mean": baseline.get("regress_rate_mean", 0.05),
        "regress_rate_std": baseline.get("regress_rate_std", 0.06),
        "scroll_velocity_norm_mean": baseline.get("scroll_velocity_norm_mean"),
        "scroll_velocity_norm_std": baseline.get("scroll_velocity_norm_std"),
        "presentation_profile": baseline.get("presentation_profile"),
    }

    z_dict = dataclasses.asdict(result.z_scores)
    features_dict = dataclasses.asdict(features)

    return DriftDebugResponse(
        session_id=session_id,
        user_id=session.user_id,
        baseline_used=baseline_used,
        beta_effective=result.beta_effective,
        beta_ema=result.beta_ema,
        beta_raw=result.beta_components.get("beta_raw", result.beta_effective),
        attention_score=result.attention_score,
        drift_score=result.drift_score,
        drift_ema=result.drift_ema,
        confidence=result.confidence,
        n_batches_in_window=features.n_batches,
        elapsed_minutes=elapsed_min,
        beta_components=result.beta_components,
        z_scores=z_dict,
        features=features_dict,
        baseline_snapshot=baseline_snapshot,
        pace_ratio=features.pace_ratio,
        pace_dev=features.pace_dev,
        pace_available=features.pace_available,
        window_wpm_effective=features.window_wpm_effective,
    )
