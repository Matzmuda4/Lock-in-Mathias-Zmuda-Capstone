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
from typing import Any, Optional

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
from app.services.drift.windowing import fetch_progress_markers_count, fetch_window

router = APIRouter(tags=["drift"])
log = logging.getLogger(__name__)

_STALE_THRESHOLD_S: int = 10


# ── Response schemas ──────────────────────────────────────────────────────────


class DriftStateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: int
    # Primary bidirectional state
    drift_level: float = 0.0
    drift_ema: float = 0.0
    disruption_score: float = 0.0
    engagement_score: float = 0.0
    confidence: float = 0.0
    # Legacy / compat
    beta_effective: float = 0.0
    attention_score: float = 1.0
    drift_score: float = 0.0
    # Optional pace info
    pace_ratio: Optional[float] = None
    pace_available: bool = False
    baseline_used: bool = False
    updated_at: datetime


class DriftDebugResponse(BaseModel):
    """Full explainability payload — returned only when settings.debug=True."""

    session_id: int
    user_id: int
    baseline_used: bool
    # Primary state
    drift_level: float
    drift_ema: float
    disruption_score: float
    engagement_score: float
    confidence: float
    # Legacy compat
    beta_effective: float
    beta_ema: float
    attention_score: float
    drift_score: float
    # Window metadata
    n_batches_in_window: int
    elapsed_minutes: float
    # Per-term breakdown
    beta_components: dict[str, Any]
    z_scores: dict[str, Any]
    # Extracted window features
    features: dict[str, Any]
    # Baseline values actually used
    baseline_snapshot: dict[str, Any]
    # Pace details
    pace_ratio: Optional[float]
    pace_dev: float
    pace_available: bool
    window_wpm_effective: float
    # Data quality flags
    telemetry_fault_rate: float = 0.0
    scroll_capture_fault_rate: float = 0.0
    paragraph_missing_fault_rate: float = 0.0
    quality_confidence_mult: float = 1.0
    baseline_valid: bool = True
    skim_fallback_z: float = 0.0


# ── Internal recompute helper ─────────────────────────────────────────────────


async def _recompute_and_save(
    session: Session,
    db: AsyncSession,
) -> SessionDriftState:
    """
    Fetch the rolling window, extract features, run the drift model,
    and upsert session_drift_states.  Returns the updated row.

    Called:
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
    progress_markers = await fetch_progress_markers_count(session.id, db)

    # ── Elapsed time (wall clock minutes from started_at) ─────────────────────
    now = datetime.now(timezone.utc)
    started = session.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed_s = max(0.0, (now - started).total_seconds())
    elapsed_min = elapsed_minutes_from_seconds(elapsed_s)

    # ── Previous state (beta EMA continuity across cycles) ───────────────────
    prev_row = await get_drift_state(session.id, db)
    prev_ema = prev_row.drift_ema if prev_row else 0.0
    # beta_ema carries the smoothed decay rate; starts at BETA0 (natural decay)
    prev_beta_ema = prev_row.beta_ema if prev_row else BETA0

    # ── Personalized pause threshold ──────────────────────────────────────────
    b = baseline
    pause_thresh = max(2.0, b.get("para_dwell_median_s", 10.0) / 3.0)

    # ── Feature extraction + model ────────────────────────────────────────────
    baseline_wpm = b.get("wpm_effective") or b.get("wpm_gross") or 0.0
    features = extract_features(
        batches, paragraph_word_counts, baseline_wpm, pause_threshold_s=pause_thresh
    )
    # Attach progress markers
    features.progress_markers_count = progress_markers

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

    If the state is missing or older than 10 seconds (for active sessions),
    recompute it on demand before responding.
    """
    session = await _get_owned_session(session_id, current_user.id, db)

    existing = await get_drift_state(session_id, db)
    now = datetime.now(timezone.utc)

    needs_recompute = existing is None or (
        session.status == "active"
        and (now - existing.updated_at.replace(tzinfo=timezone.utc))
        > timedelta(seconds=_STALE_THRESHOLD_S)
    )

    if needs_recompute and session.status == "active":
        row = await _recompute_and_save(session, db)
        await db.commit()
    else:
        row = existing

    # Fall back to a zero-state object if nothing exists
    if row is None:
        return DriftStateResponse(
            session_id=session_id,
            drift_level=0.0,
            drift_ema=0.0,
            disruption_score=0.0,
            engagement_score=0.0,
            confidence=0.0,
            beta_effective=0.0,
            attention_score=1.0,
            drift_score=0.0,
            pace_ratio=None,
            pace_available=False,
            baseline_used=False,
            updated_at=now,
        )

    # Determine whether a baseline exists
    baseline_row = (
        await db.execute(
            select(UserBaseline).where(UserBaseline.user_id == current_user.id)
        )
    ).scalar_one_or_none()

    return DriftStateResponse(
        session_id=session_id,
        drift_level=row.drift_level,
        drift_ema=row.drift_ema,
        disruption_score=row.disruption_score,
        engagement_score=row.engagement_score,
        confidence=row.confidence,
        beta_effective=row.beta_effective,
        attention_score=row.attention_score,
        drift_score=row.drift_score,
        pace_ratio=None,
        pace_available=False,
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
    Full debug view: features, z-scores, component breakdown, baseline snapshot.
    Only available when settings.debug is True.

    Check beta_components (now: disruption/engagement breakdown) to see
    which signals are driving drift up or down.
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
    progress_markers = await fetch_progress_markers_count(session.id, db)

    now = datetime.now(timezone.utc)
    started = session.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed_s = max(0.0, (now - started).total_seconds())
    elapsed_min = elapsed_minutes_from_seconds(elapsed_s)

    prev_row = await get_drift_state(session.id, db)
    prev_ema = prev_row.drift_ema if prev_row else 0.0
    prev_beta_ema = prev_row.beta_ema if prev_row else BETA0

    b = baseline
    pause_thresh = max(2.0, b.get("para_dwell_median_s", 10.0) / 3.0)
    baseline_wpm = b.get("wpm_effective") or b.get("wpm_gross") or 0.0
    features = extract_features(
        batches, paragraph_word_counts, baseline_wpm, pause_threshold_s=pause_thresh
    )
    features.progress_markers_count = progress_markers

    result = compute_drift_result(
        features, baseline, elapsed_min, prev_ema, prev_beta_ema
    )

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
        "para_dwell_median_s": baseline.get("para_dwell_median_s"),
        "para_dwell_iqr_s": baseline.get("para_dwell_iqr_s"),
        "scroll_velocity_norm_mean": baseline.get("scroll_velocity_norm_mean"),
        "scroll_velocity_norm_std": baseline.get("scroll_velocity_norm_std"),
        "presentation_profile": baseline.get("presentation_profile"),
    }

    z_dict = dataclasses.asdict(result.z_scores)
    features_dict = dataclasses.asdict(features)

    wpm_eff = baseline.get("wpm_effective") or baseline.get("wpm_gross") or 0.0
    baseline_valid = baseline_used and wpm_eff > 0

    return DriftDebugResponse(
        session_id=session_id,
        user_id=session.user_id,
        baseline_used=baseline_used,
        drift_level=result.drift_level,
        drift_ema=result.drift_ema,
        disruption_score=result.disruption_score,
        engagement_score=result.engagement_score,
        confidence=result.confidence,
        beta_effective=result.beta_effective,
        beta_ema=result.beta_ema,
        attention_score=result.attention_score,
        drift_score=result.drift_score,
        n_batches_in_window=features.n_batches,
        elapsed_minutes=elapsed_min,
        beta_components=result.beta_components,
        z_scores=z_dict,
        features=features_dict,
        baseline_snapshot=baseline_snapshot,
        pace_ratio=result.pace_ratio,
        pace_dev=features.pace_dev,
        pace_available=features.pace_available,
        window_wpm_effective=features.window_wpm_effective,
        telemetry_fault_rate=features.telemetry_fault_rate,
        scroll_capture_fault_rate=features.scroll_capture_fault_rate,
        paragraph_missing_fault_rate=features.paragraph_missing_fault_rate,
        quality_confidence_mult=features.quality_confidence_mult,
        baseline_valid=baseline_valid,
        skim_fallback_z=result.beta_components.get("skim_fallback_z", 0.0),
    )


# ── Part D: LLM-ready state packet ────────────────────────────────────────────


class StatePacketFlags(BaseModel):
    baseline_valid: bool
    baseline_wpm_valid: bool
    telemetry_fault_rate: float
    scroll_capture_fault_rate: float
    paragraph_missing_fault_rate: float
    quality_confidence_mult: float


class StatePacketDrift(BaseModel):
    drift_ema: float
    drift_level: float
    beta_ema: float
    disruption_score: float
    engagement_score: float
    confidence: float


class StatePacketContext(BaseModel):
    progress_ratio: float
    current_paragraph_id: Optional[str]
    pace_ratio: Optional[float]
    pace_available: bool
    progress_velocity: float


class StatePacket(BaseModel):
    """
    Structured payload prepared for the future LLM attentional-state classifier.

    The LLM will later map these signals to probabilistic state labels
    (Focused / Drifting / Hyperfocused / Fatigued) and select an
    intervention tier.  This endpoint produces the payload only — no LLM
    is invoked here.
    """
    session_id: int
    user_id: int
    computed_at: datetime
    # Compact baseline snapshot (keys the LLM needs)
    baseline_snapshot: dict[str, Any]
    # Extracted window features (current 30 s)
    window_features: dict[str, Any]
    # Normalised z-scores
    z_scores: dict[str, Any]
    # Drift state
    drift: StatePacketDrift
    # Data-quality / validity flags
    flags: StatePacketFlags
    # Reading context
    context: StatePacketContext


@router.get("/sessions/{session_id}/state-packet", response_model=StatePacket)
async def get_state_packet(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StatePacket:
    """
    LLM-ready input packet for session {session_id}.

    Returns a structured payload containing:
    - Calibration baseline snapshot (selected keys)
    - Extracted 30-second window features
    - Normalised z-scores for all signals
    - Current drift state (drift_ema, beta_ema, disruption, engagement)
    - Data-quality flags (baseline_valid, telemetry_fault, etc.)
    - Reading context (progress_ratio, paragraph_id)

    No LLM is invoked.  This endpoint is preparation for Phase 8.
    Ownership enforced: only the session owner can access.
    """
    session = await _get_owned_session(session_id, current_user.id, db)

    # Baseline
    baseline_row = (
        await db.execute(
            select(UserBaseline).where(UserBaseline.user_id == session.user_id)
        )
    ).scalar_one_or_none()
    baseline: dict = baseline_row.baseline_json if baseline_row else {}

    wpm_eff = baseline.get("wpm_effective") or baseline.get("wpm_gross") or 0.0
    baseline_valid = baseline_row is not None and wpm_eff > 0

    # Paragraph word counts
    chunks_result = await db.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == session.document_id)
    )
    chunks = chunks_result.scalars().all()
    paragraph_word_counts: dict[str, int] = {}
    for c in chunks:
        wc = (c.meta or {}).get("word_count") or len(c.text.split())
        paragraph_word_counts[f"chunk-{c.id}"] = wc
        paragraph_word_counts[f"calib-{c.chunk_index}"] = wc

    batches = await fetch_window(session.id, db)
    progress_markers = await fetch_progress_markers_count(session.id, db)

    now = datetime.now(timezone.utc)
    started = session.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed_s = max(0.0, (now - started).total_seconds())
    elapsed_min = elapsed_s / 60.0

    b = baseline
    pause_thresh = max(2.0, b.get("para_dwell_median_s", 10.0) / 3.0)
    features = extract_features(batches, paragraph_word_counts, wpm_eff, pause_threshold_s=pause_thresh)
    features.progress_markers_count = progress_markers

    from app.services.drift.model import compute_z_scores, compute_drift_result, BETA0 as _BETA0
    prev_row = await get_drift_state(session.id, db)
    prev_ema = prev_row.drift_ema if prev_row else 0.0
    prev_beta_ema = prev_row.beta_ema if prev_row else _BETA0

    result = compute_drift_result(features, baseline, elapsed_min, prev_ema, prev_beta_ema)

    import dataclasses as _dc
    z_dict = _dc.asdict(result.z_scores)
    feat_dict = _dc.asdict(features)

    # Latest batch context
    last_batch = batches[-1] if batches else {}

    baseline_snapshot = {
        "wpm_effective": wpm_eff,
        "wpm_gross": baseline.get("wpm_gross"),
        "idle_ratio_mean": baseline.get("idle_ratio_mean", 0.35),
        "idle_ratio_std": baseline.get("idle_ratio_std", 0.20),
        "scroll_jitter_mean": baseline.get("scroll_jitter_mean"),
        "regress_rate_mean": baseline.get("regress_rate_mean"),
        "para_dwell_median_s": baseline.get("para_dwell_median_s"),
        "para_dwell_iqr_s": baseline.get("para_dwell_iqr_s"),
        "scroll_velocity_norm_mean": baseline.get("scroll_velocity_norm_mean"),
        "calibration_duration_seconds": baseline.get("calibration_duration_seconds"),
    }

    return StatePacket(
        session_id=session_id,
        user_id=session.user_id,
        computed_at=now,
        baseline_snapshot=baseline_snapshot,
        window_features=feat_dict,
        z_scores=z_dict,
        drift=StatePacketDrift(
            drift_ema=result.drift_ema,
            drift_level=result.drift_level,
            beta_ema=result.beta_ema,
            disruption_score=result.disruption_score,
            engagement_score=result.engagement_score,
            confidence=result.confidence,
        ),
        flags=StatePacketFlags(
            baseline_valid=baseline_valid,
            baseline_wpm_valid=wpm_eff > 0,
            telemetry_fault_rate=features.telemetry_fault_rate,
            scroll_capture_fault_rate=features.scroll_capture_fault_rate,
            paragraph_missing_fault_rate=features.paragraph_missing_fault_rate,
            quality_confidence_mult=features.quality_confidence_mult,
        ),
        context=StatePacketContext(
            progress_ratio=last_batch.get("viewport_progress_ratio", 0.0),
            current_paragraph_id=last_batch.get("current_paragraph_id"),
            pace_ratio=result.pace_ratio,
            pace_available=features.pace_available,
            progress_velocity=features.progress_velocity,
        ),
    )
