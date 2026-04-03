"""
Classification endpoints.

GET /classifier/health
    → whether the RF model is loaded and which implementation is active.

GET /sessions/{session_id}/attentional-state
    → latest cached ClassificationResult for the session.
    → 404 if no full-window packet has been classified yet.
    → 503 if the classifier is disabled.

GET /sessions/{session_id}/attentional-state/history  [placeholder]
    → will return the persisted trajectory once DB tables are added.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import Session, SessionAttentionalState, User
from app.db.session import get_db
from app.services.classifier.registry import get_cache, get_classifier, is_available

log = logging.getLogger(__name__)
router = APIRouter(tags=["classification"])


# ── Response schemas ──────────────────────────────────────────────────────────


class ClassifierHealthResponse(BaseModel):
    available: bool
    classifier_type: Optional[str] = None
    model_reachable: Optional[bool] = None
    cache_size: int = 0
    reason: Optional[str] = None


class AttentionalStateResponse(BaseModel):
    """
    Distribution output consumed by the frontend top bar and, in future,
    by the intervention LLM prompt builder.

    Distribution values are calibrated probabilities in [0, 1] summing to 1.0,
    produced by the RF + isotonic calibration pipeline.  They represent the
    model's posterior belief over the four attentional states and are suitable
    for direct use as soft-label inputs to an intervention generation model.
    """
    session_id: int
    packet_seq: int
    classified_at: Any          # datetime — Any to avoid TZ serialisation issues
    distribution: dict[str, float]   # float probabilities summing to 1.0
    primary_state: str
    confidence: float                # max(distribution.values())
    rationale: str
    latency_ms: int
    parse_ok: bool

    # Intervention-ready context — available now, consumed by future LLM
    intervention_context: Optional[dict[str, Any]] = None


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_owned_session(
    session_id: int,
    user_id: int,
    db: AsyncSession,
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return row


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/classifier/health", response_model=ClassifierHealthResponse)
async def classifier_health() -> ClassifierHealthResponse:
    """
    Check whether the classifier is configured and ready.

    For the RF classifier this means the pkl file was found and loaded at
    startup.  The check is synchronous (no network call required) and
    completes in < 1 ms.
    """
    cache = get_cache()

    if not is_available():
        return ClassifierHealthResponse(
            available=False,
            reason=(
                "no_classifier_configured — set CLASSIFY_ENABLED=true "
                "and CLASSIFY_USE_RF=true in .env"
            ),
            cache_size=len(cache),
        )

    clf = get_classifier()
    clf_type = type(clf).__name__

    try:
        reachable = await clf.health_check()  # type: ignore[union-attr]
    except Exception as exc:
        log.warning("Classifier health check error: %s", exc)
        reachable = False

    return ClassifierHealthResponse(
        available=reachable,
        classifier_type=clf_type,
        model_reachable=reachable,
        cache_size=len(cache),
        reason=None if reachable else "model_not_ready",
    )


@router.get(
    "/sessions/{session_id}/attentional-state",
    response_model=AttentionalStateResponse,
)
async def get_attentional_state(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AttentionalStateResponse:
    """
    Return the most recent attentional-state classification for a session.

    Results are served from the in-memory cache populated every ~10 seconds
    when a full-window state packet is written.

    Returns 404 when:
    - The session is very new (< 16 telemetry batches, ~30 s).
    - The classifier is enabled but hasn't fired yet.

    Returns 503 when the classifier is disabled (CLASSIFY_ENABLED=false).

    The response includes intervention_context — a structured dict ready for
    the future intervention LLM prompt builder — so the frontend can render
    the distribution and the LLM can consume the context without extra calls.
    """
    await _get_owned_session(session_id, current_user.id, db)

    cached = get_cache().get(session_id)

    if cached is None:
        if not is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Classifier is not enabled. "
                    "Set CLASSIFY_ENABLED=true and CLASSIFY_USE_RF=true in .env."
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No classification available yet for this session. "
                "The first classification fires after ~30 seconds (16 telemetry batches)."
            ),
        )

    r = cached.result
    distribution = {
        "focused":            round(r.focused, 4),
        "drifting":           round(r.drifting, 4),
        "hyperfocused":       round(r.hyperfocused, 4),
        "cognitive_overload": round(r.cognitive_overload, 4),
    }
    confidence = max(distribution.values())

    return AttentionalStateResponse(
        session_id=session_id,
        packet_seq=cached.packet_seq,
        classified_at=cached.classified_at,
        distribution=distribution,
        primary_state=r.primary_state,
        confidence=round(confidence, 4),
        rationale=r.rationale,
        latency_ms=r.latency_ms,
        parse_ok=r.parse_ok,
        intervention_context=r.as_intervention_context(),
    )


class AttentionalStateRecord(BaseModel):
    """
    One row from session_attentional_states as returned by the history endpoint.

    This is the primary document the intervention LLM consumes.  Each record
    is self-contained — the intervention_context JSONB carries all signals
    needed to choose a tier and generate a response:

      primary_state, confidence, distribution, ambiguous,
      drift_level, drift_ema, packet_seq, session_id.
    """
    session_id: int
    created_at: Any
    packet_seq: int
    primary_state: str
    confidence: float
    distribution: dict[str, float]
    drift_level: float
    drift_ema: float
    ambiguous: bool
    intervention_context: dict[str, Any]
    latency_ms: int
    parse_ok: bool


class AttentionalStateHistoryResponse(BaseModel):
    session_id: int
    total_records: int
    records: list[AttentionalStateRecord]

    # Convenience summary for the intervention LLM:
    # How many of the last N packets were in each state?
    state_counts: dict[str, int]

    # Sustained-state flag: True when all records share the same primary_state.
    # Useful as a quick "is the user stuck in drift?" signal.
    sustained: bool
    sustained_state: Optional[str] = None


@router.get(
    "/sessions/{session_id}/attentional-state/history",
    response_model=AttentionalStateHistoryResponse,
)
async def get_attentional_state_history(
    session_id: int,
    limit: int = Query(default=10, ge=1, le=100, description="Number of recent records (newest first)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AttentionalStateHistoryResponse:
    """
    Return the last `limit` attentional-state classifications for a session,
    newest first.

    **Primary use case: Intervention LLM context.**
    Pass the returned `records` to the intervention prompt builder.  Each
    record's `intervention_context` is a self-contained JSONB document —
    no further queries or joins are needed.

    **Convenience fields:**
    - `state_counts`   — frequency of each state in the returned window.
    - `sustained`      — True when all records share the same primary_state
                         (e.g. the user has been drifting for the last N packets).
    - `sustained_state`— which state is sustained (null when sustained=False).

    Returns 404 when no classifications have been persisted yet (< 30 s into
    the session, or the classifier is disabled).
    """
    await _get_owned_session(session_id, current_user.id, db)

    rows_result = await db.execute(
        select(SessionAttentionalState)
        .where(SessionAttentionalState.session_id == session_id)
        .order_by(SessionAttentionalState.created_at.desc())
        .limit(limit)
    )
    rows = list(rows_result.scalars().all())

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No attentional-state history available yet for this session. "
                "History is populated every ~10 seconds after the first 30 seconds."
            ),
        )

    records: list[AttentionalStateRecord] = []
    state_counts: dict[str, int] = {
        "focused": 0, "drifting": 0,
        "hyperfocused": 0, "cognitive_overload": 0,
    }

    for row in rows:
        ctx = row.intervention_context or {}
        dist = ctx.get("distribution", {
            "focused":            row.prob_focused,
            "drifting":           row.prob_drifting,
            "hyperfocused":       row.prob_hyperfocused,
            "cognitive_overload": row.prob_cognitive_overload,
        })
        state_counts[row.primary_state] = state_counts.get(row.primary_state, 0) + 1
        records.append(AttentionalStateRecord(
            session_id=row.session_id,
            created_at=row.created_at,
            packet_seq=row.packet_seq,
            primary_state=row.primary_state,
            confidence=row.confidence,
            distribution=dist,
            drift_level=row.drift_level,
            drift_ema=row.drift_ema,
            ambiguous=ctx.get("ambiguous", False),
            intervention_context=ctx,
            latency_ms=row.latency_ms,
            parse_ok=row.parse_ok,
        ))

    unique_states = {r.primary_state for r in records}
    sustained = len(unique_states) == 1
    sustained_state = records[0].primary_state if sustained else None

    return AttentionalStateHistoryResponse(
        session_id=session_id,
        total_records=len(records),
        records=records,
        state_counts=state_counts,
        sustained=sustained,
        sustained_state=sustained_state,
    )
