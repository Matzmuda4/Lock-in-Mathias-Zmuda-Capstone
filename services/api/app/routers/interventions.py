"""
interventions.py — Intervention engine API endpoints.

POST /sessions/{id}/interventions/trigger
    Full LLM pipeline: assembles prompt → calls Ollama → logs to DB.
    Returns the intervention payload (or null when tier='none').

POST /sessions/{id}/interventions/manual
    Dev/test endpoint: bypasses the LLM entirely.
    Accepts a type + optional content override and returns a canned payload.
    Intended for testing every intervention type one-by-one.

GET  /sessions/{id}/interventions/pending
    Returns the latest unfired/unacknowledged intervention for the session.
    Frontend polls this every 10 s; 204 when nothing is pending.

POST /sessions/{id}/interventions/{intervention_id}/acknowledge
    Marks an intervention as seen by the user.  Clears the pending slot.

GET  /intervention-engine/health
    Liveness check — confirms the Ollama model is reachable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import Intervention, Session, SessionAttentionalState, User, UserBaseline
from app.db.session import get_db
from app.services.intervention.engine import (
    InterventionResult,
    get_cooldown_tracker,
    get_engine,
)
from app.services.intervention.prompt import build_intervention_input, build_raw_chatml_prompt
from app.services.intervention.templates import MANUAL_TEMPLATES

router = APIRouter(tags=["interventions"])
log = logging.getLogger(__name__)

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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return row


def _elapsed_minutes(session: Session) -> float:
    now     = datetime.now(timezone.utc)
    started = session.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (now - started).total_seconds()) / 60.0


async def _fetch_attentional_window(
    session_id: int,
    db: AsyncSession,
    limit: int = 3,
) -> list[SessionAttentionalState]:
    """Fetch the most recent `limit` attentional-state rows, oldest first."""
    result = await db.execute(
        select(SessionAttentionalState)
        .where(SessionAttentionalState.session_id == session_id)
        .order_by(SessionAttentionalState.created_at.desc())
        .limit(limit)
    )
    rows = list(result.scalars().all())
    return list(reversed(rows))   # oldest → newest


def _build_attentional_window_list(
    rows: list[SessionAttentionalState],
) -> list[dict[str, Any]]:
    return [
        {
            "primary_state": row.primary_state,
            "confidence":    round(row.confidence, 4),
            "distribution": {
                "focused":            round(row.prob_focused or 0.0, 4),
                "drifting":           round(row.prob_drifting or 0.0, 4),
                "hyperfocused":       round(row.prob_hyperfocused or 0.0, 4),
                "cognitive_overload": round(row.prob_cognitive_overload or 0.0, 4),
            },
        }
        for row in rows
    ]


def _build_drift_progression(rows: list[SessionAttentionalState]) -> dict[str, Any]:
    return {
        "drift_level":    [round(r.drift_level or 0.0, 4) for r in rows],
        "engagement_score": [
            round((r.intervention_context or {}).get("engagement_score", 0.0), 4)
            for r in rows
        ],
        "drift_ema": round(rows[-1].drift_ema if rows else 0.0, 4),
    }


def _build_user_baseline(baseline_json: dict[str, Any]) -> dict[str, Any]:
    return {
        "wpm_effective":       round(float(baseline_json.get("wpm_effective") or
                                           baseline_json.get("wpm_gross") or 0.0), 1),
        "idle_ratio_mean":     round(float(baseline_json.get("idle_ratio_mean", 0.3)), 4),
        "regress_rate_mean":   round(float(baseline_json.get("regress_rate_mean", 0.03)), 4),
        "para_dwell_median_s": round(float(baseline_json.get("para_dwell_median_s") or
                                           baseline_json.get("paragraph_dwell_mean") or 15.0), 1),
    }


def _build_last_intervention(
    session_id: int,
    cooldown_tracker: Any,
    last_type: str | None,
    last_tier: str | None,
) -> dict[str, Any] | None:
    secs = cooldown_tracker.seconds_since_last(session_id)
    if secs is None or last_type is None:
        return None
    return {"type": last_type, "tier": last_tier or "subtle", "seconds_ago": secs}


async def _log_intervention(
    session_id: int,
    result: InterventionResult,
    db: AsyncSession,
) -> Intervention:
    row = Intervention(
        session_id = session_id,
        type       = result.type or "none",
        intensity  = result.tier,
        payload    = {
            "intervene":   result.intervene,
            "tier":        result.tier,
            "type":        result.type,
            "content":     result.content,
            "raw_json":    result.raw_json,
            "latency_ms":  result.latency_ms,
        },
    )
    db.add(row)
    await db.flush()   # populate row.id before commit
    return row


# ── Response schemas ──────────────────────────────────────────────────────────


class InterventionPayload(BaseModel):
    """Shared response shape for trigger, manual, and pending endpoints."""
    intervention_id: Optional[int] = None
    session_id:      int
    intervene:       bool
    tier:            str
    type:            Optional[str]
    content:         Optional[dict[str, Any]]
    latency_ms:      int
    cooldown_status: str
    fired_at:        Optional[datetime] = None


class ManualTriggerRequest(BaseModel):
    """Body for POST /interventions/manual."""
    type:    str
    tier:    str = "moderate"
    content: Optional[dict[str, Any]] = None   # override template if supplied


class EngineHealthResponse(BaseModel):
    available:   bool
    model:       str
    reason:      Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/intervention-engine/health", response_model=EngineHealthResponse)
async def engine_health() -> EngineHealthResponse:
    """Check that the Ollama lockin-intervention model is reachable."""
    engine = get_engine()
    ok     = await engine.health_check()
    return EngineHealthResponse(
        available = ok,
        model     = engine._model,
        reason    = None if ok else "Ollama model not registered or Ollama not running",
    )


@router.post(
    "/sessions/{session_id}/interventions/trigger",
    response_model=InterventionPayload,
    status_code=status.HTTP_200_OK,
)
async def trigger_intervention(
    session_id:   int,
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> InterventionPayload:
    """
    Full LLM pipeline trigger.

    1. Fetches the last 3 attentional-state rows from DB.
    2. Builds the ChatML prompt via prompt.py.
    3. Calls the intervention LLM.
    4. If LLM says intervene=True AND cooldown is clear → logs to DB.
    5. Returns the full payload regardless.

    Returns 422 when fewer than 1 attentional-state row exists for the session
    (i.e. the RF classifier hasn't fired yet — call again after 30 s).
    """
    session   = await _get_owned_session(session_id, current_user.id, db)
    cooldown  = get_cooldown_tracker()
    engine    = get_engine()

    # ── 1. Fetch context ──────────────────────────────────────────────────────
    rows = await _fetch_attentional_window(session_id, db)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No attentional-state data yet for this session. "
                "The first classification fires after ~30 seconds."
            ),
        )

    baseline_row = (
        await db.execute(
            select(UserBaseline).where(UserBaseline.user_id == session.user_id)
        )
    ).scalar_one_or_none()
    baseline_json: dict = baseline_row.baseline_json if baseline_row else {}

    # ── 2. Build prompt inputs ────────────────────────────────────────────────
    cooldown_status = cooldown.status(session_id)
    elapsed_min     = _elapsed_minutes(session)

    # Recover last fired type/tier from most recent logged intervention
    last_row = (
        await db.execute(
            select(Intervention)
            .where(Intervention.session_id == session_id)
            .order_by(Intervention.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    last_type = last_row.type if last_row else None
    last_tier = last_row.intensity if last_row else None

    # Text window comes from the most recent attentional-state context
    last_ctx         = rows[-1].intervention_context or {}
    text_window      = last_ctx.get("text_window", [])
    chunk_index      = last_ctx.get("current_chunk_index")

    input_dict = build_intervention_input(
        elapsed_minutes        = elapsed_min,
        attentional_window     = _build_attentional_window_list(rows),
        drift_progression      = _build_drift_progression(rows),
        user_baseline          = _build_user_baseline(baseline_json),
        text_window            = text_window,
        current_paragraph_index= chunk_index,
        xp                     = 0,          # Phase 2: gamification XP tracking
        badges_earned          = [],         # Phase 2: badge tracking
        last_intervention      = _build_last_intervention(
                                     session_id, cooldown, last_type, last_tier),
        cooldown_status        = cooldown_status,
    )
    prompt = build_raw_chatml_prompt(input_dict)

    # ── 3. Call the LLM ───────────────────────────────────────────────────────
    result = await engine.call(prompt)

    # ── 4. Conditionally fire ─────────────────────────────────────────────────
    intervention_id: int | None = None
    fired_at: datetime | None   = None

    if result.is_actionable() and cooldown_status == "clear":
        logged   = await _log_intervention(session_id, result, db)
        await db.commit()
        cooldown.mark_fired(session_id)
        intervention_id = logged.id
        fired_at        = logged.created_at
        log.info(
            "[interventions] FIRED session=%d id=%d type=%s tier=%s",
            session_id, intervention_id, result.type, result.tier,
        )
    else:
        reason = "cooling" if cooldown_status == "cooling" else "not_actionable"
        log.info(
            "[interventions] SKIP session=%d reason=%s type=%s tier=%s",
            session_id, reason, result.type, result.tier,
        )

    return InterventionPayload(
        intervention_id = intervention_id,
        session_id      = session_id,
        intervene       = result.intervene and cooldown_status == "clear",
        tier            = result.tier,
        type            = result.type,
        content         = result.content,
        latency_ms      = result.latency_ms,
        cooldown_status = cooldown_status,
        fired_at        = fired_at,
    )


@router.post(
    "/sessions/{session_id}/interventions/manual",
    response_model=InterventionPayload,
    status_code=status.HTTP_200_OK,
)
async def manual_intervention(
    session_id:   int,
    body:         ManualTriggerRequest,
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> InterventionPayload:
    """
    Dev/test endpoint — bypasses the LLM entirely.

    Returns a canned payload for the requested intervention type so every
    intervention component can be tested independently before the full LLM
    pipeline is wired in.

    Pass ``content`` to override the canned template for that type.
    The intervention IS logged to the DB and cooldown IS updated so the
    full data-flow can be verified end-to-end.
    """
    session  = await _get_owned_session(session_id, current_user.id, db)
    cooldown = get_cooldown_tracker()

    itype = body.type.lower()
    tier  = body.tier.lower()

    from app.services.intervention.engine import VALID_TYPES, VALID_TIERS
    if itype not in VALID_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown intervention type '{itype}'. Valid: {sorted(VALID_TYPES)}",
        )
    if tier not in VALID_TIERS - {"none"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown tier '{tier}'. Valid: {sorted(VALID_TIERS - {'none'})}",
        )

    content = body.content or MANUAL_TEMPLATES.get(itype, {})

    result = InterventionResult(
        intervene  = True,
        tier       = tier,
        type       = itype,
        content    = content,
        raw_json   = "{}",   # no LLM call
        latency_ms = 0,
        parse_ok   = True,
    )

    logged   = await _log_intervention(session_id, result, db)
    await db.commit()
    cooldown.mark_fired(session_id)

    log.info(
        "[interventions] MANUAL session=%d id=%d type=%s tier=%s",
        session_id, logged.id, itype, tier,
    )

    return InterventionPayload(
        intervention_id = logged.id,
        session_id      = session_id,
        intervene       = True,
        tier            = tier,
        type            = itype,
        content         = content,
        latency_ms      = 0,
        cooldown_status = cooldown.status(session_id),
        fired_at        = logged.created_at,
    )


@router.get(
    "/sessions/{session_id}/interventions/pending",
    response_model=Optional[InterventionPayload],
    status_code=status.HTTP_200_OK,
)
async def get_pending_intervention(
    session_id:   int,
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> InterventionPayload | None:
    """
    Return the most recent intervention for this session, or null.

    The frontend polls this every 10 s to discover new interventions without
    needing WebSockets.  The response includes the full content payload so
    the frontend can render the intervention immediately.

    Returns null (HTTP 200 with JSON ``null``) when no intervention has been
    fired yet, or when the cooldown is active.
    """
    await _get_owned_session(session_id, current_user.id, db)
    cooldown = get_cooldown_tracker()

    row = (
        await db.execute(
            select(Intervention)
            .where(Intervention.session_id == session_id)
            .order_by(Intervention.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    if row is None:
        return None

    payload = row.payload or {}
    return InterventionPayload(
        intervention_id = row.id,
        session_id      = session_id,
        intervene       = payload.get("intervene", True),
        tier            = row.intensity,
        type            = row.type,
        content         = payload.get("content"),
        latency_ms      = payload.get("latency_ms", 0),
        cooldown_status = cooldown.status(session_id),
        fired_at        = row.created_at,
    )


@router.post(
    "/sessions/{session_id}/interventions/{intervention_id}/acknowledge",
    response_class=Response,
    status_code=status.HTTP_204_NO_CONTENT,
)
async def acknowledge_intervention(
    session_id:      int,
    intervention_id: int,
    current_user:    User = Depends(get_current_user),
    db:              AsyncSession = Depends(get_db),
) -> Response:
    """
    Acknowledge that the user has seen/dismissed this intervention.

    Currently a no-op beyond ownership verification — logged for audit trail.
    Phase 2 will use this to record interaction time and dismiss inline UI.
    """
    await _get_owned_session(session_id, current_user.id, db)

    row = (
        await db.execute(
            select(Intervention).where(
                Intervention.id == intervention_id,
                Intervention.session_id == session_id,
            )
        )
    ).scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Intervention not found for this session",
        )

    log.info(
        "[interventions] ACKNOWLEDGED session=%d id=%d type=%s",
        session_id, intervention_id, row.type,
    )
    # Phase 2: update an acknowledged_at column here
    return Response(status_code=status.HTTP_204_NO_CONTENT)
