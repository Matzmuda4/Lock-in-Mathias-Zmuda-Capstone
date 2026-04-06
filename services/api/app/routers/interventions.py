"""
interventions.py — Intervention engine API endpoints.

GET  /intervention-engine/health
    Liveness check — confirms the Ollama model is reachable.

POST /sessions/{id}/interventions/trigger
    Full LLM pipeline: assembles prompt → calls Ollama → gate check → logs.

POST /sessions/{id}/interventions/manual
    Dev/test: bypasses the LLM entirely; fires a canned payload for any type.

GET  /sessions/{id}/interventions/active
    Returns all currently active (unacknowledged, non-expired) interventions.
    Frontend polls this every ~10 s to know what to display.

GET  /sessions/{id}/interventions/pending   [legacy alias for /active]
    Returns the single most-recent active intervention, or null.
    Kept for backward compatibility during frontend development.

POST /sessions/{id}/interventions/{id}/acknowledge
    User has dismissed an intervention.  Frees its slot in the tracker.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import Intervention, Session, SessionAttentionalState, User, UserBaseline
from app.db.session import get_db
from app.services.intervention.engine import (
    VALID_TIERS,
    VALID_TYPES,
    ActiveIntervention,
    InterventionResult,
    get_active_tracker,
    get_engine,
)
from app.services.intervention import rules as intervention_rules
from app.services.intervention.prompt import build_intervention_input, build_raw_chatml_prompt
from app.services.intervention.templates import MANUAL_TEMPLATES

router = APIRouter(tags=["interventions"])
log = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────

async def _get_owned_session(
    session_id: int,
    user_id:    int,
    db:         AsyncSession,
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
    db:         AsyncSession,
    limit:      int = 3,
) -> list[SessionAttentionalState]:
    """Fetch the most recent ``limit`` attentional-state rows, oldest first."""
    result = await db.execute(
        select(SessionAttentionalState)
        .where(SessionAttentionalState.session_id == session_id)
        .order_by(SessionAttentionalState.created_at.desc())
        .limit(limit)
    )
    return list(reversed(list(result.scalars().all())))


def _build_attentional_window_list(
    rows: list[SessionAttentionalState],
) -> list[dict[str, Any]]:
    return [
        {
            "primary_state": row.primary_state,
            "confidence":    round(row.confidence, 4),
            "drift_ema":     round(row.drift_ema or 0.0, 4),
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
        "drift_level": [round(r.drift_level or 0.0, 4) for r in rows],
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
    tracker:    Any,
    last_type:  str | None,
    last_tier:  str | None,
) -> dict[str, Any] | None:
    secs = tracker.seconds_since_last(session_id)
    if secs is None or last_type is None:
        return None
    return {"type": last_type, "tier": last_tier or "subtle", "seconds_ago": secs}


def _build_active_interventions_list(
    active: list[ActiveIntervention],
) -> list[dict[str, Any]]:
    """Convert tracker ActiveIntervention objects to prompt-ready dicts."""
    now = datetime.now(timezone.utc)
    return [
        {
            "type":           ai.itype,
            "tier":           ai.tier,
            "seconds_active": int((now - ai.fired_at).total_seconds()),
        }
        for ai in active
    ]


async def _log_intervention(
    session_id: int,
    result:     InterventionResult,
    db:         AsyncSession,
) -> Intervention:
    row = Intervention(
        session_id = session_id,
        type       = result.type or "none",
        intensity  = result.tier,
        payload    = {
            "intervene":  result.intervene,
            "tier":       result.tier,
            "type":       result.type,
            "content":    result.content,
            "raw_json":   result.raw_json,
            "latency_ms": result.latency_ms,
        },
    )
    db.add(row)
    await db.flush()
    return row


# ── Response schemas ──────────────────────────────────────────────────────────

class InterventionPayload(BaseModel):
    """Shared response shape for trigger, manual, active, and pending endpoints."""
    intervention_id: Optional[int]           = None
    session_id:      int
    intervene:       bool
    tier:            str
    type:            Optional[str]
    content:         Optional[dict[str, Any]]
    latency_ms:      int
    cooldown_status: str
    fired_at:        Optional[datetime]      = None


class ManualTriggerRequest(BaseModel):
    type:    str
    tier:    str                           = "moderate"
    content: Optional[dict[str, Any]]     = None


class EngineHealthResponse(BaseModel):
    available: bool
    model:     str
    reason:    Optional[str] = None


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
    current_user: User        = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> InterventionPayload:
    """
    Full LLM pipeline trigger.

    1. Fetch last 3 attentional-state rows — 422 if none exist yet.
    2. Build ChatML prompt with full session context.
    3. Call the intervention LLM.
    4. Gate check for the type the LLM returned.
    5. If gate is clear AND LLM says intervene=True → log to DB + mark fired.
    """
    session = await _get_owned_session(session_id, current_user.id, db)
    tracker = get_active_tracker()
    engine  = get_engine()

    # ── 1. Fetch attentional context ──────────────────────────────────────────
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

    # ── 2. Build prompt context ───────────────────────────────────────────────
    elapsed_min   = _elapsed_minutes(session)
    active_now    = tracker.active_for_session(session_id)
    primary_state = rows[-1].primary_state or "focused"
    gate_overall  = tracker.check(session_id, "re_engagement", primary_state)

    last_row = (
        await db.execute(
            select(Intervention)
            .where(Intervention.session_id == session_id)
            .order_by(Intervention.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    last_type = last_row.type      if last_row else None
    last_tier = last_row.intensity if last_row else None

    last_ctx    = rows[-1].intervention_context or {}
    text_window = last_ctx.get("text_window", [])
    chunk_index = last_ctx.get("current_chunk_index")

    input_dict = build_intervention_input(
        elapsed_minutes         = elapsed_min,
        attentional_window      = _build_attentional_window_list(rows),
        drift_progression       = _build_drift_progression(rows),
        user_baseline           = _build_user_baseline(baseline_json),
        text_window             = text_window,
        current_paragraph_index = chunk_index,
        xp                      = 0,
        badges_earned           = [],
        last_intervention       = _build_last_intervention(
                                      session_id, tracker, last_type, last_tier),
        cooldown_status         = gate_overall.cooldown_status,
        active_interventions    = _build_active_interventions_list(active_now),
    )
    prompt = build_raw_chatml_prompt(input_dict)

    # ── 3. Call LLM ───────────────────────────────────────────────────────────
    result = await engine.call(prompt)
    log.info(
        "[interventions] LLM session=%d type=%s tier=%s intervene=%s",
        session_id, result.type, result.tier, result.intervene,
    )

    # ── 4. Gate check for the type the LLM returned ───────────────────────────
    intervention_id: int | None      = None
    fired_at:        datetime | None = None
    final_status = "not_actionable"

    if result.is_actionable():
        decision = tracker.check(session_id, result.type, primary_state)  # type: ignore[arg-type]
        final_status = decision.reason

        if decision.allowed:
            # Attach reading position to section_summary so the frontend can
            # render the card inline at the right chunk in the document.
            if result.type == "section_summary" and chunk_index is not None:
                content_with_pos = dict(result.content or {})
                content_with_pos["_chunk_index"] = chunk_index
                result = result.__class__(
                    intervene  = result.intervene,
                    tier       = result.tier,
                    type       = result.type,
                    content    = content_with_pos,
                    raw_json   = result.raw_json,
                    latency_ms = result.latency_ms,
                    parse_ok   = result.parse_ok,
                )
            logged  = await _log_intervention(session_id, result, db)
            await db.commit()
            tracker.mark_fired(session_id, logged.id, result.type, result.tier)  # type: ignore[arg-type]
            intervention_id = logged.id
            fired_at        = logged.created_at
            log.info(
                "[interventions] FIRED session=%d id=%d type=%s tier=%s",
                session_id, intervention_id, result.type, result.tier,
            )
        else:
            log.info(
                "[interventions] SKIP session=%d reason=%s type=%s tier=%s",
                session_id, decision.reason, result.type, result.tier,
            )

    # ── 5. Backend supplementary rules ───────────────────────────────────────
    # chime and text_reformat are rarely selected by the LLM due to gradient
    # imbalance during training.  These lightweight rules fire them when the
    # signals clearly call for them, independent of what the LLM decided.
    #
    # chime  — fires as a supplement alongside the LLM result (INSTANT type,
    #          no UI slot consumed).  Signals: drift rising noticeably.
    #
    # text_reformat — fires only when the LLM produced nothing actionable and
    #                 the student is clearly in sustained cognitive overload.

    await intervention_rules.maybe_fire_chime(
        session_id    = session_id,
        rows          = rows,
        primary_state = primary_state,
        tracker       = tracker,
        db            = db,
    )

    if not result.is_actionable():
        await intervention_rules.maybe_fire_text_reformat(
            session_id    = session_id,
            rows          = rows,
            primary_state = primary_state,
            tracker       = tracker,
            db            = db,
        )

    return InterventionPayload(
        intervention_id = intervention_id,
        session_id      = session_id,
        intervene       = intervention_id is not None,
        tier            = result.tier,
        type            = result.type,
        content         = result.content,
        latency_ms      = result.latency_ms,
        cooldown_status = final_status,
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
    current_user: User        = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> InterventionPayload:
    """
    Dev/test endpoint — bypasses the LLM and slot gate entirely.

    Every type can be fired freely here so each intervention component can be
    tested in isolation before the full LLM pipeline is wired in.
    The fire IS logged to DB and tracker IS updated for realistic state.
    """
    await _get_owned_session(session_id, current_user.id, db)
    tracker = get_active_tracker()

    itype = body.type.lower()
    tier  = body.tier.lower()

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
    result  = InterventionResult(
        intervene  = True,
        tier       = tier,
        type       = itype,
        content    = content,
        raw_json   = "{}",
        latency_ms = 0,
        parse_ok   = True,
    )

    logged = await _log_intervention(session_id, result, db)
    await db.commit()
    tracker.mark_fired(session_id, logged.id, itype, tier)

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
        cooldown_status = tracker.check(session_id, itype).reason,
        fired_at        = logged.created_at,
    )


@router.get(
    "/sessions/{session_id}/interventions/active",
    response_model=list[InterventionPayload],
    status_code=status.HTTP_200_OK,
)
async def get_active_interventions(
    session_id:   int,
    current_user: User        = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> list[InterventionPayload]:
    """
    Return all currently active (unacknowledged, non-expired) interventions.

    The frontend polls this endpoint to know what to display on screen.
    Empty list when nothing is active.

    Stale text prompts (> auto_dismiss_seconds unacknowledged) are silently
    expired before this query so the list is always up-to-date.
    """
    await _get_owned_session(session_id, current_user.id, db)
    tracker = get_active_tracker()

    active = tracker.active_for_session(session_id)
    if not active:
        return []

    ids = [ai.intervention_id for ai in active]
    rows_result = await db.execute(
        select(Intervention).where(Intervention.id.in_(ids))
    )
    rows_by_id = {row.id: row for row in rows_result.scalars().all()}

    payloads: list[InterventionPayload] = []
    for ai in active:
        row = rows_by_id.get(ai.intervention_id)
        if row is None:
            continue
        payload_data = row.payload or {}
        payloads.append(InterventionPayload(
            intervention_id = row.id,
            session_id      = session_id,
            intervene       = True,
            tier            = row.intensity,
            type            = row.type,
            content         = payload_data.get("content"),
            latency_ms      = payload_data.get("latency_ms", 0),
            cooldown_status = "active",
            fired_at        = row.created_at,
        ))
    return payloads


@router.get(
    "/sessions/{session_id}/interventions/pending",
    response_model=Optional[InterventionPayload],
    status_code=status.HTTP_200_OK,
)
async def get_pending_intervention(
    session_id:   int,
    current_user: User        = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> InterventionPayload | None:
    """
    Legacy alias — returns the single most-recent active intervention or null.

    Prefer /active for new frontend code (returns the full list).
    """
    active_list = await get_active_interventions(
        session_id=session_id, current_user=current_user, db=db,
    )
    if not active_list:
        tracker = get_active_tracker()
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
        payload_data = row.payload or {}
        return InterventionPayload(
            intervention_id = row.id,
            session_id      = session_id,
            intervene       = payload_data.get("intervene", True),
            tier            = row.intensity,
            type            = row.type,
            content         = payload_data.get("content"),
            latency_ms      = payload_data.get("latency_ms", 0),
            cooldown_status = tracker.status(session_id),
            fired_at        = row.created_at,
        )
    return sorted(active_list, key=lambda p: p.fired_at or datetime.min.replace(tzinfo=timezone.utc))[-1]


@router.post(
    "/sessions/{session_id}/interventions/{intervention_id}/acknowledge",
    response_class=Response,
    status_code=status.HTTP_204_NO_CONTENT,
)
async def acknowledge_intervention(
    session_id:      int,
    intervention_id: int,
    current_user:    User        = Depends(get_current_user),
    db:              AsyncSession = Depends(get_db),
) -> Response:
    """
    User has dismissed/acknowledged this intervention.

    Frees the type's slot in the ActiveInterventionTracker so a new
    intervention of the same type can fire on the next window.

    For break_suggestion: starts the 5-minute post-break cooldown.
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

    tracker = get_active_tracker()
    tracker.acknowledge(session_id, row.type)

    log.info(
        "[interventions] ACKNOWLEDGED session=%d id=%d type=%s",
        session_id, intervention_id, row.type,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
