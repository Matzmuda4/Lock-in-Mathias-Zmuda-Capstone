"""
Classification endpoints — Phase 9.

GET /classifier/health
    → whether the Ollama model is reachable and which implementation is active.

GET /sessions/{session_id}/attentional-state
    → latest cached ClassificationResult for the session (no DB write yet).

GET /sessions/{session_id}/attentional-state/history  [placeholder]
    → will return the persisted trajectory once DB tables are added.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import Session, User
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
    session_id: int
    packet_seq: int
    classified_at: Any          # datetime — Any to avoid TZ serialisation issues
    distribution: dict[str, int]
    primary_state: str
    rationale: str
    latency_ms: int
    parse_ok: bool


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
    Check whether the classifier is configured and whether the underlying
    Ollama model is reachable.

    Returns available=False with a reason when the classifier is not wired
    (CLASSIFY_ENABLED=false) or when Ollama is unreachable.
    """
    cache = get_cache()

    if not is_available():
        return ClassifierHealthResponse(
            available=False,
            reason="no_classifier_configured — set CLASSIFY_ENABLED=true in .env",
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
        reason=None if reachable else "ollama_unreachable",
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
    when a state packet is written.  Returns 404 if no classification has
    been produced yet (e.g. the session is very new or the classifier is off).

    Note: no DB persistence yet.  Once DB tables are migrated, this endpoint
    will fall back to the DB when the cache entry is missing.
    """
    # Ownership check — user can only read their own sessions
    await _get_owned_session(session_id, current_user.id, db)

    cached = get_cache().get(session_id)
    if cached is None:
        if not is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Classifier is not enabled. "
                    "Set CLASSIFY_ENABLED=true in .env and restart the API."
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No classification available yet for this session. "
                "Classifications are produced every ~10 seconds after the session starts."
            ),
        )

    r = cached.result
    return AttentionalStateResponse(
        session_id=session_id,
        packet_seq=cached.packet_seq,
        classified_at=cached.classified_at,
        distribution={
            "focused":            r.focused,
            "drifting":           r.drifting,
            "hyperfocused":       r.hyperfocused,
            "cognitive_overload": r.cognitive_overload,
        },
        primary_state=r.primary_state,
        rationale=r.rationale,
        latency_ms=r.latency_ms,
        parse_ok=r.parse_ok,
    )


@router.get("/sessions/{session_id}/attentional-state/history")
async def get_attentional_state_history(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Placeholder for the full trajectory endpoint.

    Will query session_attentional_states once the DB table is migrated.
    Returns an informational response until then.
    """
    await _get_owned_session(session_id, current_user.id, db)
    return {
        "session_id": session_id,
        "message": (
            "History endpoint is not yet available. "
            "DB table migration is pending — use /attentional-state for the latest result."
        ),
        "records": [],
    }
