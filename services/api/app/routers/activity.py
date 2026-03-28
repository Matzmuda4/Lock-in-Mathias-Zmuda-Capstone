from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import ActivityEvent, Session, User
from app.db.session import get_db
from app.schemas.activity import (
    ActivityBatchCreate,
    ActivityBatchResponse,
    ActivityEventCreate,
    ActivityEventResponse,
)
from app.routers.drift import _recompute_and_save as _recompute_drift

router = APIRouter(prefix="/activity", tags=["activity"])

_VALID_EVENT_TYPES = frozenset(
    {
        "scroll_forward",
        "scroll_backward",
        "idle",
        "blur",
        "focus",
        "heartbeat",
        # Explicit user interaction with the adaptive side panel.
        # Signals that idle time in a window is intentional engagement,
        # not distraction — used by the drift model to dampen z_idle.
        "panel_interaction",
    }
)


@router.post("", response_model=ActivityEventResponse, status_code=status.HTTP_201_CREATED)
async def post_activity(
    body: ActivityEventCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ActivityEventResponse:
    """
    Ingest a single telemetry event for an active reading session.

    The client may supply its own `created_at` timestamp to support
    buffered / offline event delivery. If omitted, the server sets it to now.
    """
    if body.event_type not in _VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown event_type '{body.event_type}'. "
            f"Valid types: {sorted(_VALID_EVENT_TYPES)}",
        )

    result = await db.execute(
        select(Session).where(
            Session.id == body.session_id,
            Session.user_id == current_user.id,
            Session.status.in_(["active", "paused"]),
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or does not belong to the current user",
        )

    event = ActivityEvent(
        user_id=current_user.id,
        session_id=body.session_id,
        event_type=body.event_type,
        payload=body.payload,
        created_at=body.created_at or datetime.now(timezone.utc),
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return ActivityEventResponse.model_validate(event)


@router.post(
    "/batch",
    response_model=ActivityBatchResponse,
    status_code=status.HTTP_201_CREATED,
)
async def post_activity_batch(
    body: ActivityBatchCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ActivityBatchResponse:
    """
    Ingest one aggregated 2-second telemetry batch from the reader.

    Validates that the session is active and belongs to the caller, then
    inserts a single row into the activity_events hypertable with
    event_type="telemetry_batch" and the full payload as JSONB.

    Only active sessions are accepted (not paused) — we do not collect
    telemetry while the reader is paused.
    """
    result = await db.execute(
        select(Session).where(
            Session.id == body.session_id,
            Session.user_id == current_user.id,
            Session.status == "active",
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active session not found or does not belong to the current user",
        )

    # ── B1/B2: Data quality guardrails ────────────────────────────────────────
    # Apply corrections and tag faults before storage so the drift model
    # can apply confidence penalties on retrieval.
    _WINDOW_S = 2.0
    _PROGRESS_JUMP_THRESH = 0.20

    raw_idle = body.idle_seconds
    telemetry_fault = raw_idle > _WINDOW_S
    # Server-side clamp — idle must be 0..WINDOW_S
    clamped_idle = min(raw_idle, _WINDOW_S)

    scroll_abs = body.scroll_delta_abs_sum
    # scroll_capture_fault CANNOT be reliably computed from a single batch
    # because we would need the previous batch's progress_ratio to know whether
    # "no scroll events" is a normal reading pause or a listener failure.
    # Computing it here with only the current batch would create a massive false-
    # positive rate: every pause to read is flagged as a capture failure.
    # Instead, features.py computes this properly from consecutive batch pairs
    # in the rolling window, where the full context is available.
    scroll_capture_fault = False

    paragraph_missing_fault = body.current_paragraph_id is None

    # Store the full batch payload in JSONB for downstream analysis.
    # One row per 2-second window — low overhead on the hypertable.
    payload = body.model_dump(exclude={"session_id", "client_timestamp"})
    # Overwrite idle_seconds with the server-clamped value
    payload["idle_seconds"] = clamped_idle
    # Attach quality flags so feature extraction + model can apply penalties
    payload["telemetry_fault"] = telemetry_fault
    payload["scroll_capture_fault"] = scroll_capture_fault
    payload["paragraph_missing_fault"] = paragraph_missing_fault

    event = ActivityEvent(
        user_id=current_user.id,
        session_id=body.session_id,
        event_type="telemetry_batch",
        payload=payload,
        created_at=datetime.now(timezone.utc),
    )
    db.add(event)
    # Flush so the new event is visible inside this transaction for the drift query
    await db.flush()

    # Use a SAVEPOINT so that drift errors cannot abort the main transaction.
    # If drift recompute fails (e.g. missing column, model error), the savepoint
    # is released/rolled back and the telemetry event is still committed.
    try:
        async with db.begin_nested():
            await _recompute_drift(session, db)
    except Exception:  # noqa: BLE001
        pass  # drift failure must not prevent telemetry storage

    await db.commit()
    await db.refresh(event)
    return ActivityBatchResponse.model_validate(event)
