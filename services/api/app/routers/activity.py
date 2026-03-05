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

router = APIRouter(prefix="/activity", tags=["activity"])

_VALID_EVENT_TYPES = frozenset(
    {
        "scroll_forward",
        "scroll_backward",
        "idle",
        "blur",
        "focus",
        "heartbeat",
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
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active session not found or does not belong to the current user",
        )

    # Store the full batch payload in JSONB for downstream analysis.
    # One row per 2-second window — low overhead on the hypertable.
    payload = body.model_dump(exclude={"session_id", "client_timestamp"})

    event = ActivityEvent(
        user_id=current_user.id,
        session_id=body.session_id,
        event_type="telemetry_batch",
        payload=payload,
        created_at=datetime.now(timezone.utc),
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return ActivityBatchResponse.model_validate(event)
