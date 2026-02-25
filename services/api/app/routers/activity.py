from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import ActivityEvent, Session, User
from app.db.session import get_db
from app.schemas.activity import ActivityEventCreate, ActivityEventResponse

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
