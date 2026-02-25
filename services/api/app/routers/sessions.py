from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import Document, Session, User
from app.db.session import get_db
from app.schemas.sessions import SessionCreate, SessionListResponse, SessionResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


async def _get_owned_session(session_id: int, user_id: int, db: AsyncSession) -> Session:
    result = await db.execute(
        select(Session).where(Session.id == session_id, Session.user_id == user_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


@router.post("/start", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def start_session(
    body: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Create a new reading session for the authenticated user."""
    doc_result = await db.execute(
        select(Document).where(
            Document.id == body.document_id, Document.user_id == current_user.id
        )
    )
    if doc_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    session = Session(
        user_id=current_user.id,
        document_id=body.document_id,
        name=body.name.strip(),
        mode=body.mode,
        status="active",
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return SessionResponse.model_validate(session)


@router.post("/{session_id}/pause", response_model=SessionResponse)
async def pause_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    session = await _get_owned_session(session_id, current_user.id, db)
    if session.status != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause a session with status '{session.status}'",
        )
    session.status = "paused"
    await db.commit()
    await db.refresh(session)
    return SessionResponse.model_validate(session)


@router.post("/{session_id}/resume", response_model=SessionResponse)
async def resume_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    session = await _get_owned_session(session_id, current_user.id, db)
    if session.status != "paused":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot resume a session with status '{session.status}'",
        )
    session.status = "active"
    await db.commit()
    await db.refresh(session)
    return SessionResponse.model_validate(session)


@router.post("/{session_id}/end", response_model=SessionResponse)
async def end_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    session = await _get_owned_session(session_id, current_user.id, db)
    if session.status == "ended":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session has already ended",
        )
    now = datetime.now(timezone.utc)
    session.status = "ended"
    session.ended_at = now
    session.duration_seconds = max(0, int((now - session.started_at).total_seconds()))
    await db.commit()
    await db.refresh(session)
    return SessionResponse.model_validate(session)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionListResponse:
    result = await db.execute(
        select(Session).where(Session.user_id == current_user.id)
    )
    sessions = result.scalars().all()
    return SessionListResponse(
        sessions=[SessionResponse.model_validate(s) for s in sessions],
        total=len(sessions),
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    session = await _get_owned_session(session_id, current_user.id, db)
    return SessionResponse.model_validate(session)
