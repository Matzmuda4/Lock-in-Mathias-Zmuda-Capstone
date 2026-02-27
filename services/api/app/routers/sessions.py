from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import Document, DocumentAsset, DocumentChunk, DocumentParseJob, Session, User
from app.db.session import get_db
from app.schemas.parsing import AssetSummary, ChunkResponse, SessionReaderResponse
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


_TERMINAL_STATUSES = frozenset({"ended", "completed"})


def _close_session(session: Session, final_status: str) -> None:
    """Shared logic for both /end and /complete."""
    now = datetime.now(timezone.utc)
    session.status = final_status
    session.ended_at = now
    session.duration_seconds = max(0, int((now - session.started_at).total_seconds()))


@router.post("/{session_id}/end", response_model=SessionResponse)
async def end_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Stop a session early (abandoned / timed-out)."""
    session = await _get_owned_session(session_id, current_user.id, db)
    if session.status in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is already terminal (status='{session.status}')",
        )
    _close_session(session, "ended")
    await db.commit()
    await db.refresh(session)
    return SessionResponse.model_validate(session)


@router.post("/{session_id}/complete", response_model=SessionResponse)
async def complete_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Mark a session as completed — the user finished reading intentionally.
    Tracked separately from /end to measure completion rates in the thesis."""
    session = await _get_owned_session(session_id, current_user.id, db)
    if session.status in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is already terminal (status='{session.status}')",
        )
    _close_session(session, "completed")
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


@router.get("/{session_id}/reader", response_model=SessionReaderResponse)
async def get_session_reader(
    session_id: int,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=30, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionReaderResponse:
    """
    Return everything the reader page needs in one call:
    session metadata, parse status, first page of chunks, and asset list.
    """
    session = await _get_owned_session(session_id, current_user.id, db)
    doc_id = session.document_id

    # Parse job status (may not exist if doc was uploaded before Phase 4)
    job_result = await db.execute(
        select(DocumentParseJob).where(DocumentParseJob.document_id == doc_id)
    )
    job = job_result.scalar_one_or_none()
    parse_status = job.status if job else "unknown"

    # Chunk count
    count_result = await db.execute(
        select(func.count()).where(DocumentChunk.document_id == doc_id)
    )
    total_chunks = count_result.scalar_one()

    # Paginated chunks
    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.chunk_index)
        .offset(offset)
        .limit(limit)
    )
    chunks = chunks_result.scalars().all()

    # All assets
    assets_result = await db.execute(
        select(DocumentAsset)
        .where(DocumentAsset.document_id == doc_id)
        .order_by(DocumentAsset.id)
    )
    assets = assets_result.scalars().all()

    return SessionReaderResponse(
        session=SessionResponse.model_validate(session),
        document_id=doc_id,
        parse_status=parse_status,
        chunks=[ChunkResponse.model_validate(c) for c in chunks],
        assets=[AssetSummary.model_validate(a) for a in assets],
        total_chunks=total_chunks,
    )
