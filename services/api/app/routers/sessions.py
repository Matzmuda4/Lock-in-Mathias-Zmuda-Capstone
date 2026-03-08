import csv
import io
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user
from app.db.models import ActivityEvent, Document, DocumentAsset, DocumentChunk, DocumentParseJob, Session, User
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
    now = datetime.now(timezone.utc)
    # Accumulate the seconds from this active interval into elapsed_seconds.
    # started_at here means "last resumed at" (set on creation and on each resume).
    session.elapsed_seconds = (session.elapsed_seconds or 0) + max(
        0, int((now - session.started_at).total_seconds())
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
    # Reset the interval start time so elapsed counting is correct next pause/close.
    session.started_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(session)
    return SessionResponse.model_validate(session)


_TERMINAL_STATUSES = frozenset({"ended", "completed"})


def _close_session(session: Session, final_status: str) -> None:
    """Shared logic for both /end and /complete."""
    now = datetime.now(timezone.utc)
    # If the session is currently active, add the current interval first.
    current_interval = 0
    if session.status == "active":
        current_interval = max(0, int((now - session.started_at).total_seconds()))
    session.status = final_status
    session.ended_at = now
    session.duration_seconds = (session.elapsed_seconds or 0) + current_interval


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

    # Parse job status. Calibration (and legacy txt-based) documents have no
    # parse job — if chunks already exist we treat that as "succeeded".
    job_result = await db.execute(
        select(DocumentParseJob).where(DocumentParseJob.document_id == doc_id)
    )
    job = job_result.scalar_one_or_none()
    if job:
        parse_status = job.status
    else:
        # No parse job: succeeded if chunks exist, pending otherwise
        has_chunks_result = await db.execute(
            select(func.count()).where(DocumentChunk.document_id == doc_id)
        )
        parse_status = "succeeded" if (has_chunks_result.scalar_one() or 0) > 0 else "pending"

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


_CSV_FIELDS = [
    "created_at",
    # scroll
    "scroll_delta_sum",
    "scroll_delta_abs_sum",
    "scroll_delta_pos_sum",
    "scroll_delta_neg_sum",
    "scroll_event_count",
    "scroll_direction_changes",
    "scroll_pause_seconds",
    # engagement
    "idle_seconds",
    # mouse
    "mouse_path_px",
    "mouse_net_px",
    # focus
    "window_focus_state",
    # reading position
    "current_paragraph_id",
    "current_chunk_index",
    "viewport_progress_ratio",
    # presentation profile
    "viewport_height_px",
    "viewport_width_px",
    "reader_container_height_px",
]


@router.get("/{session_id}/export.csv")
async def export_session_csv(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Export all telemetry_batch events for a session as CSV.

    The file is also written to exports/user_{id}/session_{id}.csv on disk
    (gitignored) for offline analysis.
    """
    session = await _get_owned_session(session_id, current_user.id, db)

    ev_result = await db.execute(
        select(ActivityEvent).where(
            ActivityEvent.session_id == session_id,
            ActivityEvent.event_type == "telemetry_batch",
        ).order_by(ActivityEvent.created_at)
    )
    events = ev_result.scalars().all()

    # Build CSV in memory
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for event in events:
        row = {k: event.payload.get(k, "") for k in _CSV_FIELDS}
        row["created_at"] = event.created_at.isoformat()
        writer.writerow(row)

    csv_content = buf.getvalue()

    # Persist to disk (best-effort; failure must not break the response)
    try:
        export_path = settings.exports_dir / f"user_{current_user.id}" / f"session_{session_id}.csv"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(csv_content, encoding="utf-8")
    except Exception:
        pass

    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="session_{session_id}.csv"'
        },
    )
