"""
Calibration endpoints — Phase B.

Flow
----
1. GET  /calibration/status     → has_baseline, calib_available, parse_status
2. POST /calibration/start      → create (or reuse) calibration document + chunks, start session
3. POST /calibration/complete   → end session, compute + store baseline
4. GET  /calibration/baseline   → fetch stored baseline (optional utility)

Calibration uses a plain-text file (callibration.txt) instead of a PDF so that
chunks are created instantly (no Docling parse job needed).  The word count for
the calibration text is ~330 words.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import (
    ActivityEvent,
    Document,
    DocumentChunk,
    Session,
    User,
    UserBaseline,
)
from app.db.session import get_db
from app.routers.sessions import _close_session
from app.schemas.calibration import (
    CalibrationCompleteRequest,
    CalibrationCompleteResponse,
    CalibrationReaderResponse,
    CalibrationSessionInfo,
    CalibrationStartResponse,
    CalibrationStatus,
    UserBaselineResponse,
)
from app.services.calibration.baseline import compute_baseline

router = APIRouter(prefix="/calibration", tags=["calibration"])

# Calibration text file — plain text, no Docling needed.
# Repo layout: Lock-in-Mathias-Zmuda-Capstone/callibration/callibration.txt
_CALIB_TXT: Path = (
    Path(__file__).parent.parent.parent.parent.parent
    / "callibration"
    / "callibration.txt"
)

_CALIB_TITLE = "Calibration Reading Material"

# Known total word count of the calibration text (used as a fallback for WPM).
_CALIB_TOTAL_WORDS = 330


# ─── Helpers ──────────────────────────────────────────────────────────────────


async def _get_user_baseline(user_id: int, db: AsyncSession) -> UserBaseline | None:
    result = await db.execute(
        select(UserBaseline).where(UserBaseline.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def _get_user_calib_doc(user_id: int, db: AsyncSession) -> Document | None:
    """Return this user's calibration document (is_calibration=True), if any."""
    result = await db.execute(
        select(Document).where(
            Document.user_id == user_id,
            Document.is_calibration.is_(True),
        )
    )
    return result.scalar_one_or_none()


async def _get_calib_parse_status(doc: Document | None, db: AsyncSession) -> str:
    """
    For txt-based calibration documents, 'succeeded' as soon as at least one
    chunk exists; 'none' otherwise.  No DocumentParseJob is used.
    """
    if doc is None:
        return "none"
    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc.id)
        .limit(1)
    )
    return "succeeded" if result.scalar_one_or_none() else "none"


async def _get_or_create_calib_doc(user: User, db: AsyncSession) -> Document:
    """
    Idempotent: create the calibration document + text chunks the first time
    the user attempts calibration; return the existing one on subsequent calls.

    Everything is done synchronously — no background task, no 503 delay.
    """
    doc = await _get_user_calib_doc(user.id, db)

    if doc is None:
        if not _CALIB_TXT.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Calibration text file not found on server. "
                    f"Expected at: {_CALIB_TXT}"
                ),
            )

        raw_text = _CALIB_TXT.read_text(encoding="utf-8")

        doc = Document(
            user_id=user.id,
            title=_CALIB_TITLE,
            filename=_CALIB_TXT.name,
            file_path=str(_CALIB_TXT),
            file_size=_CALIB_TXT.stat().st_size,
            is_calibration=True,
        )
        db.add(doc)
        await db.flush()  # get doc.id

        # Split into non-empty paragraphs and store as chunks
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        for idx, para in enumerate(paragraphs):
            word_count = len(para.split())
            db.add(DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                page_start=1,
                page_end=1,
                text=para,
                meta={
                    "chunk_type": "text",
                    "label": "paragraph",
                    "word_count": word_count,
                },
            ))

        await db.commit()
        await db.refresh(doc)

    return doc


# ─── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/status", response_model=CalibrationStatus)
async def calibration_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CalibrationStatus:
    """
    Returns whether the user has a stored baseline and whether the calibration
    document is available and ready.
    """
    baseline = await _get_user_baseline(current_user.id, db)
    calib_doc = await _get_user_calib_doc(current_user.id, db)
    parse_status = await _get_calib_parse_status(calib_doc, db)

    return CalibrationStatus(
        has_baseline=baseline is not None,
        calib_available=_CALIB_TXT.exists(),
        parse_status=parse_status,
    )


@router.post(
    "/start",
    response_model=CalibrationStartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def calibration_start(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CalibrationStartResponse:
    """
    Create (or reuse) the user's calibration document, then start a calibration
    session.  Returns immediately — no parse job or waiting required.
    """
    doc = await _get_or_create_calib_doc(current_user, db)

    # Create a new calibration session (always fresh)
    session = Session(
        user_id=current_user.id,
        document_id=doc.id,
        name="Calibration Session",
        mode="calibration",
        status="active",
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return CalibrationStartResponse(session_id=session.id, document_id=doc.id)


@router.post("/complete", response_model=CalibrationCompleteResponse)
async def calibration_complete(
    body: CalibrationCompleteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CalibrationCompleteResponse:
    """
    End the calibration session, compute the reading baseline from telemetry
    batches, and store it in user_baselines (upsert).
    """
    result = await db.execute(
        select(Session).where(
            Session.id == body.session_id,
            Session.user_id == current_user.id,
            Session.mode == "calibration",
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Calibration session not found",
        )
    if session.status in ("ended", "completed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Calibration session is already finished",
        )

    # Close the session and compute duration
    _close_session(session, "completed")
    duration = session.duration_seconds or 0
    await db.commit()
    await db.refresh(session)

    # Fetch telemetry batches for this session (ordered chronologically)
    ev_result = await db.execute(
        select(ActivityEvent).where(
            ActivityEvent.session_id == body.session_id,
            ActivityEvent.event_type == "telemetry_batch",
        ).order_by(ActivityEvent.created_at)
    )
    events = ev_result.scalars().all()
    batches = [e.payload for e in events]

    # Fetch all chunks for the calibration document
    ch_result = await db.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == session.document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = ch_result.scalars().all()

    # int-keyed map (chunk.id → word_count) for backwards-compat path in baseline.py
    chunk_word_counts: dict[int, int] = {}
    # string-keyed map ("calib-N" → word_count) for effective WPM paragraph matching
    paragraph_word_counts: dict[str, int] = {}

    for c in chunks:
        wc = (c.meta or {}).get("word_count") or len(c.text.split())
        chunk_word_counts[c.id] = wc
        paragraph_word_counts[f"calib-{c.chunk_index}"] = wc
        paragraph_word_counts[f"chunk-{c.chunk_index}"] = wc

    # Total words = entire calibration text (user reads all of it).
    total_words: int = sum(chunk_word_counts.values())
    if total_words == 0 and _CALIB_TXT.exists():
        total_words = len(_CALIB_TXT.read_text(encoding="utf-8").split())

    paragraph_count_total = len(chunks)

    # Compute baseline with all enriched data
    baseline_data = compute_baseline(
        batches,
        chunk_word_counts,
        duration,
        total_words=total_words,
        paragraph_word_counts=paragraph_word_counts,
        calibration_text_word_count=total_words,
        paragraph_count_total=paragraph_count_total,
    )

    # Upsert UserBaseline
    now = datetime.now(timezone.utc)
    existing = await _get_user_baseline(current_user.id, db)
    if existing:
        existing.baseline_json = baseline_data
        existing.completed_at = now
    else:
        db.add(UserBaseline(
            user_id=current_user.id,
            baseline_json=baseline_data,
            completed_at=now,
        ))
    await db.commit()

    from app.schemas.calibration import BaselineData
    return CalibrationCompleteResponse(
        baseline=BaselineData(**baseline_data),
        completed_at=now,
        session_id=body.session_id,
    )


@router.get("/session/{session_id}", response_model=CalibrationReaderResponse)
async def calibration_reader(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CalibrationReaderResponse:
    """
    Returns calibration session info + the plain-text paragraphs read directly
    from the .txt file.  No parse job or chunk lookup — always instant.
    """
    result = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == current_user.id,
            Session.mode == "calibration",
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Calibration session not found",
        )

    if not _CALIB_TXT.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Calibration text file not found: {_CALIB_TXT}",
        )

    raw_text = _CALIB_TXT.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    total_words = sum(len(p.split()) for p in paragraphs)

    return CalibrationReaderResponse(
        session=CalibrationSessionInfo(
            id=session.id,
            status=session.status,
            mode=session.mode,
            started_at=session.started_at,
            elapsed_seconds=session.elapsed_seconds or 0,
        ),
        paragraphs=paragraphs,
        total_words=total_words,
    )


@router.get("/baseline", response_model=UserBaselineResponse)
async def get_baseline(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserBaselineResponse:
    """Fetch the stored baseline for the current user."""
    baseline = await _get_user_baseline(current_user.id, db)
    if baseline is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No baseline found. Complete calibration first.",
        )
    return UserBaselineResponse.model_validate(baseline)
