"""
Parsing-related endpoints.

GET  /documents/{id}/parse-status    → parse job status
GET  /documents/{id}/parsed          → paginated chunks + asset list
GET  /documents/{id}/assets/{aid}    → stream extracted image file
POST /documents/{id}/reparse         → clear prior output and restart
"""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import (
    Document,
    DocumentAsset,
    DocumentChunk,
    DocumentParseJob,
    User,
)
from app.db.session import get_db
from app.schemas.parsing import AssetSummary, ChunkResponse, ParsedDocumentResponse, ParseJobStatus
from app.services.parsing.parser import run_parse_job

router = APIRouter(tags=["parsing"])


# ─── Shared helpers ────────────────────────────────────────────────────────────

async def _get_owned_document(doc_id: int, user_id: int, db: AsyncSession) -> Document:
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.user_id == user_id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


async def _get_parse_job_or_404(doc_id: int, db: AsyncSession) -> DocumentParseJob:
    result = await db.execute(
        select(DocumentParseJob).where(DocumentParseJob.document_id == doc_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No parse job found for this document",
        )
    return job


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/documents/{doc_id}/parse-status", response_model=ParseJobStatus)
async def get_parse_status(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ParseJobStatus:
    """Return the current parse job status for a document."""
    await _get_owned_document(doc_id, current_user.id, db)
    job = await _get_parse_job_or_404(doc_id, db)
    return ParseJobStatus.model_validate(job)


@router.get("/documents/{doc_id}/parsed", response_model=ParsedDocumentResponse)
async def get_parsed_document(
    doc_id: int,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=30, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ParsedDocumentResponse:
    """Return paginated chunks and the full asset list for a document."""
    await _get_owned_document(doc_id, current_user.id, db)

    # Total chunk count (for frontend pagination)
    count_result = await db.execute(
        select(func.count()).where(DocumentChunk.document_id == doc_id)
    )
    total = count_result.scalar_one()

    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.chunk_index)
        .offset(offset)
        .limit(limit)
    )
    chunks = chunks_result.scalars().all()

    assets_result = await db.execute(
        select(DocumentAsset)
        .where(DocumentAsset.document_id == doc_id)
        .order_by(DocumentAsset.id)
    )
    assets = assets_result.scalars().all()

    return ParsedDocumentResponse(
        document_id=doc_id,
        chunks=[ChunkResponse.model_validate(c) for c in chunks],
        assets=[AssetSummary.model_validate(a) for a in assets],
        total_chunks=total,
        offset=offset,
        limit=limit,
    )


@router.get("/documents/{doc_id}/assets/{asset_id}")
async def get_asset_file(
    doc_id: int,
    asset_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Stream an extracted image asset file."""
    await _get_owned_document(doc_id, current_user.id, db)

    result = await db.execute(
        select(DocumentAsset).where(
            DocumentAsset.id == asset_id, DocumentAsset.document_id == doc_id
        )
    )
    asset = result.scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

    if not asset.file_path or not Path(asset.file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Asset file not found on disk"
        )

    return FileResponse(
        path=asset.file_path,
        media_type="image/png",
        filename=Path(asset.file_path).name,
    )


@router.post("/documents/{doc_id}/reparse", response_model=ParseJobStatus)
async def reparse_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ParseJobStatus:
    """Clear prior parse output and restart the parse job."""
    doc = await _get_owned_document(doc_id, current_user.id, db)

    # Wipe prior chunks, assets, and the job record
    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc_id))
    await db.execute(delete(DocumentAsset).where(DocumentAsset.document_id == doc_id))
    await db.execute(delete(DocumentParseJob).where(DocumentParseJob.document_id == doc_id))

    # Create fresh pending job
    job = DocumentParseJob(document_id=doc_id, status="pending")
    db.add(job)
    await db.commit()
    await db.refresh(job)

    background_tasks.add_task(run_parse_job, doc_id, doc.file_path)

    return ParseJobStatus.model_validate(job)
