from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user
from app.db.models import Document, User
from app.db.session import get_db
from app.schemas.documents import DocumentListResponse, DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_CONTENT_TYPES = {"application/pdf"}
_ALLOWED_EXTENSIONS = {".pdf"}


async def _get_owned_document(doc_id: int, user_id: int, db: AsyncSession) -> Document:
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.user_id == user_id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    title: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Upload a PDF file and create a document record."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Store under uploads/<user_id>/<uuid>_<original_filename>
    user_dir: Path = settings.upload_dir / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}_{file.filename}"
    file_path = user_dir / stored_name

    async with aiofiles.open(file_path, "wb") as fp:
        await fp.write(content)

    doc = Document(
        user_id=current_user.id,
        title=title.strip(),
        filename=file.filename or stored_name,
        file_path=str(file_path),
        file_size=len(content),
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return DocumentResponse.model_validate(doc)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    result = await db.execute(
        select(Document).where(Document.user_id == current_user.id)
    )
    docs = result.scalars().all()
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=len(docs),
    )


@router.get("/{doc_id}/file")
async def get_document_file(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Stream the PDF file back to the client."""
    doc = await _get_owned_document(doc_id, current_user.id, db)
    if not Path(doc.file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk",
        )
    return FileResponse(
        path=doc.file_path,
        media_type="application/pdf",
        filename=doc.filename,
    )


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete document record and remove the file from disk."""
    doc = await _get_owned_document(doc_id, current_user.id, db)
    file_path = Path(doc.file_path)

    await db.delete(doc)
    await db.commit()

    if file_path.exists():
        file_path.unlink()
