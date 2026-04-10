"""
Study document seeding endpoint.

POST /study/seed-documents
    Parses the two condition text files (baseline.txt / adaptive.txt) from the
    repository and inserts them as Document + DocumentChunk records for the
    authenticated participant user.  Idempotent: documents already seeded for
    this user are returned without re-inserting.

Returns
-------
    { "baseline_doc_id": int, "adaptive_doc_id": int }
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import Document, DocumentChunk, DocumentParseJob, User
from app.db.session import get_db

router = APIRouter(prefix="/study", tags=["study"])

# ── Filesystem paths ────────────────────────────────────────────────────────────
# study_setup.py lives at services/api/app/routers/ — five levels up is repo root.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent.parent
_CONDITION_DIR: Path = _REPO_ROOT / "UserStudy" / "ConditionTexts"
_RESULTS_DIR:   Path = _REPO_ROOT / "experimentresults"

BASELINE_TXT = _CONDITION_DIR / "baseline.txt"
ADAPTIVE_TXT = _CONDITION_DIR / "adaptive.txt"

# Matches "PART I", "PART IV", "CHAPTER I. TITLE …" etc.
_SECTION_RE = re.compile(r"^(PART|CHAPTER)\s+[IVX]+", re.IGNORECASE)


# ── Response schema ────────────────────────────────────────────────────────────

class SeedDocumentsResponse(BaseModel):
    baseline_doc_id: int
    adaptive_doc_id: int


# ── Text parser ────────────────────────────────────────────────────────────────

def _parse_txt(file_path: Path) -> tuple[str, str, list[dict]]:
    """
    Parse a condition text file into (title, author, chunk_dicts).

    The file format is:
        Title: <title>
        <blank>
        Author: <author>
        <blank>
        PART/CHAPTER headings and prose paragraphs (blank-line-separated)
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    title = ""
    author = ""

    for line in lines:
        if line.startswith("Title:"):
            title = line[6:].strip()
        elif line.startswith("Author:"):
            author = line[7:].strip()

    # Locate content start — everything after the Author: line
    past_author = False
    content_lines: list[str] = []
    for line in lines:
        if past_author:
            content_lines.append(line)
        elif line.startswith("Author:"):
            past_author = True

    # Build chunk records
    chunks: list[dict] = []
    chunk_index = 0
    current_section: str | None = None
    para_buf: list[str] = []

    def _flush() -> None:
        nonlocal chunk_index
        text = " ".join(para_buf).strip()
        para_buf.clear()
        if len(text) < 30:
            return
        chunks.append(
            {
                "chunk_index": chunk_index,
                "page_start": 1,
                "page_end": 1,
                "text": text,
                "meta": {
                    "chunk_type": "text",
                    "label": "paragraph",
                    "section": current_section,
                    "bbox": None,
                },
            }
        )
        chunk_index += 1

    for line in content_lines:
        stripped = line.strip()
        if not stripped:
            _flush()
        elif _SECTION_RE.match(stripped):
            _flush()
            current_section = stripped
            # Section headings render as <h2> in the reader
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "page_start": 1,
                    "page_end": 1,
                    "text": stripped,
                    "meta": {
                        "chunk_type": "text",
                        "label": "section_header",
                        "section": stripped,
                        "bbox": None,
                    },
                }
            )
            chunk_index += 1
        else:
            para_buf.append(stripped)

    _flush()  # Flush trailing paragraph
    return title, author, chunks


# ── DB helper ──────────────────────────────────────────────────────────────────

async def _seed_one(
    db: AsyncSession,
    user_id: int,
    filename: str,
    file_path: Path,
) -> int:
    """
    Ensure a document seeded from *file_path* exists for *user_id*.
    Returns the document ID.  Idempotent by (user_id, filename).
    """
    # Check for existing record
    result = await db.execute(
        select(Document).where(
            Document.user_id == user_id,
            Document.filename == filename,
        )
    )
    existing = result.scalar_one_or_none()
    if existing is not None:
        return existing.id

    title, _author, chunk_dicts = _parse_txt(file_path)
    content_bytes = file_path.read_bytes()

    doc = Document(
        user_id=user_id,
        title=title,
        filename=filename,
        file_path=str(file_path),
        file_size=len(content_bytes),
    )
    db.add(doc)
    await db.flush()  # assign doc.id

    # Insert chunks
    for cd in chunk_dicts:
        db.add(
            DocumentChunk(
                document_id=doc.id,
                chunk_index=cd["chunk_index"],
                page_start=cd["page_start"],
                page_end=cd["page_end"],
                text=cd["text"],
                meta=cd["meta"],
            )
        )

    # Mark parse job as succeeded (no Docling needed for plain text)
    now = datetime.now(timezone.utc)
    db.add(
        DocumentParseJob(
            document_id=doc.id,
            status="succeeded",
            started_at=now,
            finished_at=now,
        )
    )

    await db.commit()
    await db.refresh(doc)
    return doc.id


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("/seed-documents", response_model=SeedDocumentsResponse)
async def seed_study_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SeedDocumentsResponse:
    """
    Seed both condition documents for the authenticated participant user.

    Safe to call multiple times — already-seeded documents are returned as-is.
    """
    for path in (BASELINE_TXT, ADAPTIVE_TXT):
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Study condition file not found on server: {path.name}",
            )

    baseline_id = await _seed_one(db, current_user.id, "baseline.txt", BASELINE_TXT)
    adaptive_id  = await _seed_one(db, current_user.id, "adaptive.txt", ADAPTIVE_TXT)

    return SeedDocumentsResponse(
        baseline_doc_id=baseline_id,
        adaptive_doc_id=adaptive_id,
    )


# ── Save-export endpoint ───────────────────────────────────────────────────────

class SaveExportRequest(BaseModel):
    participant_id: str
    master_json: str
    baseline_csv: str | None = None
    adaptive_csv: str | None = None
    timeline_csv: str | None = None


class SaveExportResponse(BaseModel):
    saved_files: list[str]
    directory: str


@router.post("/save-export", response_model=SaveExportResponse)
async def save_study_export(
    body: SaveExportRequest,
    _current_user: User = Depends(get_current_user),
) -> SaveExportResponse:
    """
    Persist the three study-export files to experimentresults/ in the repo.

    Called automatically by the frontend at the end of Step 16 so data is
    stored server-side without depending on the browser download location.
    The browser download still fires as a secondary fallback.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pid = body.participant_id.replace(" ", "_")
    saved: list[str] = []

    def _write(filename: str, content: str) -> None:
        path = _RESULTS_DIR / filename
        path.write_text(content, encoding="utf-8")
        saved.append(filename)

    _write(f"lockin_{pid}_study.json",              body.master_json)
    if body.baseline_csv:
        _write(f"lockin_{pid}_baseline_telemetry.csv", body.baseline_csv)
    if body.adaptive_csv:
        _write(f"lockin_{pid}_adaptive_telemetry.csv", body.adaptive_csv)
    if body.timeline_csv:
        _write(f"lockin_{pid}_timeline.csv",            body.timeline_csv)

    return SaveExportResponse(
        saved_files=saved,
        directory=str(_RESULTS_DIR),
    )
