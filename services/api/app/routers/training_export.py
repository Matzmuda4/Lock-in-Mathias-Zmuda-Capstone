"""
Training-data export endpoints — classify branch.

GET  /training/packets/export    — query + export packets to disk, return metadata
POST /training/packets/export    — same, body-driven (for larger session_ids lists)

Both endpoints:
- Enforce auth (own packets only).
- Support format=csv|jsonl, include_debug, include_text_preview, download.
- Write a single consolidated file to training/data/.
- Return { file_path, row_count, columns, from, to, session_count }.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user
from app.db.models import Document, DocumentChunk, Session, SessionStatePacket, User
from app.db.session import get_db
from app.services.training_export.flatten import (
    TRAINING_COLUMNS,
    build_jsonl_line,
    flatten_packet_to_row,
)

router = APIRouter(prefix="/training", tags=["training_export"])
log = logging.getLogger(__name__)


# ── Request / response schemas ────────────────────────────────────────────────


class PacketExportBody(BaseModel):
    session_ids: Optional[list[int]] = None
    format: str = "csv"
    include_debug: bool = False
    include_text_preview: bool = False


class PacketExportResponse(BaseModel):
    file_path: str
    row_count: int
    columns: list[str]
    from_dt: Optional[str] = None
    to_dt: Optional[str] = None
    session_count: int


# ── Shared export logic ───────────────────────────────────────────────────────


async def _run_export(
    user_id: int,
    db: AsyncSession,
    session_ids: Optional[list[int]],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    mode_filter: Optional[str],
    fmt: str,
    include_debug: bool,
    include_text_preview: bool,
) -> PacketExportResponse:
    """
    Core export logic: query packets, flatten, write file, return metadata.

    Ownership is enforced: we only return packets where
    sessions.user_id = current user.
    """
    fmt = fmt.lower()
    if fmt not in ("csv", "jsonl"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="format must be 'csv' or 'jsonl'",
        )

    # ── 1. Query packets with session join for ownership + context ────────────
    q = (
        select(SessionStatePacket, Session)
        .join(Session, Session.id == SessionStatePacket.session_id)
        .where(Session.user_id == user_id)
        .order_by(SessionStatePacket.session_id, SessionStatePacket.packet_seq)
    )

    if session_ids:
        q = q.where(SessionStatePacket.session_id.in_(session_ids))
    if start_date:
        q = q.where(SessionStatePacket.created_at >= start_date)
    if end_date:
        q = q.where(SessionStatePacket.created_at <= end_date)
    if mode_filter and mode_filter != "all":
        q = q.where(Session.mode == mode_filter)

    result = await db.execute(q)
    rows = result.all()

    # ── 2. Optional: preload chunk metadata for text_preview ─────────────────
    chunk_map: dict[tuple[int, int], dict] = {}
    if include_text_preview and rows:
        doc_ids = list({sess.document_id for _, sess in rows})
        chunks_q = await db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id.in_(doc_ids))
        )
        for chunk in chunks_q.scalars().all():
            chunk_map[(chunk.document_id, chunk.chunk_index)] = {
                "chunk_type": (chunk.meta or {}).get("type", ""),
                "word_count": (chunk.meta or {}).get("word_count"),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text_preview": (chunk.text or "")[:200].replace("\n", " "),
            }

    # ── 3. Flatten rows ───────────────────────────────────────────────────────
    flat_rows: list[dict[str, Any]] = []
    seen_sessions: set[int] = set()
    min_dt: Optional[datetime] = None
    max_dt: Optional[datetime] = None

    for packet, sess in rows:
        seen_sessions.add(sess.id)
        created_at = packet.created_at
        if created_at:
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if min_dt is None or created_at < min_dt:
                min_dt = created_at
            if max_dt is None or created_at > max_dt:
                max_dt = created_at

        meta = {
            "session_id": sess.id,
            "user_id": sess.user_id,
            "document_id": sess.document_id,
            "session_mode": sess.mode,
            "packet_seq": packet.packet_seq,
            "created_at": packet.created_at,
            "window_start_at": packet.window_start_at,
            "window_end_at": packet.window_end_at,
        }
        pjson: dict = packet.packet_json or {}

        if fmt == "csv":
            flat = flatten_packet_to_row(meta, pjson, include_debug=include_debug)
            if include_text_preview:
                # Attach chunk metadata for the active chunk in this packet
                feat = pjson.get("features") or {}
                chunk_idx = feat.get("paragraphs_observed")  # best proxy available
                chunk_info = chunk_map.get((sess.document_id, chunk_idx or 0), {})
                flat.update({f"chunk_{k}": v for k, v in chunk_info.items()})
            flat_rows.append(flat)
        else:
            line = build_jsonl_line(meta, pjson, include_debug=include_debug)
            if include_text_preview:
                feat = pjson.get("features") or {}
                chunk_idx = feat.get("paragraphs_observed")
                line["chunk_context"] = chunk_map.get((sess.document_id, chunk_idx or 0), {})
            flat_rows.append(line)

    # ── 4. Determine output columns ───────────────────────────────────────────
    if fmt == "csv":
        columns = list(TRAINING_COLUMNS)
        if include_debug and flat_rows:
            debug_cols = [k for k in flat_rows[0] if k.startswith("debug_")]
            columns = columns + debug_cols
        if include_text_preview:
            columns += [
                "chunk_chunk_type", "chunk_word_count",
                "chunk_page_start", "chunk_page_end", "chunk_text_preview",
            ]
    else:
        columns = ["meta", "baseline", "features", "z_scores", "drift",
                   "ui_aggregates", "packet_json"]
        if include_debug:
            columns.append("debug")
        if include_text_preview:
            columns.append("chunk_context")

    # ── 5. Write file ─────────────────────────────────────────────────────────
    settings.training_data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    ext = fmt
    filename = f"packets_user_{user_id}_{timestamp}.{ext}"
    out_path = settings.training_data_dir / filename

    if fmt == "csv":
        _write_csv(out_path, flat_rows, columns)
    else:
        _write_jsonl(out_path, flat_rows)

    log.info(
        "[training_export] user=%d rows=%d sessions=%d file=%s",
        user_id, len(flat_rows), len(seen_sessions), out_path,
    )

    return PacketExportResponse(
        file_path=str(out_path),
        row_count=len(flat_rows),
        columns=columns,
        from_dt=min_dt.isoformat() if min_dt else None,
        to_dt=max_dt.isoformat() if max_dt else None,
        session_count=len(seen_sessions),
    )


# ── File helpers ──────────────────────────────────────────────────────────────


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


# ── GET endpoint ──────────────────────────────────────────────────────────────


@router.get("/packets/export", response_model=PacketExportResponse)
async def export_packets_get(
    start_date: Optional[datetime] = Query(default=None, description="ISO datetime filter (inclusive)"),
    end_date: Optional[datetime] = Query(default=None, description="ISO datetime filter (inclusive)"),
    session_ids: Optional[str] = Query(default=None, description="Comma-separated session IDs"),
    mode: Optional[str] = Query(default=None, description="baseline | adaptive | all"),
    format: str = Query(default="csv", description="csv or jsonl"),
    include_debug: bool = Query(default=False),
    include_text_preview: bool = Query(default=False),
    download: bool = Query(default=False, description="Return file as attachment"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PacketExportResponse | FileResponse:
    """
    Export all training packets belonging to the current user.

    Writes a consolidated CSV or JSONL file to training/data/ and returns
    metadata.  Use ?download=1 to receive the file directly.
    """
    parsed_ids: Optional[list[int]] = None
    if session_ids:
        try:
            parsed_ids = [int(x.strip()) for x in session_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="session_ids must be comma-separated integers",
            )

    result = await _run_export(
        user_id=current_user.id,
        db=db,
        session_ids=parsed_ids,
        start_date=start_date,
        end_date=end_date,
        mode_filter=mode,
        fmt=format,
        include_debug=include_debug,
        include_text_preview=include_text_preview,
    )

    if download:
        path = Path(result.file_path)
        media = "text/csv" if format == "csv" else "application/x-ndjson"
        return FileResponse(
            path=str(path),
            media_type=media,
            filename=path.name,
        )

    return result


# ── POST endpoint ─────────────────────────────────────────────────────────────


@router.post("/packets/export", response_model=PacketExportResponse)
async def export_packets_post(
    body: PacketExportBody,
    download: bool = Query(default=False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PacketExportResponse | FileResponse:
    """
    Export training packets — body-driven variant for large session_ids lists.
    """
    result = await _run_export(
        user_id=current_user.id,
        db=db,
        session_ids=body.session_ids,
        start_date=None,
        end_date=None,
        mode_filter=None,
        fmt=body.format,
        include_debug=body.include_debug,
        include_text_preview=body.include_text_preview,
    )

    if download:
        path = Path(result.file_path)
        media = "text/csv" if body.format == "csv" else "application/x-ndjson"
        return FileResponse(
            path=str(path),
            media_type=media,
            filename=path.name,
        )

    return result
