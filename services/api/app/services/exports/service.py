"""
Training-data export service — classify branch.

Writes per-session bundles to training/exports/user_{uid}/session_{sid}/:

  session_meta.json      — session + document metadata
  baseline.json          — user calibration baseline
  state_packets.jsonl    — periodic drift packets (one per ~10 s)
  telemetry_batches.csv  — 2-second aggregated telemetry
  events.csv             — non-batch activity events
  document_chunks.csv    — optional chunk metadata

SOLID: this module only deals with DB reads + file writes.  No HTTP, no auth.
The router layer enforces ownership before calling these functions.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import (
    ActivityEvent,
    Document,
    DocumentChunk,
    Session,
    SessionStatePacket,
    UserBaseline,
)


# ── Deterministic telemetry column order ─────────────────────────────────────
# Any payload key not in this list is still captured in `payload_json`.

TELEMETRY_COLUMNS: list[str] = [
    "created_at",
    "session_id",
    # Scroll signals
    "scroll_delta_sum",
    "scroll_delta_abs_sum",
    "scroll_delta_pos_sum",
    "scroll_delta_neg_sum",
    "scroll_event_count",
    "scroll_direction_changes",
    "scroll_pause_seconds",
    # Idle / engagement
    "idle_seconds",
    "idle_since_interaction_seconds",
    # Mouse
    "mouse_path_px",
    "mouse_net_px",
    # Focus / position
    "window_focus_state",
    "current_paragraph_id",
    "current_chunk_index",
    "viewport_progress_ratio",
    # Viewport dimensions
    "viewport_height_px",
    "viewport_width_px",
    "reader_container_height_px",
    # Quality flags (server-side)
    "telemetry_fault",
    "scroll_capture_fault",
    "paragraph_missing_fault",
    # UI context (Phase 8 — adaptive panel telemetry)
    "ui_context",
    "interaction_zone",
    # Raw JSON fallback
    "payload_json",
]

EVENT_COLUMNS: list[str] = [
    "created_at",
    "session_id",
    "event_type",
    "payload_json",
]

CHUNK_COLUMNS: list[str] = [
    "chunk_index",
    "chunk_type",
    "word_count",
    "page_start",
    "page_end",
    "asset_id",
    "text_preview",
]

SCHEMA_VERSION = "1.0.0"


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class ExportResult:
    session_id: int
    user_id: int
    folder: Path
    files: list[str]
    state_packet_count: int
    telemetry_batch_count: int
    event_count: int


# ── Low-level file helpers ─────────────────────────────────────────────────────


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def zip_folder(folder: Path) -> Path:
    """Create a ZIP of the session folder. Returns the zip path."""
    zip_path = folder.parent / f"{folder.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in folder.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(folder.parent))
    return zip_path


# ── Telemetry flattener ───────────────────────────────────────────────────────


def flatten_telemetry_batch(created_at: datetime, session_id: int, payload: dict) -> dict:
    """
    Convert a JSONB payload + metadata into a flat dict using TELEMETRY_COLUMNS.

    Unknown payload fields are NOT lost — they appear in `payload_json`.
    Missing fields default to empty string.
    """
    row: dict[str, Any] = {
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else str(created_at),
        "session_id": session_id,
        "payload_json": json.dumps(payload, default=str),
    }
    for col in TELEMETRY_COLUMNS:
        if col in ("created_at", "session_id", "payload_json"):
            continue
        row[col] = payload.get(col, "")
    return row


# ── Main export entry point ───────────────────────────────────────────────────


async def export_session_bundle(
    user_id: int,
    session_id: int,
    db: AsyncSession,
    protocol_tag: Optional[str] = None,
) -> ExportResult:
    """
    Write the full training-data bundle for one session to disk.

    Does NOT enforce ownership — callers must verify ownership before calling.
    Returns an ExportResult with paths and counts.
    """
    # ── 1. Session + document metadata ────────────────────────────────────
    session_row = (
        await db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == user_id,
            )
        )
    ).scalar_one_or_none()

    if session_row is None:
        raise ValueError(f"Session {session_id} not found for user {user_id}")

    doc_row = (
        await db.execute(select(Document).where(Document.id == session_row.document_id))
    ).scalar_one_or_none()

    # ── 2. Baseline ───────────────────────────────────────────────────────
    baseline_row = (
        await db.execute(
            select(UserBaseline).where(UserBaseline.user_id == user_id)
        )
    ).scalar_one_or_none()

    baseline_json: dict = baseline_row.baseline_json if baseline_row else {}
    wpm_eff = baseline_json.get("wpm_effective") or baseline_json.get("wpm_gross") or 0.0
    baseline_valid = (
        baseline_row is not None
        and wpm_eff > 0
        and baseline_json.get("calibration_duration_seconds", 0) >= 60
    )

    # ── 3. Telemetry batches ──────────────────────────────────────────────
    batch_result = await db.execute(
        select(ActivityEvent)
        .where(
            ActivityEvent.session_id == session_id,
            ActivityEvent.event_type == "telemetry_batch",
        )
        .order_by(ActivityEvent.created_at)
    )
    batch_rows = batch_result.scalars().all()

    # ── 4. Non-batch events ───────────────────────────────────────────────
    events_result = await db.execute(
        select(ActivityEvent)
        .where(
            ActivityEvent.session_id == session_id,
            ActivityEvent.event_type != "telemetry_batch",
        )
        .order_by(ActivityEvent.created_at)
    )
    event_rows = events_result.scalars().all()

    # ── 5. State packets ──────────────────────────────────────────────────
    packets_result = await db.execute(
        select(SessionStatePacket)
        .where(SessionStatePacket.session_id == session_id)
        .order_by(SessionStatePacket.created_at)
    )
    packet_rows = packets_result.scalars().all()

    # ── 6. Document chunks (optional) ────────────────────────────────────
    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == session_row.document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunk_rows = chunks_result.scalars().all()

    # ── Build output folder ───────────────────────────────────────────────
    folder = (
        settings.training_exports_dir
        / f"user_{user_id}"
        / f"session_{session_id}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    files_written: list[str] = []

    # A — session_meta.json
    meta = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_row.id,
        "user_id": session_row.user_id,
        "document_id": session_row.document_id,
        "document_title": doc_row.title if doc_row else None,
        "document_filename": doc_row.filename if doc_row else None,
        "document_is_calibration": doc_row.is_calibration if doc_row else None,
        "session_name": session_row.name,
        "mode": session_row.mode,
        "status": session_row.status,
        "started_at": session_row.started_at.isoformat() if session_row.started_at else None,
        "ended_at": session_row.ended_at.isoformat() if session_row.ended_at else None,
        "duration_seconds": session_row.duration_seconds,
        "elapsed_seconds": session_row.elapsed_seconds,
        "created_at": session_row.created_at.isoformat(),
        "updated_at": session_row.updated_at.isoformat(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "app_version": settings.app_version,
        "protocol_tag": protocol_tag,
    }
    _write_json(folder / "session_meta.json", meta)
    files_written.append("session_meta.json")

    # B — baseline.json
    baseline_export = {
        "schema_version": SCHEMA_VERSION,
        "user_id": user_id,
        "baseline_valid": baseline_valid,
        "baseline_updated_at": baseline_row.updated_at.isoformat() if baseline_row else None,
        "baseline": baseline_json,
    }
    _write_json(folder / "baseline.json", baseline_export)
    files_written.append("baseline.json")

    # C — state_packets.jsonl
    packet_dicts = []
    for p in packet_rows:
        packet_dicts.append({
            "session_id": p.session_id,
            "created_at": p.created_at.isoformat(),
            "packet_seq": getattr(p, "packet_seq", None),
            "window_start_at": (
                p.window_start_at.isoformat() if getattr(p, "window_start_at", None) else None
            ),
            "window_end_at": (
                p.window_end_at.isoformat() if getattr(p, "window_end_at", None) else None
            ),
            "packet": p.packet_json,
        })
    _write_jsonl(folder / "state_packets.jsonl", packet_dicts)
    files_written.append("state_packets.jsonl")

    # D — telemetry_batches.csv
    telemetry_flat = [
        flatten_telemetry_batch(b.created_at, b.session_id, b.payload)
        for b in batch_rows
    ]
    _write_csv(folder / "telemetry_batches.csv", telemetry_flat, TELEMETRY_COLUMNS)
    files_written.append("telemetry_batches.csv")

    # E — events.csv
    event_flat = [
        {
            "created_at": e.created_at.isoformat(),
            "session_id": e.session_id,
            "event_type": e.event_type,
            "payload_json": json.dumps(e.payload, default=str),
        }
        for e in event_rows
    ]
    _write_csv(folder / "events.csv", event_flat, EVENT_COLUMNS)
    files_written.append("events.csv")

    # F — document_chunks.csv (optional)
    if chunk_rows:
        chunk_flat = []
        for c in chunk_rows:
            meta_j = c.meta or {}
            chunk_flat.append({
                "chunk_index": c.chunk_index,
                "chunk_type": meta_j.get("type", meta_j.get("chunk_type", "")),
                "word_count": meta_j.get("word_count", ""),
                "page_start": c.page_start,
                "page_end": c.page_end,
                "asset_id": "",
                "text_preview": (c.text or "")[:120].replace("\n", " "),
            })
        _write_csv(folder / "document_chunks.csv", chunk_flat, CHUNK_COLUMNS)
        files_written.append("document_chunks.csv")

    return ExportResult(
        session_id=session_id,
        user_id=user_id,
        folder=folder,
        files=files_written,
        state_packet_count=len(packet_rows),
        telemetry_batch_count=len(batch_rows),
        event_count=len(event_rows),
    )
