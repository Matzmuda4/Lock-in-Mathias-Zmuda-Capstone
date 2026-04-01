"""
Training-data export endpoints — classify branch.

GET  /sessions/{session_id}/export/bundle      — export one session, return paths
POST /exports/sessions                          — batch export many sessions
GET  /exports/users/me/baseline                 — quick baseline inspection

All endpoints enforce auth + ownership.  Files are written to training/exports/.
Pass ``?append_to_master=1`` to also append state packets into
``TrainingData/unlabelled.jsonl`` and write the user baseline to
``TrainingData/baselines/user_{uid}_baseline.json``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user
from app.db.models import Session, User, UserBaseline
from app.db.session import get_db
from app.services.exports.service import ExportResult, export_session_bundle, zip_folder
from app.services.training_export.master_append import (
    MasterAppendResult,
    append_session_to_master,
)

router = APIRouter(tags=["exports"])


# ── Response schemas ──────────────────────────────────────────────────────────


class MasterAppendInfo(BaseModel):
    master_jsonl_path: str
    appended_packet_count: int
    skipped_packet_count: int
    baseline_path: str
    baseline_ref: str
    baseline_embedded_in_packet: bool


class ExportBundleResponse(BaseModel):
    session_id: int
    user_id: int
    folder: str
    files: list[str]
    state_packet_count: int
    telemetry_batch_count: int
    event_count: int
    master_append: Optional[MasterAppendInfo] = None


class BatchExportRequest(BaseModel):
    session_ids: list[int]
    # Optional label for the labelling run — recorded in session_meta.json.
    # Use to distinguish collection runs (e.g. "pilot_2026", "adhd_cohort_1").
    protocol_tag: Optional[str] = None


class BatchExportResponse(BaseModel):
    results: list[ExportBundleResponse]
    errors: list[dict[str, Any]]


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_owned_session(
    session_id: int,
    user_id: int,
    db: AsyncSession,
) -> Session:
    row = (
        await db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return row


def _result_to_response(
    result: ExportResult,
    master: Optional[MasterAppendResult] = None,
) -> ExportBundleResponse:
    master_info: Optional[MasterAppendInfo] = None
    if master is not None:
        master_info = MasterAppendInfo(
            master_jsonl_path=str(master.master_jsonl_path),
            appended_packet_count=master.appended_packet_count,
            skipped_packet_count=master.skipped_packet_count,
            baseline_path=str(master.baseline_path),
            baseline_ref=master.baseline_ref,
            baseline_embedded_in_packet=master.baseline_embedded_in_packet,
        )
    return ExportBundleResponse(
        session_id=result.session_id,
        user_id=result.user_id,
        folder=str(result.folder),
        files=result.files,
        state_packet_count=result.state_packet_count,
        telemetry_batch_count=result.telemetry_batch_count,
        event_count=result.event_count,
        master_append=master_info,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/sessions/{session_id}/export/bundle",
    response_model=ExportBundleResponse,
    summary="Export training-data bundle for one session",
)
async def export_session_bundle_endpoint(
    session_id: int,
    download: bool = Query(
        default=False,
        description="If true, stream the bundle as a ZIP archive instead of returning file paths",
    ),
    protocol_tag: Optional[str] = Query(
        default=None,
        description="Optional label for the labelling run, stored in session_meta.json",
    ),
    append_to_master: bool = Query(
        default=False,
        description=(
            "If true, append this session's state packets to "
            "TrainingData/unlabelled.jsonl and write the user baseline to "
            "TrainingData/baselines/user_{uid}_baseline.json. "
            "Deduplication ensures packets are never appended twice."
        ),
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Write the full training-data bundle for *session_id* to
    ``training/exports/user_{uid}/session_{sid}/``.

    Returns a JSON summary of written files.  Pass ``?download=1`` to receive
    the bundle as a ZIP archive (application/zip).

    Pass ``?append_to_master=1`` to additionally append state packets into the
    master ``TrainingData/unlabelled.jsonl`` for LLM labelling.  The response
    will include a ``master_append`` object with paths and counts.

    Ownership enforced: you can only export your own sessions.
    """
    await _get_owned_session(session_id, current_user.id, db)

    result = await export_session_bundle(
        current_user.id, session_id, db, protocol_tag=protocol_tag
    )

    master_result: Optional[MasterAppendResult] = None
    if append_to_master:
        master_result = append_session_to_master(
            export_folder=result.folder,
            master_dir=settings.training_master_dir,
            user_id=current_user.id,
            session_id=session_id,
        )

    if download:
        zip_path = zip_folder(result.folder)
        zip_bytes = zip_path.read_bytes()
        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id}.zip"',
                "Content-Length": str(len(zip_bytes)),
            },
        )

    return _result_to_response(result, master=master_result)


@router.post(
    "/exports/sessions",
    response_model=BatchExportResponse,
    summary="Batch-export training-data bundles for multiple sessions",
)
async def batch_export_sessions(
    body: BatchExportRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BatchExportResponse:
    """
    Export bundles for all session IDs in *session_ids*.

    Only sessions owned by the authenticated user are exported.
    Sessions belonging to other users silently produce an error entry rather
    than raising a 403 to allow partial success in batch operations.
    """
    if not body.session_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_ids must be a non-empty list",
        )

    results: list[ExportBundleResponse] = []
    errors: list[dict[str, Any]] = []

    for sid in body.session_ids:
        try:
            # Ownership check (raises 404 for wrong user / missing)
            await _get_owned_session(sid, current_user.id, db)
            result = await export_session_bundle(
                current_user.id, sid, db,
                protocol_tag=body.protocol_tag,
            )
            results.append(_result_to_response(result))
        except HTTPException as exc:
            errors.append({"session_id": sid, "error": exc.detail})
        except Exception as exc:  # noqa: BLE001
            errors.append({"session_id": sid, "error": str(exc)})

    return BatchExportResponse(results=results, errors=errors)


@router.get(
    "/exports/users/me/baseline",
    summary="Quick baseline inspection for the current user",
)
async def get_my_baseline(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Return the current user's calibration baseline for quick inspection.

    Does NOT write any files — use /sessions/{id}/export/bundle to persist.
    """
    row = (
        await db.execute(
            select(UserBaseline).where(UserBaseline.user_id == current_user.id)
        )
    ).scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No baseline found. Complete calibration first.",
        )

    baseline_json = row.baseline_json or {}
    wpm_eff = baseline_json.get("wpm_effective") or baseline_json.get("wpm_gross") or 0.0
    baseline_valid = (
        wpm_eff > 0
        and baseline_json.get("calibration_duration_seconds", 0) >= 60
    )

    return {
        "user_id": current_user.id,
        "baseline_valid": baseline_valid,
        "baseline_updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "baseline": baseline_json,
    }
