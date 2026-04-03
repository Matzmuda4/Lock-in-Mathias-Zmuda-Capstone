"""
Persistence helpers for drift state — Phase 7 / classify branch.

The session_drift_states table is a simple key-value store keyed by session_id.
We upsert on every drift recompute.

session_state_packets is an append-only hypertable storing full state packets
every exactly 10 s (every 5th telemetry batch at 2s cadence).  These packets
are the primary training-data source exported by the classify pipeline.

Each packet is self-contained: it embeds the baseline_snapshot, session context
(user_id, document_id, session_mode), and ui_context/interaction_zone aggregates
so it can be used for training without any additional joins.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    SessionDriftHistory,
    SessionDriftState,
    SessionStatePacket,
    UserBaseline,
)
from app.services.drift.types import DriftResult

# Exactly one packet per 5 telemetry batches (5 × 2 s = 10 s cadence).
_HISTORY_EVERY_N: int = 5
_upsert_counters: dict[int, int] = {}


@dataclass
class PacketWrittenInfo:
    """
    Returned by upsert_drift_state when a SessionStatePacket was written
    this cycle.  Carries everything the classifier needs to build its input.
    """
    packet_json:     dict[str, Any]
    packet_seq:      int
    window_start_at: datetime
    window_end_at:   datetime


def _compute_ui_aggregates(batches: list[dict[str, Any]]) -> dict[str, float]:
    """
    Aggregate ui_context and interaction_zone fields across the window batches.

    Returns fractions (0.0–1.0) for each discrete value so the training row
    captures the distribution of UI states over the 30-second window.
    """
    n = len(batches) or 1  # avoid div-by-zero

    ui_ctx_counts: Counter = Counter()
    iz_counts: Counter = Counter()
    for b in batches:
        ui = b.get("ui_context") or "READ_MAIN"
        iz = b.get("interaction_zone") or "reader"
        ui_ctx_counts[ui] += 1
        iz_counts[iz] += 1

    panel_states = {"PANEL_OPEN", "PANEL_INTERACTING"}
    panel_share = sum(ui_ctx_counts[s] for s in panel_states) / n

    return {
        "ui_read_main_share_30s": ui_ctx_counts["READ_MAIN"] / n,
        "ui_panel_open_share_30s": ui_ctx_counts["PANEL_OPEN"] / n,
        "ui_panel_interacting_share_30s": ui_ctx_counts["PANEL_INTERACTING"] / n,
        "ui_user_paused_share_30s": ui_ctx_counts["USER_PAUSED"] / n,
        "panel_share_30s": panel_share,
        "reader_share_30s": ui_ctx_counts["READ_MAIN"] / n,
        "iz_reader_share_30s": iz_counts["reader"] / n,
        "iz_panel_share_30s": iz_counts["panel"] / n,
        "iz_other_share_30s": iz_counts["other"] / n,
    }


async def upsert_drift_state(
    session_id: int,
    result: DriftResult,
    db: AsyncSession,
    baseline_row: Optional[UserBaseline] = None,
    window_end_at: Optional[datetime] = None,
    session_context: Optional[dict[str, Any]] = None,
    batches: Optional[list[dict[str, Any]]] = None,
) -> tuple[SessionDriftState, Optional[PacketWrittenInfo]]:
    """
    Insert or update the drift state row for a session (latest snapshot).

    Also appends rows to session_drift_history and session_state_packets every
    exactly 10 seconds (_HISTORY_EVERY_N = 5 calls) to preserve the full
    trajectory.

    Parameters
    ----------
    baseline_row : optional UserBaseline ORM row for the session's user.
        Embedded in the state packet for self-contained training data.
    window_end_at : optional timestamp of the newest telemetry batch in the
        30 s window.  window_start_at = window_end_at − 30 s.
    session_context : optional dict with {user_id, document_id, session_mode}.
        Embedded in packet_json so each packet is fully self-contained.
    batches : optional list of raw batch dicts from the rolling window.
        Used to compute ui_context / interaction_zone aggregates for the packet.

    Returns
    -------
    (SessionDriftState, PacketWrittenInfo | None)
        PacketWrittenInfo is non-None only on the cycle where a state packet
        was written (every 5th call).  The caller uses it to fire the
        classifier without coupling this persistence layer to the LLM service.
    """
    now = datetime.now(timezone.utc)

    existing_q = await db.execute(
        select(SessionDriftState).where(SessionDriftState.session_id == session_id)
    )
    row = existing_q.scalar_one_or_none()

    if row is None:
        row = SessionDriftState(
            session_id=session_id,
            drift_level=result.drift_level,
            drift_ema=result.drift_ema,
            disruption_score=result.disruption_score,
            engagement_score=result.engagement_score,
            beta_effective=result.beta_effective,
            beta_ema=result.beta_ema,
            attention_score=result.attention_score,
            drift_score=result.drift_score,
            confidence=result.confidence,
            last_window_ends_at=now,
            updated_at=now,
        )
        db.add(row)
    else:
        row.drift_level = result.drift_level
        row.drift_ema = result.drift_ema
        row.disruption_score = result.disruption_score
        row.engagement_score = result.engagement_score
        row.beta_effective = result.beta_effective
        row.beta_ema = result.beta_ema
        row.attention_score = result.attention_score
        row.drift_score = result.drift_score
        row.confidence = result.confidence
        row.last_window_ends_at = now
        row.updated_at = now

    # ── Append to history and state-packet tables every exactly 10 seconds ──
    # Counter starts at 0; first packet fires after the 5th batch (10 s).
    _upsert_counters[session_id] = _upsert_counters.get(session_id, 0) + 1
    packet_info: Optional[PacketWrittenInfo] = None

    if _upsert_counters[session_id] % _HISTORY_EVERY_N == 0:
        db.add(SessionDriftHistory(
            session_id=session_id,
            created_at=now,
            drift_ema=result.drift_ema,
            beta_ema=result.beta_ema,
            disruption_score=result.disruption_score,
            engagement_score=result.engagement_score,
            confidence=result.confidence,
            pace_ratio=result.pace_ratio,
        ))

        # Compute packet_seq (next monotonic counter for this session)
        seq_result = await db.execute(
            select(func.coalesce(func.max(SessionStatePacket.packet_seq), -1) + 1).where(
                SessionStatePacket.session_id == session_id
            )
        )
        next_seq: int = seq_result.scalar() or 0

        # Window bounds
        w_end = window_end_at or now
        w_start = w_end - timedelta(seconds=30)

        pkt_json = _build_packet_json(
            result,
            baseline_row=baseline_row,
            session_context=session_context,
            batches=batches or [],
        )

        # Store full state packet for training-data export
        db.add(SessionStatePacket(
            session_id=session_id,
            created_at=now,
            packet_seq=next_seq,
            window_start_at=w_start,
            window_end_at=w_end,
            packet_json=pkt_json,
        ))

        # Signal to the caller that a packet was written this cycle so it
        # can fire the classifier without this persistence layer needing to
        # know anything about the LLM service.
        packet_info = PacketWrittenInfo(
            packet_json=pkt_json,
            packet_seq=next_seq,
            window_start_at=w_start,
            window_end_at=w_end,
        )

    await db.flush()
    return row, packet_info


def _build_packet_json(
    result: DriftResult,
    baseline_row: Optional[UserBaseline] = None,
    session_context: Optional[dict[str, Any]] = None,
    batches: Optional[list[dict[str, Any]]] = None,
) -> dict:
    """
    Serialise a DriftResult into a self-contained dict for storage.

    Embeds:
    - baseline_snapshot  (so no user_baselines join required)
    - session_context    (user_id, document_id, session_mode)
    - ui_aggregates      (ui_context / interaction_zone distributions over 30 s)
    """
    baseline_json: dict[str, Any] = baseline_row.baseline_json if baseline_row else {}
    wpm_eff = baseline_json.get("wpm_effective") or baseline_json.get("wpm_gross") or 0.0
    baseline_valid = (
        baseline_row is not None
        and wpm_eff > 0
        and baseline_json.get("calibration_duration_seconds", 0) >= 60
    )
    baseline_snapshot: dict[str, Any] = {
        "baseline_json": baseline_json or None,
        "baseline_updated_at": (
            baseline_row.updated_at.isoformat() if baseline_row and baseline_row.updated_at else None
        ),
        "baseline_valid": baseline_valid,
    }

    ctx = session_context or {}
    ui_agg = _compute_ui_aggregates(batches or [])

    # ── Reading position from the latest telemetry batch in the window ────────
    # current_paragraph_id is set by the frontend renderer and identifies which
    # DocumentChunk the user's viewport was centred on at the end of the window.
    # This lets the intervention LLM look up the exact paragraph text without
    # any additional queries — it maps directly to document_chunks.chunk_index.
    _last_b: dict[str, Any] = (batches or [{}])[-1] if batches else {}
    _text_modified_count: int = sum(
        1 for b in (batches or []) if b.get("text_modified", False)
    )
    # current_chunk_index is the integer DocumentChunk.chunk_index emitted by
    # the frontend renderer (data-chunk-index attribute).  It is the canonical
    # key for paragraph lookup and avoids any string parsing of
    # current_paragraph_id ("chunk-{db_pk}").
    _chunk_idx_raw = _last_b.get("current_chunk_index")
    reading_position: dict[str, Any] = {
        "current_paragraph_id": _last_b.get("current_paragraph_id"),
        "current_chunk_index": int(_chunk_idx_raw) if _chunk_idx_raw is not None else None,
        "viewport_progress_ratio": float(
            _last_b.get("viewport_progress_ratio") or 0.0
        ),
        # True when the reading layout was reformatted during this window.
        # Signals the drift model and RF classifier that behavioural patterns
        # may differ from the trained distribution — used for down-weighting.
        "text_modified": _text_modified_count > 0,
        "text_modified_batch_count": _text_modified_count,
    }

    return {
        # ── Session identity ─────────────────────────────────────────────
        "session_id": ctx.get("session_id"),
        "user_id": ctx.get("user_id"),
        "document_id": ctx.get("document_id"),
        "session_mode": ctx.get("session_mode"),
        # ── Drift state ───────────────────────────────────────────────────
        "drift": {
            "drift_level": result.drift_level,
            "drift_ema": result.drift_ema,
            "beta_effective": result.beta_effective,
            "beta_ema": result.beta_ema,
            "disruption_score": result.disruption_score,
            "engagement_score": result.engagement_score,
            "confidence": result.confidence,
            "pace_ratio": result.pace_ratio,
            "pace_available": result.pace_available,
        },
        # ── Window features (full dataclass → dict) ───────────────────────
        "features": dataclasses.asdict(result.features),
        # ── Z-scores vs baseline ──────────────────────────────────────────
        "z_scores": dataclasses.asdict(result.z_scores),
        # ── Debug breakdown (optional, toggled by include_debug) ──────────
        "debug": result.beta_components,
        # ── Calibration baseline (self-contained for training) ────────────
        "baseline_snapshot": baseline_snapshot,
        # ── UI context aggregates over the 30-second window ───────────────
        "ui_aggregates": ui_agg,
        # ── Reading position (intervention LLM text lookup key) ───────────
        "reading_position": reading_position,
    }


async def get_drift_state(
    session_id: int,
    db: AsyncSession,
) -> SessionDriftState | None:
    """Return the current drift state row or None if it does not exist yet."""
    result = await db.execute(
        select(SessionDriftState).where(SessionDriftState.session_id == session_id)
    )
    return result.scalar_one_or_none()
