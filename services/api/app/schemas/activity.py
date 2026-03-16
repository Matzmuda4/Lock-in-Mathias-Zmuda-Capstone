from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ActivityEventCreate(BaseModel):
    session_id: int
    event_type: str
    payload: dict = {}
    # Client may supply its own timestamp (e.g. buffered offline events).
    # If omitted the server sets created_at to now().
    created_at: Optional[datetime] = None


class ActivityEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    session_id: int
    event_type: str
    payload: dict
    created_at: datetime


# ─── Phase 5 — Telemetry batch ────────────────────────────────────────────────

class ActivityBatchCreate(BaseModel):
    """
    One 2-second aggregated telemetry batch from the reader frontend.

    All numeric fields default to 0 so the client only needs to include
    non-zero values.  scroll_pause_seconds and idle_seconds are capped on
    the client side before sending.
    """

    session_id: int

    # ── Scroll signals ──────────────────────────────────────────────────────
    # Net signed scroll displacement (positive = down, negative = up).
    scroll_delta_sum: float = Field(default=0.0)
    # Total absolute displacement (always >= 0).
    scroll_delta_abs_sum: float = Field(default=0.0, ge=0)
    # Sum of positive (downward) deltas only — for regress_rate computation.
    scroll_delta_pos_sum: float = Field(default=0.0, ge=0)
    # Sum of |negative (upward) deltas| only — for regress_rate computation.
    scroll_delta_neg_sum: float = Field(default=0.0, ge=0)
    # Number of discrete scroll events in the window.
    scroll_event_count: int = Field(default=0, ge=0)
    # How many times the scroll direction reversed in the window.
    scroll_direction_changes: int = Field(default=0, ge=0)
    # Seconds since the last scroll event (capped at 60 s on the client).
    scroll_pause_seconds: float = Field(default=0.0, ge=0)

    # ── Engagement signals ──────────────────────────────────────────────────
    # Seconds idle IN THIS 2s window (0..2) — per-window, not cumulative.
    # Server clamps to 2.0 and flags telemetry_fault if > 2.0 received.
    idle_seconds: float = Field(default=0.0, ge=0)
    # Diagnostic: total seconds since last interaction (not used by model).
    idle_since_interaction_seconds: Optional[float] = Field(default=None, ge=0)

    # ── Mouse signals ───────────────────────────────────────────────────────
    # Total physical path the cursor travelled (px).
    mouse_path_px: float = Field(default=0.0, ge=0)
    # Straight-line net displacement from start to end of the window (px).
    mouse_net_px: float = Field(default=0.0, ge=0)

    # ── Window / focus signals ──────────────────────────────────────────────
    # "focused" or "blurred" — the state at the moment the batch is flushed.
    window_focus_state: str = Field(default="focused")

    # ── Reading-position signals ────────────────────────────────────────────
    # data-paragraph-id of the most-visible paragraph element (IntersectionObserver).
    current_paragraph_id: Optional[str] = None
    # chunk_index of the most-visible chunk.
    current_chunk_index: Optional[int] = None
    # scrollTop / (scrollHeight - clientHeight), clamped [0, 1].
    viewport_progress_ratio: float = Field(default=0.0, ge=0, le=1)

    # ── Presentation profile (viewport dimensions for normalisation) ──────────
    # window.innerHeight at flush time.
    viewport_height_px: Optional[float] = Field(default=None, ge=0)
    # window.innerWidth at flush time.
    viewport_width_px: Optional[float] = Field(default=None, ge=0)
    # clientHeight of the scrollable reader container.
    reader_container_height_px: Optional[float] = Field(default=None, ge=0)

    # ── Timestamp ───────────────────────────────────────────────────────────
    # ISO-8601 timestamp from the client; server uses now() if absent.
    client_timestamp: Optional[str] = None


class ActivityBatchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    event_type: str
    created_at: datetime
