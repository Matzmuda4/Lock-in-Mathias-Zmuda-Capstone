from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class CalibrationStatus(BaseModel):
    """Returned by GET /calibration/status."""

    has_baseline: bool
    calib_available: bool
    # "none" | "pending" | "running" | "succeeded" | "failed"
    parse_status: str


class CalibrationStartResponse(BaseModel):
    """Returned by POST /calibration/start."""

    model_config = ConfigDict(from_attributes=True)

    session_id: int
    document_id: int


class CalibrationCompleteRequest(BaseModel):
    session_id: int


class BaselineData(BaseModel):
    """
    Shape of baseline_json stored in user_baselines (Phase 6 v2).

    All v1 legacy fields remain optional for backwards compatibility.
    """

    model_config = ConfigDict(extra="allow")

    # ── WPM ──────────────────────────────────────────────────────────────────
    wpm_gross: float = 0.0
    wpm_effective: float = 0.0
    words_read_estimated: int = 0
    effective_reading_seconds: float = 0.0

    # ── Scroll velocity ───────────────────────────────────────────────────────
    scroll_velocity_px_s_mean: float = 0.0
    scroll_velocity_px_s_std: float = 0.0
    scroll_velocity_norm_mean: float = 0.0
    scroll_velocity_norm_std: float = 0.0

    # ── Jitter ────────────────────────────────────────────────────────────────
    scroll_jitter_mean: float = 0.0
    scroll_jitter_std: float = 0.0

    # ── Idle ──────────────────────────────────────────────────────────────────
    idle_ratio_mean: float = 0.0
    idle_ratio_std: float = 0.0
    idle_seconds_mean: float = 0.0
    idle_seconds_std: float = 0.0

    # ── Regress rate ──────────────────────────────────────────────────────────
    regress_rate_mean: float = 0.0
    regress_rate_std: float = 0.0

    # ── Paragraph dwell distribution ──────────────────────────────────────────
    para_dwell_mean_s: float = 0.0
    para_dwell_median_s: float = 0.0
    para_dwell_iqr_s: float = 0.0
    paragraph_count_observed: int = 0

    # ── Presentation profile ──────────────────────────────────────────────────
    presentation_profile: Optional[dict[str, Any]] = None

    # ── Duration ─────────────────────────────────────────────────────────────
    calibration_duration_seconds: int = 0

    # ── Legacy aliases (v1 keys kept for backwards compatibility) ─────────────
    wpm_mean: float = 0.0
    wpm_std: float = 0.0
    scroll_velocity_mean: float = 0.0
    scroll_velocity_std: float = 0.0
    paragraph_dwell_mean: float = 0.0
    regress_rate_mean_legacy: float = 0.0


class CalibrationCompleteResponse(BaseModel):
    """Returned by POST /calibration/complete."""

    baseline: BaselineData
    completed_at: datetime
    session_id: int


class UserBaselineResponse(BaseModel):
    """Returned when fetching an existing baseline."""

    model_config = ConfigDict(from_attributes=True)

    user_id: int
    baseline_json: dict
    completed_at: datetime
    updated_at: datetime


class CalibrationSessionInfo(BaseModel):
    """Minimal session info returned by GET /calibration/session/{id}."""

    id: int
    status: str
    mode: str
    started_at: Optional[datetime]
    elapsed_seconds: int


class CalibrationReaderResponse(BaseModel):
    """Returned by GET /calibration/session/{session_id}."""

    session: CalibrationSessionInfo
    # Plain text paragraphs, read directly from the .txt file.
    paragraphs: list[str]
    total_words: int
