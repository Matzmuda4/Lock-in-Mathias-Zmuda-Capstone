from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CalibrationStatus(BaseModel):
    """Returned by GET /calibration/status."""

    has_baseline: bool
    # True when the calibration PDF file is present on the server.
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
    """Shape of the baseline_json field stored in user_baselines."""

    wpm_mean: float
    wpm_std: float
    scroll_velocity_mean: float
    scroll_velocity_std: float
    scroll_jitter_mean: float
    idle_ratio_mean: float
    regress_rate_mean: float
    paragraph_dwell_mean: float
    calibration_duration_seconds: int


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
