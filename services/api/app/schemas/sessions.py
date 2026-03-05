from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class SessionCreate(BaseModel):
    document_id: int
    name: str
    mode: Literal["baseline", "adaptive", "calibration"]


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    document_id: int
    name: str
    mode: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    duration_seconds: Optional[int]
    elapsed_seconds: int
    created_at: datetime


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int
