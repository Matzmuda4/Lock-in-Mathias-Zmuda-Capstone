from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


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
