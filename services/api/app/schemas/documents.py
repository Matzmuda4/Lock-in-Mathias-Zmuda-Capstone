from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    title: str
    filename: str
    file_size: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
