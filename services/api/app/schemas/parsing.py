from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.sessions import SessionResponse


class ParseJobStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: int
    status: str  # pending | running | succeeded | failed
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str
    meta: dict


class AssetSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    asset_type: str
    page: Optional[int] = None
    bbox: Optional[dict] = None
    meta: dict


class ParsedDocumentResponse(BaseModel):
    document_id: int
    chunks: list[ChunkResponse]
    assets: list[AssetSummary]
    total_chunks: int
    offset: int
    limit: int


class SessionReaderResponse(BaseModel):
    session: SessionResponse
    document_id: int
    parse_status: str
    chunks: list[ChunkResponse]
    assets: list[AssetSummary]
    total_chunks: int
