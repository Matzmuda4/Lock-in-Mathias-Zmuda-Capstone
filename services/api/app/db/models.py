"""
All ORM models in one module.

Keeping models co-located eliminates circular import risk that arises when
models are split across files and reference each other via relationships.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    baseline: Mapped[Optional["UserBaseline"]] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # True for the shared calibration document; hides it from normal document lists.
    is_calibration: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="documents")
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="document", passive_deletes=True
    )
    parse_job: Mapped[Optional["DocumentParseJob"]] = relationship(
        back_populates="document", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    assets: Mapped[list["DocumentAsset"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)  # baseline | adaptive
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Seconds accumulated across all completed active intervals (used for accurate
    # timer display when sessions are paused and resumed multiple times).
    elapsed_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="sessions")
    document: Mapped["Document"] = relationship(back_populates="sessions")
    # Passive deletes: DB FK cascade handles deletion of time-series rows
    activity_events: Mapped[list["ActivityEvent"]] = relationship(
        back_populates="session", passive_deletes=True
    )
    model_outputs: Mapped[list["ModelOutput"]] = relationship(
        back_populates="session", passive_deletes=True
    )
    interventions: Mapped[list["Intervention"]] = relationship(
        back_populates="session", passive_deletes=True
    )


class ActivityEvent(Base):
    """
    TimescaleDB hypertable — partitioned by created_at.

    The composite primary key (id, created_at) is required because TimescaleDB
    mandates that the partition column be part of every unique/primary constraint.
    """

    __tablename__ = "activity_events"
    __table_args__ = (PrimaryKeyConstraint("id", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    session: Mapped["Session"] = relationship(back_populates="activity_events")


class ModelOutput(Base):
    __tablename__ = "model_outputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    output_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["Session"] = relationship(back_populates="model_outputs")


# ─── Phase 4 — Parsing tables ─────────────────────────────────────────────────


class DocumentParseJob(Base):
    """One-to-one with Document. Tracks the lifecycle of a docling parse run."""

    __tablename__ = "document_parse_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    # pending → running → succeeded | failed
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="parse_job")


class DocumentChunk(Base):
    """A single text block extracted from a document by the parsing pipeline."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk"),
        Index("ix_document_chunks_document_id", "document_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Stores bbox list, label, char offsets — flexible for future LLM use
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")


class DocumentAsset(Base):
    """An extracted image (or future: table) from a document."""

    __tablename__ = "document_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset_type: Mapped[str] = mapped_column(String(20), nullable=False, default="image")
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bbox: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # Path under services/api/parsed_cache/{document_id}/
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="assets")


# ─── End Phase 4 ──────────────────────────────────────────────────────────────


# ─── Phase B — Calibration baseline ──────────────────────────────────────────


class UserBaseline(Base):
    """
    One row per user, upserted at the end of a successful calibration session.

    baseline_json shape (v1):
      wpm_mean, wpm_std, scroll_velocity_mean, scroll_velocity_std,
      scroll_jitter_mean, idle_ratio_mean, regress_rate_mean,
      paragraph_dwell_mean, calibration_duration_seconds
    """

    __tablename__ = "user_baselines"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    baseline_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="baseline")


# ─── End Phase B ───────────────────────────────────────────────────────────────


# ─── Phase 7 — Drift state ────────────────────────────────────────────────────


class SessionDriftState(Base):
    """
    Current drift model output for one session.  Upserted on every
    telemetry batch ingestion.  One row per session.
    """

    __tablename__ = "session_drift_states"

    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    beta_effective: Mapped[float] = mapped_column(nullable=False, default=0.03)
    # Smoothed beta — used as prev_beta_ema on the next cycle
    beta_ema: Mapped[float] = mapped_column(nullable=False, default=0.03)
    attention_score: Mapped[float] = mapped_column(nullable=False, default=1.0)
    drift_score: Mapped[float] = mapped_column(nullable=False, default=0.0)
    drift_ema: Mapped[float] = mapped_column(nullable=False, default=0.0)
    confidence: Mapped[float] = mapped_column(nullable=False, default=0.0)
    last_window_ends_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ─── End Phase 7 ──────────────────────────────────────────────────────────────


class Intervention(Base):
    __tablename__ = "interventions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    intensity: Mapped[str] = mapped_column(String(20), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["Session"] = relationship(back_populates="interventions")
