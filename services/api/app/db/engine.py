"""
Async SQLAlchemy engine and one-time database initialisation.

init_db() is called once at application startup (via lifespan) and is
intentionally idempotent — safe to call on every cold start.
"""

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings

# NullPool creates a fresh connection per request and discards it afterwards.
# This avoids asyncpg's internal event-loop binding — critical for tests where
# each test function runs in its own asyncio event loop.  For a local
# single-user application the per-request overhead is negligible.
engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    poolclass=NullPool,
)


async def init_db() -> None:
    """
    Three-step startup sequence (order is mandatory):

    1. CREATE EXTENSION timescaledb — requires autocommit; use raw asyncpg.
    2. CREATE TABLE ... IF NOT EXISTS — SQLAlchemy create_all handles this.
    3. create_hypertable — convert activity_events; if_not_exists makes it safe
       to call on an already-converted table.
    """
    # Ensure storage directories exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — TimescaleDB extension (needs DDL outside a transaction block)
    raw_dsn = settings.database_url.replace("postgresql+asyncpg", "postgresql")
    conn = await asyncpg.connect(raw_dsn)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
    finally:
        await conn.close()

    # Step 2 — Create all tables (no-op for existing tables)
    from app.db.base import Base
    import app.db.models  # noqa: F401 — registers all models with Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Step 3a — Add is_calibration column to documents (idempotent for existing DBs)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "ALTER TABLE documents "
                "ADD COLUMN IF NOT EXISTS is_calibration BOOLEAN NOT NULL DEFAULT FALSE"
            )
        )

    # Step 3b — Add beta_ema column to session_drift_states (idempotent)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "ALTER TABLE session_drift_states "
                "ADD COLUMN IF NOT EXISTS beta_ema DOUBLE PRECISION NOT NULL DEFAULT 0.03"
            )
        )

    # Step 3 — Convert activity_events to a TimescaleDB hypertable
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "SELECT create_hypertable("
                "  'activity_events', 'created_at',"
                "  if_not_exists => TRUE"
                ")"
            )
        )
