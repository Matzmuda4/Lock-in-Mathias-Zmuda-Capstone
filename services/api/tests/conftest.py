"""
Shared fixtures for the entire test suite.

Tests use the dedicated "lockin_test" database so the autouse clean_tables
fixture never touches production data.  DATABASE_URL is set at module level
(before any app module is imported) so pydantic-settings picks it up from
the environment rather than from .env.

Design notes
------------
* init_test_db is SYNCHRONOUS (scope="session") -- it calls asyncio.run() so
  it gets its own temporary event loop for schema setup.
* clean_tables is SYNCHRONOUS for the same reason.
* db_conn / api_client / auth_headers are async + function-scoped; NullPool
  ensures no connections are shared between tests.
"""

import asyncio
import os

# Must happen BEFORE any app import so the SQLAlchemy engine and pydantic
# Settings are created with the test URL, not the live URL from .env.
os.environ["DATABASE_URL"] = "postgresql+asyncpg://lockin:lockin@localhost:5433/lockin_test"

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

# Tests use a dedicated database — never the live "lockin" database.
# This prevents the autouse clean_tables fixture from wiping production data.
_ASYNCPG_DSN = "postgresql://lockin:lockin@localhost:5433/lockin_test"
_SQLALCHEMY_DSN = "postgresql+asyncpg://lockin:lockin@localhost:5433/lockin_test"


# ── Schema bootstrap — once per session, synchronous ─────────────────────────

@pytest.fixture(scope="session", autouse=True)
def init_test_db() -> None:
    """
    Create the TimescaleDB extension, all tables, and the activity_events
    hypertable.  Runs once before any test using asyncio.run() so it has
    its own isolated event loop and does not interfere with per-test loops.
    """
    async def _setup() -> None:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool
        from sqlalchemy import text
        from app.db.base import Base
        import app.db.models  # noqa: F401 — registers models with Base.metadata

        # Step 1 — TimescaleDB extension
        conn = await asyncpg.connect(_ASYNCPG_DSN)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        finally:
            await conn.close()

        # Step 2 & 3 — Tables + hypertable (use a fresh NullPool engine)
        eng = create_async_engine(_SQLALCHEMY_DSN, poolclass=NullPool)
        async with eng.begin() as c:
            await c.run_sync(Base.metadata.create_all)
        # Idempotent column additions for existing DBs (mirrors engine.py init_db)
        async with eng.begin() as c:
            await c.execute(
                text(
                    "ALTER TABLE documents "
                    "ADD COLUMN IF NOT EXISTS is_calibration BOOLEAN NOT NULL DEFAULT FALSE"
                )
            )
        for col_sql in [
            "ADD COLUMN IF NOT EXISTS beta_ema DOUBLE PRECISION NOT NULL DEFAULT 0.0",
            "ADD COLUMN IF NOT EXISTS drift_level DOUBLE PRECISION NOT NULL DEFAULT 0.0",
            "ADD COLUMN IF NOT EXISTS disruption_score DOUBLE PRECISION NOT NULL DEFAULT 0.0",
            "ADD COLUMN IF NOT EXISTS engagement_score DOUBLE PRECISION NOT NULL DEFAULT 0.0",
        ]:
            async with eng.begin() as c:
                await c.execute(
                    text(f"ALTER TABLE session_drift_states {col_sql}")
                )
        async with eng.begin() as c:
            await c.execute(
                text(
                    "SELECT create_hypertable("
                    "  'activity_events', 'created_at', if_not_exists => TRUE"
                    ")"
                )
            )
        await eng.dispose()

    asyncio.run(_setup())


# ── Table cleanup — after every test, synchronous ────────────────────────────

@pytest.fixture(autouse=True)
def clean_tables() -> None:
    """
    Wipe all rows after each test in reverse FK-dependency order.
    Synchronous so it runs in its own asyncio.run() loop rather than the
    test's event loop (which may already be closed by teardown time).
    """
    yield

    async def _cleanup() -> None:
        conn = await asyncpg.connect(_ASYNCPG_DSN)
        try:
            await conn.execute("DELETE FROM activity_events")
            await conn.execute("DELETE FROM session_drift_states")
            await conn.execute("DELETE FROM model_outputs")
            await conn.execute("DELETE FROM interventions")
            await conn.execute("DELETE FROM sessions")
            # Phase 4 tables (FK → documents)
            await conn.execute("DELETE FROM document_assets")
            await conn.execute("DELETE FROM document_chunks")
            await conn.execute("DELETE FROM document_parse_jobs")
            await conn.execute("DELETE FROM documents")
            # Phase B tables (FK → users)
            await conn.execute("DELETE FROM user_baselines")
            await conn.execute("DELETE FROM users")
        finally:
            await conn.close()

    asyncio.run(_cleanup())


# ── Raw asyncpg connection (for test_db_connection.py) ───────────────────────

@pytest.fixture()
async def db_conn() -> asyncpg.Connection:
    conn = await asyncpg.connect(_ASYNCPG_DSN)
    yield conn
    await conn.close()


# ── HTTPX async client ────────────────────────────────────────────────────────

@pytest.fixture()
async def api_client() -> AsyncClient:
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ── Register a fresh user and return its Bearer header ───────────────────────

@pytest.fixture()
async def auth_headers(api_client: AsyncClient) -> dict[str, str]:
    resp = await api_client.post(
        "/auth/register",
        json={
            "username": "testuser",
            "email": "testuser@example.com",
            "password": "password123",
        },
    )
    assert resp.status_code == 201, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}
