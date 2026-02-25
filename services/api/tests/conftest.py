"""
Shared fixtures for the entire test suite.

Design decisions
----------------
* init_test_db is SYNCHRONOUS (scope="session") — it calls asyncio.run() so
  it gets its own temporary event loop for schema setup.  This avoids having a
  session-scoped async fixture fight with per-test (function-scoped) event loops.

* clean_tables is SYNCHRONOUS for the same reason — teardown needs its own
  loop so it doesn't conflict with whatever loop the just-finished test used.

* db_conn / api_client / auth_headers are async + function-scoped, each
  running inside the test's own event loop.  NullPool on the SQLAlchemy engine
  means no connections are cached between tests, so there is no "attached to
  a different loop" error.
"""

import asyncio

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

_ASYNCPG_DSN = "postgresql://lockin:lockin@localhost:5433/lockin"
_SQLALCHEMY_DSN = "postgresql+asyncpg://lockin:lockin@localhost:5433/lockin"


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
            await conn.execute("DELETE FROM model_outputs")
            await conn.execute("DELETE FROM interventions")
            await conn.execute("DELETE FROM sessions")
            await conn.execute("DELETE FROM documents")
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
