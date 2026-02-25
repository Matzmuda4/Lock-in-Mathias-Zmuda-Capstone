import os
import pytest
import asyncpg
from httpx import AsyncClient, ASGITransport

# Point at the test DB (same as dev for now — port 5433)
TEST_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lockin:lockin@localhost:5433/lockin",
).replace("postgresql+asyncpg://", "postgresql://")


@pytest.fixture(scope="session")
def db_dsn() -> str:
    return TEST_DB_URL


@pytest.fixture()
async def db_conn(db_dsn):
    """Raw asyncpg connection — used only for infrastructure tests."""
    conn = await asyncpg.connect(db_dsn)
    yield conn
    await conn.close()


@pytest.fixture()
async def api_client():
    """HTTPX async client wired directly to the FastAPI app (no server needed)."""
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
