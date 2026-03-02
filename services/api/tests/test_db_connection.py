"""
Infrastructure tests — verify the TimescaleDB container is reachable
and the timescaledb extension is installed.

These tests use a raw asyncpg connection (no ORM) so they work before
any schema or migrations exist.
"""

import pytest


class TestDatabaseConnectivity:
    async def test_can_connect(self, db_conn):
        """A basic SELECT 1 proves TCP + auth work."""
        result = await db_conn.fetchval("SELECT 1")
        assert result == 1, "Expected SELECT 1 to return 1"

    async def test_correct_database(self, db_conn):
        """Verify we are connected to the test database (lockin_test)."""
        dbname = await db_conn.fetchval("SELECT current_database()")
        assert dbname == "lockin_test", f"Expected database 'lockin_test', got '{dbname}'"

    async def test_correct_user(self, db_conn):
        """Verify we are authenticated as the 'lockin' role."""
        user = await db_conn.fetchval("SELECT current_user")
        assert user == "lockin", f"Expected user 'lockin', got '{user}'"

    async def test_postgres_version_is_16(self, db_conn):
        """TimescaleDB image ships with PG 16."""
        version = await db_conn.fetchval("SELECT current_setting('server_version_num')::int")
        major = version // 10000
        assert major == 16, f"Expected PostgreSQL 16, got major version {major}"

    async def test_timescaledb_extension_available(self, db_conn):
        """timescaledb must appear in pg_available_extensions."""
        row = await db_conn.fetchrow(
            "SELECT name FROM pg_available_extensions WHERE name = 'timescaledb'"
        )
        assert row is not None, (
            "timescaledb extension not found in pg_available_extensions. "
            "Is the timescale/timescaledb Docker image running?"
        )

    async def test_timescaledb_extension_installed(self, db_conn):
        """
        TimescaleDB auto-installs itself when the container starts.
        Verify it is present in pg_extension.
        """
        row = await db_conn.fetchrow(
            "SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"
        )
        assert row is not None, (
            "timescaledb extension is NOT installed in the 'lockin' database. "
            "Run: CREATE EXTENSION IF NOT EXISTS timescaledb;"
        )

    async def test_timescaledb_version(self, db_conn):
        """Call timescaledb_information to confirm the extension is functional."""
        version = await db_conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
        )
        assert version is not None and len(version) > 0, (
            "Could not read timescaledb version from pg_extension"
        )
        print(f"\n  TimescaleDB version: {version}")
