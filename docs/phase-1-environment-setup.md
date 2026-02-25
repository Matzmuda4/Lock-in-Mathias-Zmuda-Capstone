# Phase 1 — Environment & Infrastructure Setup

**Branch:** `setup`
**Completed:** 2026-02-25
**Commit:** `feat: Phase 1 — monorepo scaffold, TimescaleDB, FastAPI skeleton`

---

## Overview

This phase establishes the full local development environment for the Lock-In project:
prerequisite tooling, a containerised TimescaleDB database, the Python/FastAPI backend
skeleton, and the pnpm monorepo structure. Everything here is done once and forms the
foundation that every subsequent phase builds on.

---

## Step 1.1 — Install Prerequisites

All of these were installed manually on the developer machine before any project files
were created.

### Node.js 20 via nvm

```bash
nvm install 20
nvm use 20
node --version   # v20.x.x
```

### pnpm

```bash
npm install -g pnpm
pnpm --version   # 9.x.x
```

### Python 3.11

```bash
# If using pyenv:
pyenv install 3.11
pyenv global 3.11

python --version   # Python 3.11.x
```

### Docker Desktop

Downloaded and installed from [https://docker.com](https://docker.com).
Confirmed running via the whale icon in the macOS menu bar.

### Rust + Tauri CLI

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# restart shell to pick up cargo in PATH, then:
cargo install tauri-cli
cargo tauri --version   # tauri-cli 2.x.x
```

> Compilation of tauri-cli takes ~6 minutes on Apple Silicon (M-series) —
> this is normal; 708 crates are compiled from source.

---

## Step 1.2 — Start TimescaleDB

A `docker-compose.yml` was created at the repo root instead of a bare `docker run`
command. This makes the container declarative, reproducible, and easy to bring up/down
with a single command.

**`docker-compose.yml` key settings:**

| Setting | Value |
|---|---|
| Image | `timescale/timescaledb:latest-pg16` |
| Container name | `lockin-db` |
| Host port | `5433` (avoids conflict with any local PostgreSQL on 5432) |
| Container port | `5432` |
| `POSTGRES_USER` | `lockin` |
| `POSTGRES_PASSWORD` | `lockin` |
| `POSTGRES_DB` | `lockin` |
| Volume | `lockin_pgdata` (named Docker volume, persists data across restarts) |
| Healthcheck | `pg_isready -U lockin -d lockin` every 10 s |

**Commands run:**

```bash
# Start the container in detached mode
docker compose up -d

# Verify it is running and healthy
docker compose ps
```

Expected `docker compose ps` output:
```
NAME        IMAGE                               STATUS
lockin-db   timescale/timescaledb:latest-pg16   Up (healthy)   0.0.0.0:5433->5432/tcp
```

> The first run also creates the Docker network
> `lock-in-mathias-zmuda-capstone_default` and the named volume `lockin_pgdata`.
> A "Conflict" error appeared on the very first attempt because an older container from a
> previous experiment still existed; running `docker compose up -d` a second time resolved
> it automatically (Docker compose reused the existing container).

**Useful operational commands (available as pnpm scripts):**

```bash
pnpm db:up      # docker compose up -d
pnpm db:down    # docker compose down
pnpm db:logs    # docker compose logs -f db
pnpm db:shell   # docker exec -it lockin-db psql -U lockin -d lockin
```

---

## Step 1.3 — Create the Monorepo

### Directory structure created

```
lock-in/
├── apps/
│   └── desktop/            ← React 18 + Tauri v2 (populated in a later phase)
├── docs/                   ← Project documentation (this file)
├── packages/
│   └── shared/
│       ├── package.json
│       └── src/
│           └── index.ts    ← Shared TypeScript types (SessionMode, ActivityEventType)
├── services/
│   └── api/
│       ├── app/
│       │   ├── __init__.py
│       │   ├── main.py           ← FastAPI app entry point
│       │   ├── core/
│       │   │   └── config.py     ← pydantic-settings (reads .env)
│       │   ├── db/               ← (populated in Phase 2 — migrations)
│       │   ├── models/           ← (populated in Phase 2 — SQLAlchemy models)
│       │   ├── routers/          ← (populated in Phase 3 — auth, etc.)
│       │   └── schemas/          ← (populated in Phase 3 — Pydantic schemas)
│       ├── tests/
│       │   ├── conftest.py
│       │   ├── test_db_connection.py
│       │   └── test_health.py
│       ├── pytest.ini
│       └── requirements.txt
├── training/               ← LLM fine-tuning pipeline (Phase 10)
├── .env                    ← Local secrets (git-ignored)
├── .env.example            ← Committed template
├── .gitignore
├── docker-compose.yml
├── package.json            ← Root pnpm scripts
└── pnpm-workspace.yaml
```

**Shell commands used to create directories:**

```bash
mkdir -p services/api/app/{routers,models,schemas,core,db} \
         services/api/tests \
         apps/desktop \
         packages/shared/src \
         training \
         docs
```

### Python virtual environment

```bash
cd services/api
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

> `pip install` was not available directly (no `/usr/local/bin/pip3.11`);
> `pip3 install` was used instead. Both install into the same `.venv`.

### Environment file

```bash
# From repo root
cp .env.example .env
```

The `.env` file is git-ignored. The `.env.example` is committed and serves as the
canonical reference for required variables.

**Variables in `.env`:**

```
POSTGRES_USER=lockin
POSTGRES_PASSWORD=lockin
POSTGRES_DB=lockin
DATABASE_URL=postgresql+asyncpg://lockin:lockin@localhost:5433/lockin

SECRET_KEY=change-me-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

DEBUG=false
```

---

## Files Created (by path)

| File | Purpose |
|---|---|
| `docker-compose.yml` | Declarative TimescaleDB container definition |
| `pnpm-workspace.yaml` | Declares `apps/*` and `packages/*` as pnpm workspace packages |
| `package.json` | Root-level pnpm scripts (`db:up`, `db:down`, `dev:api`, etc.) |
| `.env.example` | Template for all required environment variables |
| `.gitignore` | Ignores `.env`, `.venv`, `node_modules`, Rust `target/`, etc. |
| `packages/shared/package.json` | Marks shared package as `@lock-in/shared` |
| `packages/shared/src/index.ts` | `SessionMode` and `ActivityEventType` TS types |
| `services/api/requirements.txt` | Pinned Python dependencies |
| `services/api/app/main.py` | FastAPI app with `/health` endpoint and CORS middleware |
| `services/api/app/core/config.py` | `Settings` class (pydantic-settings, reads `.env`) |
| `services/api/pytest.ini` | pytest config (`asyncio_mode = auto`, `testpaths = tests`) |
| `services/api/tests/conftest.py` | Shared fixtures: `db_conn` (asyncpg), `api_client` (HTTPX) |
| `services/api/tests/test_db_connection.py` | 7 DB infrastructure tests |
| `services/api/tests/test_health.py` | 6 FastAPI endpoint tests |

---

## Python Dependencies (`requirements.txt`)

| Package | Version | Role |
|---|---|---|
| `fastapi` | 0.115.6 | Web framework |
| `uvicorn[standard]` | 0.32.1 | ASGI server |
| `asyncpg` | 0.30.0 | Async PostgreSQL driver |
| `sqlalchemy[asyncio]` | 2.0.36 | ORM (async mode) |
| `alembic` | 1.14.0 | Database migrations |
| `python-jose[cryptography]` | 3.3.0 | JWT encoding/decoding |
| `passlib[bcrypt]` | 1.7.4 | Password hashing |
| `python-multipart` | 0.0.20 | File upload support |
| `pydantic` | 2.10.4 | Data validation |
| `pydantic-settings` | 2.7.0 | Settings from env/`.env` |
| `pytest` | 8.3.4 | Test runner |
| `pytest-asyncio` | 0.25.0 | Async test support |
| `httpx` | 0.28.1 | Async HTTP client (tests) |

---

## Test Results

Run from `services/api/` with:

```bash
.venv/bin/python -m pytest -v --tb=short
```

```
============================= test session starts ==============================
platform darwin -- Python 3.11.10, pytest-8.3.4
asyncio: mode=Mode.AUTO, asyncio_default_fixture_loop_scope=function
collected 13 items

tests/test_db_connection.py::TestDatabaseConnectivity::test_can_connect              PASSED
tests/test_db_connection.py::TestDatabaseConnectivity::test_correct_database         PASSED
tests/test_db_connection.py::TestDatabaseConnectivity::test_correct_user             PASSED
tests/test_db_connection.py::TestDatabaseConnectivity::test_postgres_version_is_16   PASSED
tests/test_db_connection.py::TestDatabaseConnectivity::test_timescaledb_extension_available PASSED
tests/test_db_connection.py::TestDatabaseConnectivity::test_timescaledb_extension_installed PASSED
tests/test_db_connection.py::TestDatabaseConnectivity::test_timescaledb_version      PASSED
tests/test_health.py::TestHealthEndpoint::test_health_returns_200                    PASSED
tests/test_health.py::TestHealthEndpoint::test_health_payload_has_status_ok          PASSED
tests/test_health.py::TestHealthEndpoint::test_health_payload_has_version            PASSED
tests/test_health.py::TestHealthEndpoint::test_docs_endpoint_accessible              PASSED
tests/test_health.py::TestHealthEndpoint::test_openapi_schema_accessible             PASSED
tests/test_health.py::TestHealthEndpoint::test_cors_header_present_for_tauri_origin  PASSED

============================== 13 passed in 0.54s ==============================
```

**All 13 tests passed.**

---

## pgAdmin Access

To inspect the database visually using pgAdmin 4:

1. Open pgAdmin → right-click **Servers** → **Register → Server**
2. **General tab** — Name: `Lock-In (local)`
3. **Connection tab:**

| Field | Value |
|---|---|
| Host name/address | `localhost` |
| Port | `5433` |
| Maintenance database | `lockin` |
| Username | `lockin` |
| Password | `lockin` |

---

## What Is NOT Done Yet

The following are explicitly deferred to later phases:

- SQLAlchemy models and Alembic migrations (Phase 2)
- TimescaleDB hypertable creation for `activity_events` (Phase 2)
- Auth endpoints — register, login, JWT (Phase 3)
- Document upload (Phase 4)
- Session management (Phase 5)
- Telemetry ingestion (Phase 6)
- Tauri desktop shell (Phase 7)
- PDF reader UI (Phase 8)
- Attention model and interventions (Phase 9)
- LLM fine-tuning pipeline (Phase 10)
