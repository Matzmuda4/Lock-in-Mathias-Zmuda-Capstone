# Phase 2 — Backend API (Auth, Documents, Sessions, Telemetry)

**Branch:** `setup`
**Completed:** 2026-02-25
**Commit:** `feat: Phase 2 — full backend API with auth, documents, sessions, activity`

---

## Overview

Phase 2 builds the complete FastAPI backend on top of the TimescaleDB database
established in Phase 1. By the end, every endpoint is callable through Swagger UI at
`http://localhost:8000/docs` and backed by 56 passing pytest tests.

The two-mode architecture (Baseline / Adaptive) is baked in from the start — every
session is created with an explicit `mode` field, and both modes produce identical
telemetry so experiments are directly comparable.

---

## Architecture Decisions

### SOLID principles applied

| Principle | Decision |
|---|---|
| **S** — Single responsibility | One module per concern: `security.py` owns only bcrypt/JWT, `deps.py` owns only FastAPI dependencies, each router owns only one resource |
| **O** — Open/closed | Pydantic schemas use base → response inheritance; adding a new field only requires extending the schema |
| **L** — Liskov | All schema response types are proper `ConfigDict(from_attributes=True)` models that substitute cleanly |
| **I** — Interface segregation | `get_db` provides only a DB session; `get_current_user` provides only a User — they are composed, never merged |
| **D** — Dependency inversion | Routers depend on `get_db` and `get_current_user` abstractions injected via `Depends()`, not concrete instances |

### All ORM models in one file

`db/models.py` contains all six models. Splitting models into per-file modules
causes circular import pain in SQLAlchemy's relationship resolution. One file
eliminates this entirely.

### NullPool for the async engine

`create_async_engine(..., poolclass=NullPool)` was chosen instead of the default
connection pool. NullPool creates a fresh connection per request and discards it
immediately. Reasons:

1. **Test isolation** — asyncpg's connection pool binds to the event loop at first
   use. pytest gives each test function its own event loop. NullPool means no loop
   references are cached, eliminating "Future attached to a different loop" errors.
2. **Correctness for a local tool** — the application has one user. Connection pool
   overhead is a net negative (pool management cost > savings from reuse).

### bcrypt pinned to 4.2.1

`passlib 1.7.4` (the last stable passlib release) is incompatible with `bcrypt>=5.0`
which added a strict 72-byte password limit check. Pinned `bcrypt==4.2.1` to preserve
compatibility until passlib is replaced in a future phase.

---

## Files Created / Modified

| File | Change | Purpose |
|---|---|---|
| `app/db/base.py` | NEW | `DeclarativeBase` — single source of truth for all models |
| `app/db/models.py` | NEW | All 6 ORM models: User, Document, Session, ActivityEvent, ModelOutput, Intervention |
| `app/db/engine.py` | NEW | `create_async_engine` (NullPool) + `init_db()` startup function |
| `app/db/session.py` | NEW | `async_sessionmaker` + `get_db` dependency |
| `app/core/security.py` | NEW | `hash_password`, `verify_password`, `create_access_token`, `decode_access_token` |
| `app/core/deps.py` | NEW | `get_current_user` FastAPI dependency |
| `app/core/config.py` | UPDATED | Added `upload_dir: Path` setting |
| `app/schemas/auth.py` | NEW | `UserRegister`, `UserLogin`, `TokenResponse`, `UserResponse` |
| `app/schemas/documents.py` | NEW | `DocumentResponse`, `DocumentListResponse` |
| `app/schemas/sessions.py` | NEW | `SessionCreate`, `SessionResponse`, `SessionListResponse` |
| `app/schemas/activity.py` | NEW | `ActivityEventCreate`, `ActivityEventResponse` |
| `app/routers/auth.py` | NEW | `POST /auth/register`, `POST /auth/login`, `GET /auth/me` |
| `app/routers/documents.py` | NEW | `POST /documents/upload`, `GET /documents`, `GET /documents/{id}/file`, `DELETE /documents/{id}` |
| `app/routers/sessions.py` | NEW | `POST /sessions/start`, `POST /sessions/{id}/pause`, `POST /sessions/{id}/resume`, `POST /sessions/{id}/end`, `GET /sessions`, `GET /sessions/{id}` |
| `app/routers/activity.py` | NEW | `POST /activity` |
| `app/main.py` | UPDATED | `lifespan` context manager (replaces `@on_event`), routers registered |
| `requirements.txt` | UPDATED | Added `email-validator`, `aiofiles`, pinned `bcrypt==4.2.1` |
| `pytest.ini` | UPDATED | `asyncio_default_fixture_loop_scope = function`, passlib warning suppressed |
| `tests/conftest.py` | UPDATED | Session-sync `init_test_db`, function-sync `clean_tables`, `auth_headers` fixture |
| `tests/test_auth.py` | NEW | 9 auth tests |
| `tests/test_documents.py` | NEW | 10 document tests |
| `tests/test_sessions.py` | NEW | 14 session tests |
| `tests/test_activity.py` | NEW | 8 activity tests |

---

## Database Schema

All tables created by `init_db()` at startup via `Base.metadata.create_all`.

```
users
  id            SERIAL PRIMARY KEY
  username      VARCHAR(50)  UNIQUE  NOT NULL
  email         VARCHAR(255) UNIQUE  NOT NULL
  password_hash VARCHAR(255)         NOT NULL
  created_at    TIMESTAMPTZ  DEFAULT now()

documents
  id          SERIAL PRIMARY KEY
  user_id     INT → users.id  (CASCADE DELETE)
  title       VARCHAR(255)
  filename    VARCHAR(255)
  file_path   VARCHAR(500)
  file_size   BIGINT
  uploaded_at TIMESTAMPTZ DEFAULT now()

sessions
  id               SERIAL PRIMARY KEY
  user_id          INT → users.id     (CASCADE DELETE)
  document_id      INT → documents.id (CASCADE DELETE)
  name             VARCHAR(255)
  mode             VARCHAR(20)   -- 'baseline' | 'adaptive'
  status           VARCHAR(20)   -- 'active' | 'paused' | 'ended'
  started_at       TIMESTAMPTZ DEFAULT now()
  ended_at         TIMESTAMPTZ NULL
  duration_seconds INT NULL
  created_at       TIMESTAMPTZ DEFAULT now()
  updated_at       TIMESTAMPTZ DEFAULT now()

activity_events   ← TimescaleDB HYPERTABLE partitioned by created_at
  id           BIGSERIAL
  user_id      INT → users.id    (CASCADE DELETE)
  session_id   INT → sessions.id (CASCADE DELETE)
  event_type   VARCHAR(50)
  payload      JSONB
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
  PRIMARY KEY (id, created_at)   ← composite required by TimescaleDB

model_outputs
  id           SERIAL PRIMARY KEY
  session_id   INT → sessions.id (CASCADE DELETE)
  output_type  VARCHAR(50)
  payload      JSONB
  created_at   TIMESTAMPTZ DEFAULT now()

interventions
  id         SERIAL PRIMARY KEY
  session_id INT → sessions.id (CASCADE DELETE)
  type       VARCHAR(50)
  intensity  VARCHAR(20)
  payload    JSONB
  created_at TIMESTAMPTZ DEFAULT now()
```

### init_db() startup sequence

```python
# 1. TimescaleDB extension — uses raw asyncpg (DDL outside transaction)
await asyncpg.connect(dsn).execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

# 2. Create all tables
await engine.begin().run_sync(Base.metadata.create_all)

# 3. Convert activity_events to hypertable
SELECT create_hypertable('activity_events', 'created_at', if_not_exists => TRUE)
```

All three steps are idempotent — safe to call on every cold start.

---

## API Endpoints

### Auth (`/auth`)

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/register` | No | Register user, returns JWT |
| POST | `/auth/login` | No | Login (form-encoded), returns JWT. Compatible with Swagger Authorize button. |
| GET  | `/auth/me` | Bearer | Return current user profile |

### Documents (`/documents`)

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/documents/upload` | Bearer | Upload PDF (multipart), `title` + `file` fields |
| GET  | `/documents` | Bearer | List authenticated user's documents |
| GET  | `/documents/{id}/file` | Bearer | Stream PDF file back |
| DELETE | `/documents/{id}` | Bearer | Delete DB record + file from disk |

Files are stored under `uploads/{user_id}/{uuid}_{filename}`.

### Sessions (`/sessions`)

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/sessions/start` | Bearer | Create session (`document_id`, `name`, `mode`) |
| POST | `/sessions/{id}/pause` | Bearer | Set status → `paused` |
| POST | `/sessions/{id}/resume` | Bearer | Set status → `active` |
| POST | `/sessions/{id}/end` | Bearer | Set status → `ended`, record `ended_at` + `duration_seconds` |
| GET  | `/sessions` | Bearer | List user's sessions |
| GET  | `/sessions/{id}` | Bearer | Get session by ID |

Status transitions: `active` → `paused` → `active` → `ended`

### Activity (`/activity`)

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/activity` | Bearer | Ingest one telemetry event into the hypertable |

Valid `event_type` values: `scroll_forward`, `scroll_backward`, `idle`, `blur`, `focus`, `heartbeat`

The `created_at` field is optional — if omitted the server sets it to `now()`. When set by the client it supports buffered/offline event delivery.

---

## Running the API Manually

```bash
cd services/api
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` → click **Authorize** → enter credentials from
`POST /auth/login` to test every endpoint interactively.

---

## Test Suite

Run from `services/api/`:

```bash
.venv/bin/python -m pytest -v --tb=short
```

### Test isolation strategy

- `init_test_db` — **synchronous**, session-scoped. Runs once via `asyncio.run()`
  before any test. Creates the schema in an isolated temporary event loop.
- `clean_tables` — **synchronous**, function-scoped, autouse. Runs `asyncio.run()`
  after each test to DELETE all rows in reverse FK order. Hypertable rows are
  deleted with a plain `DELETE FROM activity_events` (TimescaleDB supports this).
- Each async test function gets its **own event loop** (`asyncio_default_fixture_loop_scope = function`).
- NullPool means no connections are cached between test loops.

### Results

```
============================= test session starts ==============================
collected 56 items

tests/test_activity.py   8 passed
tests/test_auth.py       9 passed
tests/test_db_connection.py  7 passed
tests/test_documents.py 10 passed
tests/test_health.py     6 passed
tests/test_sessions.py  14 passed

============================== 2 warnings summary =============================
56 passed in 27.47s
```

**All 56 tests passed.**

---

## Known Issues / Deferred

| Item | Status |
|---|---|
| Alembic migrations | Deferred — `create_all` is used for now; Alembic will be added before deployment |
| Session duration ignores paused time | Intentional simplification for the thesis prototype |
| `look_away` event type (camera) | Intentionally excluded — not implemented yet |
| `passlib` / `crypt` deprecation warning | Suppressed in `pytest.ini`; will be replaced with `argon2-cffi` in a later phase |

---

## What Is Next

- **Frontend branch** — Tauri v2 desktop shell + React 18 PDF reader
- Phase 3 (deferred): Alembic migration files committed alongside model changes
- Phase 9: Attention model → intervention loop
