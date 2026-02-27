# Phase 4 — PDF Parsing Layer

Branch: `doclingparse`

## Overview

Phase 4 adds a non-blocking PDF parsing pipeline powered by **docling** (IBM Research).  
After a document is uploaded, the backend parses it asynchronously and stores structured text chunks and extracted images in the database.  
A new **Session Reader** page in the frontend displays the parsed content when a reading session is started.

---

## Architecture

```
Upload PDF
    ↓
POST /documents/upload
    ├── Saves file to uploads/{user_id}/
    ├── Creates Document row
    ├── Creates DocumentParseJob (status=pending)
    └── Schedules BackgroundTask → run_parse_job()
                                       ↓
                            ThreadPoolExecutor (non-blocking)
                                       ↓
                            docling DocumentConverter
                                  ├── Text blocks → DocumentChunk rows
                                  └── Images      → parsed_cache/{doc_id}/ + DocumentAsset rows
                                       ↓
                            DocumentParseJob status → succeeded | failed
```

---

## New Files

### Backend

| File | Purpose |
|---|---|
| `app/services/parsing/__init__.py` | Package marker |
| `app/services/parsing/models.py` | Internal Pydantic types: `TextBlock`, `ImageBlock`, `ParseResult` |
| `app/services/parsing/chunking.py` | Paragraph-aware chunking: merges short consecutive blocks, re-indexes |
| `app/services/parsing/parser.py` | Docling integration, DB persistence, `run_parse_job` entry point |
| `app/schemas/parsing.py` | API response schemas: `ParseJobStatus`, `ChunkResponse`, `AssetSummary`, `ParsedDocumentResponse`, `SessionReaderResponse` |
| `app/routers/parsing.py` | Four new endpoints (see below) |
| `scripts/parse_local_pdf.py` | Dev script for manual verification |

### Frontend

| File | Purpose |
|---|---|
| `apps/desktop/src/pages/ReaderPage.tsx` | Session reader page |

---

## Database Schema Changes

Three new tables added in `app/db/models.py`:

### `document_parse_jobs`
Tracks lifecycle of one docling parse run per document.
- `status`: `pending` → `running` → `succeeded` | `failed`
- `error`: populated on failure
- `started_at`, `finished_at`: timestamps

### `document_chunks`
Stores extracted text blocks.
- `chunk_index`: reading order, 0-based
- `page_start`, `page_end`: source page numbers (1-indexed, from docling)
- `text`: cleaned paragraph text
- `meta` (JSONB): `{label, bbox}` — preserves docling element type and bounding box for future LLM use
- Unique constraint on `(document_id, chunk_index)`

### `document_assets`
Stores extracted images.
- `asset_type`: currently `"image"` (future: `"table"`)
- `page`, `bbox` (JSONB): provenance
- `file_path`: path under `services/api/parsed_cache/{document_id}/asset_{n}.png`
- `meta` (JSONB): extensible

All three tables are created by the existing `init_db()` → `create_all` call at startup.  
No migration needed yet — Alembic support can be layered in later without restructuring.

---

## New API Endpoints

### Parsing Router

| Method | Path | Description |
|---|---|---|
| `GET` | `/documents/{id}/parse-status` | Current job status + timestamps + error |
| `GET` | `/documents/{id}/parsed` | Paginated chunks + asset list (`?offset=&limit=`) |
| `GET` | `/documents/{id}/assets/{asset_id}` | Stream extracted image file |
| `POST` | `/documents/{id}/reparse` | Clear prior output and restart parse job |

### Sessions Router (added)

| Method | Path | Description |
|---|---|---|
| `GET` | `/sessions/{id}/reader` | Session info + parse status + first 30 chunks + all assets |

All new endpoints require authentication and enforce ownership (returns 404 for other users' resources).

---

## Parsing Pipeline Detail

**`app/services/parsing/parser.py`**

- `_sync_parse(file_path, cache_dir)` — runs in `ThreadPoolExecutor`; calls `DocumentConverter` with OCR and table-structure disabled (fast, CPU-only).
- `build_chunks(raw_items)` — (`chunking.py`) merges short consecutive blocks on the same page, re-indexes, strips empty items.
- Images saved as PNG to `services/api/parsed_cache/{document_id}/asset_{n}.png`.
- Full docling document exported as dict and accessible via the raw `raw_doc` field in `ParseResult` (not currently stored in DB, but trivially addable to a JSONB column if needed).

**Why `ThreadPoolExecutor`?**  
Docling is synchronous and CPU-bound. Wrapping it in `asyncio.get_event_loop().run_in_executor()` keeps the FastAPI event loop free during parsing.

**Why `BackgroundTasks`?**  
FastAPI's `BackgroundTasks` runs the task after the response is sent, within the same ASGI process. It's simpler than Celery/Redis for a local single-user app.

---

## Frontend Changes

### `documentService.ts`
New methods: `getParseStatus`, `getParsed`, `reparse`, `getAssetUrl`.

### `sessionService.ts`
New interface `SessionReaderData` and method `getReader`.

### `HomePage.tsx`
- After upload, document immediately appears with a **"Parsing…"** spinner badge.
- Polling starts at 1 s intervals, backing off to 5 s.
- On failure, badge changes to **"Parse failed — retry"** (click to reparse).
- **"Start session"** now navigates directly to `/sessions/:id/reader`.

### `ReaderPage.tsx` (`/sessions/:id/reader`)
- **Top bar**: session name, status badge, mode, elapsed timer, "View original PDF ↗" button.
- **Main column**: chunk cards (page label + text). "Load more" button for pagination.
- **Side panel**: extracted image thumbnails with page labels (hidden if no assets).
- Shows a "still parsing" notice if parse hasn't completed yet.

---

## Running the Application

```bash
# Start everything (DB + API + frontend)
./start.sh

# Or backend only
cd services/api && .venv/bin/uvicorn app.main:app --reload --port 8000

# Install new Python dependencies (first run after pulling)
cd services/api && .venv/bin/pip install -r requirements.txt
```

---

## Manual Parse Verification

Place a test PDF at `./pdfs/test.pdf` (repo root), then:

```bash
cd services/api
source .venv/bin/activate
python scripts/parse_local_pdf.py ../../pdfs/test.pdf
```

Output:

```
[parse] /path/to/pdfs/test.pdf
[cache] /path/to/services/api/parsed_cache/local_test

──────────────────────────────────────────────────
  Chunks : 42
  Assets : 3
──────────────────────────────────────────────────

── First 3 chunks ──
  [0] page=1  'Introduction\nThis paper presents...'
  [1] page=1  'Lorem ipsum dolor sit amet...'
  [2] page=2  'Section 1 — Background...'

── Assets ──
  [0] page=3  path=.../parsed_cache/local_test/asset_0.png
```

Extracted images appear under `services/api/parsed_cache/local_test/` (git-ignored).

---

## Tests

```bash
cd services/api
.venv/bin/pytest tests/ -q
# 68 passed
```

New test file `tests/test_parsing.py` covers:
- Upload creates a pending parse job
- `GET /parse-status` returns expected fields
- Ownership enforced (403/404 for other user)
- `GET /parsed` returns zero chunks before parse
- `GET /sessions/{id}/reader` returns session + parse data
- `POST /reparse` clears prior data and creates fresh job
- Reparse ownership enforced

Docling is **mocked** in all tests — no PDF processing in CI.

---

## Gitignore Additions

```
services/api/parsed_cache/   # extracted images
pdfs/                        # local test PDFs
```

---

## Known Limitations / Deferred

- Images from the reader page are served without the JWT token (browser `<img>` tags can't set headers). For production, swap to pre-fetched `ObjectURL` blobs.
- OCR is disabled (`do_ocr=False`). Enable in `parser.py` for scanned PDFs.
- LLM-driven chunking, drift detection, and interventions are **not** implemented in this phase.
- Alembic migrations not yet configured — schema is managed by `create_all`.
