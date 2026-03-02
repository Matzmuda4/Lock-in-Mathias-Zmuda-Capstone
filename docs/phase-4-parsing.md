# Phase 4 — PDF Parsing Layer

Branch: `doclingparse`

## Overview

Phase 4 adds a non-blocking PDF parsing pipeline powered by **docling** (IBM Research).
After a document is uploaded, the backend parses it asynchronously and stores structured
text chunks, images, and tables in the database — in document reading order.
A **Session Reader** page in the frontend displays the parsed content when a reading
session is started.

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
                                  ├── Text blocks   → DocumentChunk rows  (chunk_type="text")
                                  ├── Images        → parsed_cache/{doc_id}/image_{n}.png
                                  │                   DocumentAsset + DocumentChunk (chunk_type="image")
                                  └── Tables        → parsed_cache/{doc_id}/table_{n}.png  +  markdown
                                                      DocumentAsset + DocumentChunk (chunk_type="table")
                                       ↓
                            DocumentParseJob status → succeeded | failed
```

All three content types (text, image, table) are stored in **reading order** so the
Session Reader can replay the document layout exactly as it appears in the PDF.

---

## Files

### Backend

| File | Purpose |
|---|---|
| `app/services/parsing/__init__.py` | Package marker |
| `app/services/parsing/models.py` | Internal Pydantic types: `TextContentItem`, `ImageContentItem`, `TableContentItem`, `ParseResult` |
| `app/services/parsing/chunking.py` | Merges short consecutive text blocks; images and tables pass through unchanged |
| `app/services/parsing/parser.py` | Docling integration, image/table extraction, text cleaning, DB persistence, `run_parse_job` entry point |
| `app/schemas/parsing.py` | API response schemas: `ParseJobStatus`, `ChunkResponse`, `AssetSummary`, `ParsedDocumentResponse`, `SessionReaderResponse` |
| `app/routers/parsing.py` | Four new endpoints (see below) |
| `scripts/parse_local_pdf.py` | Dev script for manual verification |

### Frontend

| File | Purpose |
|---|---|
| `apps/desktop/src/pages/ReaderPage.tsx` | Session reader — renders chunks inline in reading order |

---

## Database Schema Changes

Three new tables added in `app/db/models.py`. All created by the existing `init_db()` →
`create_all` call at startup. No Alembic migration needed yet.

### `document_parse_jobs`

Tracks the lifecycle of one docling parse run per document.

| Column | Type | Notes |
|---|---|---|
| `status` | VARCHAR | `pending` → `running` → `succeeded` \| `failed` |
| `error` | TEXT | Populated on failure |
| `started_at` | TIMESTAMPTZ | Set when worker picks it up |
| `finished_at` | TIMESTAMPTZ | Set on success or failure |

### `document_chunks`

Stores every content item in reading order — text, image, and table chunks share this table.

| Column | Type | Notes |
|---|---|---|
| `chunk_index` | INT | 0-based reading order; unique per document |
| `page_start`, `page_end` | INT | Source page numbers (1-indexed, from docling) |
| `text` | TEXT | Cleaned paragraph text (text chunks) or markdown (table chunks); empty string for image chunks |
| `meta` | JSONB | Always contains `chunk_type` (`"text"` \| `"image"` \| `"table"`). Text chunks also have `label` and `bbox`. Image/table chunks also have `asset_id`, `caption`, and `bbox`. |

Unique constraint on `(document_id, chunk_index)`.

### `document_assets`

Stores the image files extracted for pictures and tables.

| Column | Type | Notes |
|---|---|---|
| `asset_type` | VARCHAR | `"image"` (picture) or `"table"` |
| `page` | INT | Source page number |
| `bbox` | JSONB | `{x0, y0, x1, y1}` in docling coordinate space |
| `file_path` | VARCHAR | Path under `services/api/parsed_cache/{document_id}/image_{n}.png` or `table_{n}.png` |
| `meta` | JSONB | `{label, caption}` — caption from `item.caption_text(doc)` |

### `sessions` (updated)

Added `elapsed_seconds INT NOT NULL DEFAULT 0` to track accumulated active time across
pause/resume cycles, enabling the frontend timer to persist correctly.

---

## New API Endpoints

### Parsing Router

| Method | Path | Description |
|---|---|---|
| `GET` | `/documents/{id}/parse-status` | Current job status + timestamps + error |
| `GET` | `/documents/{id}/parsed` | Paginated chunks + asset list (`?offset=&limit=`, default limit 30) |
| `GET` | `/documents/{id}/assets/{asset_id}` | Stream extracted image file from `parsed_cache/` |
| `POST` | `/documents/{id}/reparse` | Clear prior chunks/assets/job and restart |

The `/documents/{id}/file` endpoint also accepts `?token=<jwt>` as an alternative to
the `Authorization` header (used for direct URL embedding).

### Sessions Router (additions)

| Method | Path | Description |
|---|---|---|
| `GET` | `/sessions/{id}/reader` | Session info + parse status + first 30 chunks + all assets |
| `POST` | `/sessions/{id}/pause` | Accumulate elapsed time, set status `paused` |
| `POST` | `/sessions/{id}/resume` | Reset interval start time, set status `active` |
| `POST` | `/sessions/{id}/complete` | Finalise total duration, set status `completed` |
| `POST` | `/sessions/{id}/end` | Finalise total duration, set status `ended` |

All endpoints require authentication and enforce ownership (404 for other users' resources).

---

## Parsing Pipeline Detail

### `parser.py` — `_sync_parse(file_path, cache_dir)`

Runs inside a `ThreadPoolExecutor` thread. Steps:

1. Obtains the module-level `DocumentConverter` singleton (ML models loaded once per process).
2. Calls `converter.convert(source=str(file_path))`.
3. Iterates `doc.iterate_items()` in reading order:
   - **`TableItem`** → export markdown via `item.export_to_markdown(doc)`, save PNG via `item.get_image(doc)`, extract caption via `item.caption_text(doc)`.
   - **`PictureItem`** → save PNG via `item.get_image(doc)`, extract caption.
   - **Text items** → extract `.text`, run `_clean_text()` to strip broken glyph names (`/uniXXXX`) and Unicode replacement characters (`\ufffd`).
4. Passes the flat ordered list to `build_text_chunks()` (chunking.py).
5. Returns a `ParseResult(items, raw_doc)`.

### `_get_converter()` — Singleton

The `DocumentConverter` is expensive to initialise (loads layout, table-structure, and
picture-detection ML models). It is created once using a thread-safe double-checked lock
and reused for all subsequent parses.

Pipeline options used:

| Option | Value | Reason |
|---|---|---|
| `do_ocr` | `False` | CPU-only; enable for scanned PDFs |
| `do_table_structure` | `True` | Required for table extraction |
| `generate_picture_images` | `True` | Required for `item.get_image()` to return a PIL image |
| `generate_table_images` | `True` | Required for table PNG rendering |
| `images_scale` | `2.0` | 2× resolution for readable images |

### `chunking.py` — `build_text_chunks(raw_items)`

- **Text items**: consecutive text blocks on the same page are merged if the accumulated
  buffer is shorter than 80 characters AND neither block has a heading label
  (`section_header`, `title`, `page_header`, `page_footer`). Merged bboxes are the union
  of their individual boxes.
- **Image and table items**: passed through directly — never merged, never reordered.
- All items are re-indexed sequentially (0-based) after merging.

### Image filtering in `ReaderPage.tsx`

Only images whose caption matches `/fig\./i` (e.g. "Fig. 1.") are rendered.
Decorative images, logos, and unlabelled pictures are silently skipped.

### Text cleaning

`_clean_text()` applies two passes:
1. Strips `/uniXXXX` PostScript glyph names left over from broken font encodings.
2. Collapses runs of double-spaces introduced by the removal.

---

## Frontend

### `documentService.ts`

Methods: `getParseStatus`, `getParsed`, `reparse`, `getAssetUrl`.

### `sessionService.ts`

Interface `SessionReaderData` and method `getReader`.
`Session` interface includes `elapsed_seconds` (accumulated active seconds) and
`created_at` (original creation timestamp — `started_at` is reset on each resume).

### `HomePage.tsx`

- After upload, the document immediately appears with a **"Parsing…"** spinner badge.
- Polling starts at 1 s, backing off to 5 s max until status is terminal.
- On failure, badge shows **"Parse failed — retry"** (click to reparse).
- Clicking a document card fetches the PDF with the auth token and opens it in a **new browser tab**.
- **"Start session"** button navigates to `/sessions/:id/reader`.

### `ReaderPage.tsx` (`/sessions/:id/reader`)

- **Top bar**: session name, status badge, mode, elapsed timer, pause/resume/complete controls.
- **Main column**: text, image, and table chunks rendered inline in reading order.
  - Text chunks: plain paragraphs; heading-labelled items are larger and bold.
  - Image chunks: only figures with a `Fig.` caption are shown; caption displayed below.
  - Table chunks: markdown table with optional caption.
- Timer uses `elapsed_seconds` from the backend plus the current active interval to
  display accurate cumulative time that persists across pause/resume cycles.
- Shows a "still parsing" notice when the parse job has not yet completed.

### Auth-aware asset loading

Image and table assets are fetched with the `Authorization` header and converted to
`data:image/png;base64,...` URLs via a `useAssetDataUrl` hook. This is necessary because
browser `<img>` tags cannot attach custom headers — using data URLs bypasses that limit.

---

## Running the Application

```bash
# Start everything (DB + API + frontend)
./start.sh

# Backend only
cd services/api && .venv/bin/uvicorn app.main:app --reload --port 8000

# Install/update Python dependencies
cd services/api && .venv/bin/pip install -r requirements.txt
```

---

## Manual Parse Verification

Place a test PDF anywhere, then:

```bash
cd services/api
source .venv/bin/activate
python scripts/parse_local_pdf.py ../../pdfs/978-3-030-50506-6_10.pdf
```

Expected output (numbers will vary):

```
[parse] .../pdfs/978-3-030-50506-6_10.pdf
[cache] .../services/api/parsed_cache/local_test

──────────────────────────────────────────────────
  Text chunks  : 90
  Image chunks :  5
  Table chunks :  8
──────────────────────────────────────────────────

── First 3 text chunks ──
  [0] page=1  'Introduction ...'
  ...

── Image chunks ──
  [n] page=6  caption='Fig. 1. ...'  path=.../image_0.png

── Table chunks ──
  [n] page=3  caption='Table 1. ...'  path=.../table_0.png
```

Extracted files appear under `services/api/parsed_cache/local_test/` (git-ignored).

---

## Tests

```bash
cd services/api
.venv/bin/pytest tests/ -q
```

`tests/test_parsing.py` covers:

- Upload creates a pending parse job
- `GET /parse-status` returns expected fields
- Ownership enforced (404 for another user)
- `GET /parsed` returns zero chunks before parse completes
- `GET /sessions/{id}/reader` returns session + parse data
- `POST /reparse` clears prior data and creates a fresh job
- Reparse ownership enforced

Docling is **mocked** in all tests — no actual PDF processing in CI.

---

## Gitignore Additions

```
services/api/parsed_cache/   # extracted image/table PNGs
pdfs/                        # local test PDFs
```

---

## Known Limitations / Deferred

- OCR is disabled (`do_ocr=False`). Enable in `_get_converter()` for scanned PDFs (requires a server restart to recreate the singleton).
- Changing `do_table_structure` also requires a server restart for the same reason.
- LLM-driven chunking, drift detection, and interventions are **not** implemented in this phase.
- Alembic migrations not yet configured — schema is managed by `create_all`.
- The full docling document dict (`raw_doc` in `ParseResult`) is not persisted to the DB; it is available in memory after a parse and could be stored in a JSONB column if needed later.
