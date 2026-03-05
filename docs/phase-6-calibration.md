# Phase 6 — Calibration (Baseline Collection)

> **Scope:** One-time reading calibration per user.  
> No drift model yet. Raw baseline stored in `user_baselines` for future use.

---

## Overview

Before focus tracking can be personalised, we need each user's **natural reading baseline**.  A one-time calibration session measures:

- Estimated words-per-minute (WPM)
- Scroll velocity mean & standard deviation
- Scroll jitter / direction-change ratio
- Idle-time ratio
- Paragraph dwell time

These values are stored in a `user_baselines` DB row and will be used by the drift model in Phase 7.

---

## Calibration PDF

File: `callibration/Callibration Test.pdf` (repo root, note the spelling).

The backend resolves the path from `services/api/app/routers/calibration.py` using `__file__`, so it works regardless of the server's working directory.

---

## New Database Changes

### `is_calibration` column on `documents`

```sql
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS is_calibration BOOLEAN NOT NULL DEFAULT FALSE;
```

Applied idempotently in both `init_db()` (production) and the test `init_test_db` fixture.

Calibration documents are hidden from `GET /documents` (the normal library).

### `user_baselines` table (new)

| Column | Type | Notes |
|---|---|---|
| `user_id` | PK, FK → users.id | One row per user |
| `baseline_json` | JSONB | Full baseline object (see shape below) |
| `completed_at` | TIMESTAMPTZ | When the calibration was finished |
| `created_at` | TIMESTAMPTZ | First calibration |
| `updated_at` | TIMESTAMPTZ | Last re-calibration |

**`baseline_json` shape (v1):**

```json
{
  "wpm_mean": 185.0,
  "wpm_std": 0.0,
  "scroll_velocity_mean": 55.3,
  "scroll_velocity_std": 12.1,
  "scroll_jitter_mean": 0.12,
  "idle_ratio_mean": 0.08,
  "regress_rate_mean": 0.12,
  "paragraph_dwell_mean": 6.4,
  "calibration_duration_seconds": 147
}
```

---

## New Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/calibration/status` | ✓ | Returns `has_baseline`, `calib_available`, `parse_status` |
| `POST` | `/calibration/start` | ✓ | Creates (or reuses) calibration doc + starts session |
| `POST` | `/calibration/complete` | ✓ | Ends session, computes + stores baseline |
| `GET` | `/calibration/baseline` | ✓ | Returns stored baseline for the current user |
| `GET` | `/sessions/{id}/export.csv` | ✓ | Download telemetry_batch events as CSV |

---

## Calibration Flow

```
User logs in
  └── HomePage mounts
        └── GET /calibration/status
              ├── has_baseline = true  → show dashboard normally
              └── has_baseline = false AND calib_available = true
                    └── redirect to /calibration

CalibrationPage
  └── "Start Calibration" button
        └── POST /calibration/start
              ├── creates Document (is_calibration=True) if needed
              ├── kicks off docling parse via BackgroundTasks
              ├── if parse not succeeded → 503 "still parsing"
              └── creates Session(mode="calibration") → returns session_id
                    └── navigate to /sessions/{id}/reader

ReaderPage (mode="calibration")
  ├── Calibration banner shown
  ├── "Finish Calibration" button enabled when:
  │     elapsed >= 120 s  OR  (elapsed >= 90 s AND viewport_progress >= 0.95)
  └── On "Finish Calibration":
        └── POST /calibration/complete
              ├── ends session
              ├── reads all telemetry_batch events for session
              ├── computes baseline from telemetry
              ├── upserts user_baselines
              └── returns baseline + shows summary overlay
                    └── "Go to Dashboard" → navigate("/")
```

---

## Baseline Computation (`services/calibration/baseline.py`)

All functions are pure and exported for unit testing.

### `scroll_velocities(batches)` → `list[float]`
`scroll_delta_abs_sum / 2.0` per batch window.

### `scroll_jitter_values(batches)` → `list[float]`
`scroll_direction_changes / scroll_event_count` per batch (only when events > 0).

### `idle_ratios(batches)` → `list[float]`
`min(idle_seconds / 2.0, 1.0)` per batch.

### `paragraph_dwells(batches)` → `dict[str, int]`
Counts how many 2-second windows each `current_paragraph_id` was the most-visible element.

### `estimate_wpm(batches, chunk_word_counts, duration_seconds)` → `float`
- Collects all distinct paragraphs seen across batches.
- Looks up word count for each from the document chunks (using `chunk-{id}` ID format).
- Returns `total_words / (duration_seconds / 60)`.

### `compute_baseline(batches, chunk_word_counts, duration_seconds)` → `dict`
Calls all helpers and returns the full baseline dict.

---

## Export CSV (`GET /sessions/{id}/export.csv`)

Returns all `telemetry_batch` activity events for a session as CSV with columns:

```
created_at, scroll_delta_sum, scroll_delta_abs_sum, scroll_event_count,
scroll_direction_changes, scroll_pause_seconds, idle_seconds, mouse_path_px,
mouse_net_px, window_focus_state, current_paragraph_id, current_chunk_index,
viewport_progress_ratio
```

A copy is also written to `services/api/exports/user_{id}/session_{id}.csv` (gitignored).

The "Export CSV" button is always visible during calibration sessions and in dev mode.

---

## New Files

### Backend

| File | Purpose |
|---|---|
| `app/db/models.py` | `UserBaseline` model; `is_calibration` on `Document` |
| `app/core/config.py` | `exports_dir` setting |
| `app/db/engine.py` | `ALTER TABLE` migration + `exports_dir.mkdir()` |
| `app/schemas/calibration.py` | Pydantic schemas for all calibration endpoints |
| `app/schemas/sessions.py` | Added `"calibration"` to mode `Literal` |
| `app/services/calibration/__init__.py` | Package marker |
| `app/services/calibration/baseline.py` | Pure baseline computation helpers |
| `app/routers/calibration.py` | All calibration API endpoints |
| `app/routers/sessions.py` | `GET /sessions/{id}/export.csv` endpoint |
| `app/routers/documents.py` | Filter `is_calibration=True` from document list |
| `app/main.py` | Register `calibration.router` |
| `tests/test_calibration.py` | 18 tests — all pass |
| `exports/` | Gitignored folder for CSV files |

### Frontend

| File | Purpose |
|---|---|
| `services/calibrationService.ts` | API calls for calibration endpoints |
| `pages/CalibrationPage.tsx` | Explanation + start button page |
| `pages/ReaderPage.tsx` | Calibration banner, `CalibrationControls`, `BaselineSummary`, `ExportCsvButton` |
| `App.tsx` | `/calibration` route added |
| `pages/HomePage.tsx` | Redirect to `/calibration` if `has_baseline=false` |
| `services/sessionService.ts` | Added `"calibration"` to `SessionMode` |
| `test/calibration.test.ts` | 11 unit tests — all pass |

---

## Test Coverage

### Backend (pytest) — 18 tests

| Test | What it verifies |
|---|---|
| Unit: `test_empty_batches_returns_zeros` | Empty batch → all-zero baseline |
| Unit: `test_scroll_velocity_computed_correctly` | px/s calculation |
| Unit: `test_jitter_values_empty_when_no_events` | Edge case: no scroll events |
| Unit: `test_jitter_ratio_computed` | Direction-change ratio |
| Unit: `test_wpm_estimate_from_paragraphs` | 180 words / 2 min = 90 WPM |
| Unit: `test_paragraph_dwell_counted_correctly` | Dwell count accumulation |
| Unit: `test_full_compute_with_batches` | Full baseline from 5 batches |
| `test_status_no_baseline` | Returns `has_baseline=false` for new user |
| `test_status_requires_auth` | 401 without token |
| `test_start_creates_calibration_session` | Session with `mode="calibration"` created |
| `test_start_requires_auth` | 401 without token |
| `test_complete_creates_user_baseline` | `user_baselines` row inserted |
| `test_complete_status_shows_has_baseline` | `has_baseline=true` after completion |
| `test_complete_rejects_wrong_mode` | 404 for non-calibration session |
| `test_complete_enforces_ownership` | User B gets 404 for User A session |
| `test_export_returns_csv` | CSV with correct header returned |
| `test_export_requires_auth` | 401 without token |
| `test_export_enforces_ownership` | User B gets 404 for User A session |

### Frontend (vitest) — 11 tests

| Test | What it verifies |
|---|---|
| `shouldRedirectToCalibration` (4) | Redirect condition matrix |
| `canFinishCalibration` (7) | All finish-button threshold cases |

---

## Manual Testing Steps

1. Start backend: `uvicorn app.main:app --reload` (from `services/api/`)
2. Start frontend: `npm run dev` (from `apps/desktop/`)
3. Register a new user → you are redirected to `/calibration`
4. The calibration page will show "Preparing calibration document…" while docling parses it (30–60 s first run)
5. Once parsed, click **Start Calibration** → you land in the reader
6. The purple banner "⊙ Read at your natural pace…" is visible
7. Read normally — the timer counts up; **Finish Calibration** is greyed out until 120 s
8. After 2 minutes, click **Finish Calibration** → baseline summary overlay appears
9. Baseline stats (WPM, scroll velocity, idle %, duration) are shown
10. Click **Export CSV** to download telemetry data
11. Click **Go to Dashboard** — you can now start normal reading sessions
12. On subsequent logins, `/calibration/status` returns `has_baseline=true` → dashboard loads directly

To reset your baseline (re-calibrate): `DELETE FROM user_baselines WHERE user_id = {id};`

---

## Known Limitations / Deferred

- **WPM std is always 0** — v1 computes a single-pass estimate; per-batch WPM would require knowing which words were *newly* visible, which needs scroll direction tracking.
- **`calib_available = false` blocks** — if the calibration PDF is not on the server, `calib_available=false` and the user cannot complete calibration. The redirect is suppressed in this case.
- **Calibration doc is per-user** — each user gets their own `Document` row pointing to the same physical PDF. This avoids user_id nullable complications but wastes slightly redundant parse runs. Phase 7 may consolidate to a shared doc.
