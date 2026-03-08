# Phase 6 — Calibration (Baseline Collection)

> **Scope:** One-time reading calibration per user.  
> Stores a personalised reading profile in `user_baselines`.  
> No drift scoring yet — baselines are consumed by Phase 7.

---

## Purpose

Attention drift cannot be detected without knowing what *normal* looks like for a specific person.
A 300 WPM reader and a 150 WPM reader will produce completely different scroll velocities, idle ratios, and dwell times while reading attentively. Without calibration, any threshold we set would be arbitrary and would misclassify one or both users constantly.

Calibration solves this by measuring each user's **natural, attentive reading behaviour** on a short, controlled text and storing the resulting profile. Every metric captured during calibration becomes a **personal baseline**. In Phase 7, deviations from that baseline — not absolute values — drive the drift score.

---

## What Calibration Measures (and Why Each Metric Matters)

### 1. `wpm_mean` — Words Per Minute

**How computed:** `total_words_in_text / session_duration_minutes`

The calibration text is short enough to read in one pass. The user is told to read at their natural pace, so the result is their comfortable reading speed.

**How used in Phase 7:**  
WPM sets the **expected scroll rate**. For a chunk of N words, the system expects the user to spend approximately `N / wpm_mean` minutes on it. If actual dwell time is much shorter (fast scroll past) or much longer (stuck, confused, distracted), drift is flagged. This also sets the **per-paragraph expected dwell time**, which becomes the anchor for all paragraph-level anomaly detection.

---

### 2. `scroll_velocity_mean` + `scroll_velocity_std` — Scroll Speed

**How computed:** `scroll_delta_abs_sum / 2.0` per 2-second batch window, then `mean` and `stdev` across all windows.

**How used in Phase 7:**  
These two values define the user's **normal scroll speed band**:
- `scroll_velocity_mean` is the expected cruising speed.
- `scroll_velocity_std` is the natural variation — some windows are faster, some slower.

A scroll velocity outside `mean ± k * std` (where k is tuned, e.g. 2.0) indicates either rapid skimming (potential disengagement) or very slow progress (confusion, distraction, or high cognitive load). The wider the user's natural std, the more tolerant the model should be before flagging.

---

### 3. `scroll_jitter_mean` — Direction-Change Ratio

**How computed:** `scroll_direction_changes / scroll_event_count` per batch, then `mean`.

Jitter measures how much the user re-reads — scrolling up slightly then continuing down, typical of attentive reading and normal checking behaviour.

**How used in Phase 7:**  
A user who naturally re-reads often (high baseline jitter) should not be penalised for it during a session. The calibration value sets the **expected re-read rate**. A sudden *drop* in jitter (pure linear scroll with no re-reading) can indicate mindless scrolling. A sudden *spike* beyond the baseline can indicate confusion or high cognitive load.

---

### 4. `idle_ratio_mean` — Fraction of Time Spent Idle

**How computed:** `idle_seconds / 2.0` per batch, clamped to `[0, 1]`, then `mean`.

Idle is defined as no scroll, mouse, or keyboard activity within a 2-second window.

**How used in Phase 7:**  
Some people naturally pause to reflect while reading. The calibration text has no images or tables, so `idle_ratio_mean` captures **pure reflective pause behaviour** on plain text only. This is the baseline idle fraction for plain text.

For sessions with images and tables, additional idle time is *expected* and *budget-adjusted* (see **Content-Type Idle Budgets** below).

---

### 5. `paragraph_dwell_mean` — Average Time Per Paragraph

**How computed:** Count of 2-second windows the `current_paragraph_id` was the most-visible element, multiplied by 2 to get seconds, then `mean` across all paragraphs.

**How used in Phase 7:**  
The baseline dwell time per paragraph, combined with WPM, gives an expected reading time per unit of text. When a user dwells far longer than `paragraph_dwell_mean` on a specific paragraph (especially if idle ratio is also high), it signals a comprehension difficulty or distraction event — not just slow reading.

---

### 6. `regress_rate_mean` — Backward Scroll Rate

**How computed (v1):** Same as `scroll_jitter_mean` (same proxy). Future versions will distinguish forward/backward scroll by direction.

**How used in Phase 7:**  
Will weight the drift model toward "re-reading" interpretations versus "random fidgeting" based on the user's calibrated tendency to backtrack.

---

## Content-Type Idle Budgets (Phase 7 Design)

The calibration text contains no images or tables, so `idle_ratio_mean` reflects **pure text reading**. Real documents contain:

- **Figures / diagrams** — the reader stops scrolling to study the image.
- **Tables** — the reader cross-references cells, often much slower than plain text.

To avoid false drift signals when a user is simply paying attention to a figure or table, Phase 7 will apply a **content-type idle budget**: when the active chunk is classified as an image or table chunk (`meta.chunk_type`), the expected idle time is *extended* before the drift signal is triggered.

### Default Budget Values (Rule-of-Thumb Starting Points)

| Content type | Default idle budget | Rationale |
|---|---|---|
| Plain text paragraph | `idle_ratio_mean * 2s_window` | Directly from calibration |
| Figure / diagram | **60 seconds** | ~1 min to interpret a chart |
| Table | **90 seconds** | ~1.5 min to read a table |

These are **starting defaults** that are immediately *personalised* using calibration:

```
figure_budget_seconds = 60 × (1 + idle_ratio_mean)
table_budget_seconds  = 90 × (1 + idle_ratio_mean)
```

A user with `idle_ratio_mean = 0.30` (reflective pauser) gets:
- Figure budget: `60 × 1.30 = 78 s`
- Table budget: `90 × 1.30 = 117 s`

A user with `idle_ratio_mean = 0.05` (active reader) gets:
- Figure budget: `60 × 1.05 = 63 s`
- Table budget: `90 × 1.05 = 94.5 s`

This means the same figure viewing behaviour is not penalised for one user and excused for another — the model adapts to the individual.

### Implementation hook (ready for Phase 7)

When computing the per-batch expected idle fraction in a reading session, the drift service should:

1. Look up the `meta.chunk_type` of the current active chunk.
2. If `chunk_type == "image"`: replace per-batch idle threshold with `figure_budget_seconds / total_session_seconds`.
3. If `chunk_type == "table"`: replace per-batch idle threshold with `table_budget_seconds / total_session_seconds`.
4. Otherwise: use `idle_ratio_mean` from the user's `user_baselines.baseline_json`.

---

## Extrapolation to Full Reading Sessions (Phase 7 Design)

The calibration text (~450 words, no media) is intentionally simple. Every baseline value recorded must scale correctly to longer, richer documents.

### WPM → Expected Dwell Time Per Chunk

```
expected_dwell_seconds(chunk) = chunk.meta.word_count / (wpm_mean / 60)
```

If a chunk has 80 words and the user reads at 180 WPM:
```
expected_dwell = 80 / (180 / 60) = 80 / 3 = 26.7 seconds
```

The drift signal triggers when actual dwell deviates from this by a configurable factor (e.g. `< 0.4×` or `> 3×` expected).

### Scroll Velocity → Normalised Scroll Signal

Scroll velocity from calibration is measured on text rendered at the same font size and line height as the reader page. Because the typography is identical between calibration and reading sessions, scroll velocity is directly comparable. No normalisation needed for text size.

If the user changes font size in the future, velocity should be re-normalised by the ratio of new line-height to calibration line-height.

### Idle Ratio → Contextual Idle Budget

As described above — plain text idle uses the raw baseline; images and tables use the expanded budget.

### Paragraph Dwell → Anomaly Detection Anchor

For each chunk, Phase 7 will compare:
```
drift_signal_paragraph = actual_dwell / expected_dwell_seconds(chunk)
```

Values:
- `0.3 – 0.5` → fast pass, possible skip (mild drift)
- `0.5 – 2.0` → normal reading (within calibration range)
- `2.0 – 4.0` → slow reading or distraction (watch for idle spike)
- `> 4.0`    → likely disengaged / walked away

The thresholds `0.5` and `2.0` are starting defaults; they will be tuned by the user's `paragraph_dwell_mean` (e.g. a slow reader's "2.0×" threshold will be set at a higher absolute value).

---

## Calibration Text File

**Path:** `callibration/callibration.txt` (repo root)

Plain UTF-8 text, one sentence per paragraph (separated by `\n\n`). The backend reads it at request time — editing the file takes effect immediately on the next calibration start, with no restart required.

**Word count:** ~450 words (count is computed live from the file via `len(text.split())`).

The text was chosen to be:
- Factual but engaging (not too easy, not too hard)
- Long enough for a meaningful scroll velocity measurement
- Free of images/tables (so idle baseline = pure reading only)

---

## Current Calibration Flow (Implemented)

```
User logs in
  └── HomePage mounts
        └── GET /calibration/status
              ├── has_baseline = true  → show dashboard normally
              └── has_baseline = false AND calib_available = true
                    └── redirect to /calibration

CalibrationPage (/calibration)
  └── "Start Calibration" button (enabled as soon as calib_available = true)
        └── POST /calibration/start (instant — no parsing needed)
              ├── reads callibration.txt
              ├── creates Document (is_calibration=True) + chunks in DB
              ├── creates Session(mode="calibration", status="active")
              └── returns session_id
                    └── navigate to /sessions/{id}/calibration-reader

CalibrationReaderPage (/sessions/:id/calibration-reader)
  ├── Fetches session info from /sessions list (for timer)
  ├── Fetches paragraphs from GET /calibration/session/{id} (live from txt file)
  ├── Displays text with same typography as ReaderPage (19px, 1.75 line-height, 72ch)
  ├── useTelemetry hook runs — POSTs batches to /activity/batch every 2s
  ├── Timer counts up; "Done" button enabled after 10s grace period
  └── On "Done":
        └── POST /calibration/complete
              ├── ends session, computes duration
              ├── reads all telemetry_batch events for session
              ├── sums total word count from DB chunks (= callibration.txt word count)
              ├── WPM = total_words / duration_minutes
              ├── computes all other metrics from telemetry batches
              ├── upserts user_baselines row
              └── returns baseline → shows BaselineSummary overlay
                    └── "Go to Dashboard" → navigate("/")
```

---

## Database Schema

### `documents.is_calibration` (added column)

```sql
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS is_calibration BOOLEAN NOT NULL DEFAULT FALSE;
```

Calibration documents are hidden from `GET /documents`.

### `user_baselines` table

| Column | Type | Notes |
|---|---|---|
| `user_id` | PK, FK → users.id | One row per user; upserted on re-calibration |
| `baseline_json` | JSONB | Full baseline object |
| `completed_at` | TIMESTAMPTZ | Last time calibration was finished |
| `created_at` | TIMESTAMPTZ | First calibration |
| `updated_at` | TIMESTAMPTZ | Auto-updated on upsert |

**`baseline_json` shape (current):**

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

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/calibration/status` | ✓ | `has_baseline`, `calib_available`, `parse_status` |
| `POST` | `/calibration/start` | ✓ | Creates doc+chunks from txt, starts session |
| `GET` | `/calibration/session/{id}` | ✓ | Returns session info + paragraphs from txt file |
| `POST` | `/calibration/complete` | ✓ | Ends session, computes+stores baseline |
| `GET` | `/calibration/baseline` | ✓ | Returns stored baseline |
| `GET` | `/sessions/{id}/export.csv` | ✓ | Download telemetry_batch events as CSV |

---

## Baseline Computation (`services/calibration/baseline.py`)

All functions are pure and unit-tested.

| Function | Input | Output |
|---|---|---|
| `scroll_velocities(batches)` | batch list | `[px/s per window]` |
| `scroll_jitter_values(batches)` | batch list | `[direction_changes/events per window]` |
| `idle_ratios(batches)` | batch list | `[idle_seconds/2.0 per window, clamped 0–1]` |
| `paragraph_dwells(batches)` | batch list | `{paragraph_id: window_count}` |
| `estimate_wpm(batches, chunk_word_counts, duration_s, total_words_override)` | — | WPM float |
| `compute_baseline(batches, chunk_word_counts, duration_s, total_words)` | — | baseline dict |

**WPM computation (current):**  
`total_words_override` (= sum of all chunk word counts from the calibration doc) is passed directly, so `WPM = total_words / duration_minutes`. This avoids the `calib-N` paragraph ID parsing issue and gives an accurate full-text reading speed.

---

## CSV Export

`GET /sessions/{id}/export.csv` returns all `telemetry_batch` events as CSV:

```
created_at, scroll_delta_sum, scroll_delta_abs_sum, scroll_event_count,
scroll_direction_changes, scroll_pause_seconds, idle_seconds, mouse_path_px,
mouse_net_px, window_focus_state, current_paragraph_id, current_chunk_index,
viewport_progress_ratio
```

A copy is written to `services/api/exports/user_{id}/session_{id}.csv` (gitignored).

---

## New and Modified Files

### Backend

| File | What changed |
|---|---|
| `app/db/models.py` | `UserBaseline` model; `is_calibration` on `Document` |
| `app/core/config.py` | `exports_dir` |
| `app/db/engine.py` | `ALTER TABLE` migration for `is_calibration`; `exports_dir.mkdir()` |
| `app/schemas/calibration.py` | All calibration Pydantic schemas incl. `CalibrationReaderResponse` |
| `app/schemas/sessions.py` | `"calibration"` added to mode `Literal` |
| `app/services/calibration/baseline.py` | Baseline computation; `total_words_override` param added |
| `app/routers/calibration.py` | All calibration endpoints; txt-based approach (no Docling) |
| `app/routers/sessions.py` | `export.csv` endpoint; parse-status fallback when no parse job |
| `app/routers/documents.py` | Filter `is_calibration=True` from document list |
| `app/main.py` | Register `calibration.router` |
| `tests/test_calibration.py` | 21 tests — all pass |

### Frontend

| File | What changed |
|---|---|
| `services/calibrationService.ts` | API client for calibration endpoints |
| `services/sessionService.ts` | `"calibration"` added to `SessionMode` |
| `pages/CalibrationPage.tsx` | Explanation + start button; instant (no parse wait) |
| `pages/CalibrationReaderPage.tsx` | Dedicated calibration reader; hardcoded text fallback + live fetch |
| `App.tsx` | `/sessions/:id/calibration-reader` route added |
| `pages/HomePage.tsx` | Redirect to `/calibration` when `has_baseline=false` |
| `test/calibration.test.ts` | 9 tests — all pass |

---

## Test Coverage

### Backend — 21 tests (pytest)

| Test | What it verifies |
|---|---|
| `test_empty_batches_returns_zeros` | Empty batch → all-zero baseline |
| `test_scroll_velocity_computed_correctly` | px/s calculation |
| `test_jitter_values_empty_when_no_events` | Edge case |
| `test_jitter_ratio_computed` | Direction-change ratio |
| `test_wpm_estimate_from_paragraphs` | chunk-N paragraph IDs |
| `test_wpm_uses_total_words_override` | 300 words / 2 min = 150 WPM |
| `test_wpm_handles_calib_paragraph_ids` | calib-N paragraph IDs |
| `test_paragraph_dwell_counted_correctly` | Dwell accumulation |
| `test_full_compute_with_batches` | Full baseline from 5 batches |
| `test_status_no_baseline` | `has_baseline=false` for new user |
| `test_status_requires_auth` | 401 without token |
| `test_start_creates_calibration_session` | Session `mode="calibration"` |
| `test_start_creates_chunks_from_txt` | Chunks created from txt file |
| `test_start_requires_auth` | 401 without token |
| `test_complete_creates_user_baseline` | `user_baselines` row inserted |
| `test_complete_status_shows_has_baseline` | `has_baseline=true` after completion |
| `test_complete_rejects_wrong_mode` | 404 for non-calibration session |
| `test_complete_enforces_ownership` | Cross-user 404 |
| `test_export_returns_csv` | CSV with correct header |
| `test_export_requires_auth` | 401 without token |
| `test_export_enforces_ownership` | Cross-user 404 |

### Frontend — 9 tests (vitest)

| Test | What it verifies |
|---|---|
| `shouldRedirectToCalibration` (4) | Redirect condition matrix |
| `canFinishCalibration` (5) | Finish-button grace period |

---

## Manual Testing Steps

1. Start backend: `uvicorn app.main:app --reload` (from `services/api/`)
2. Start frontend: `npm run dev` (from `apps/desktop/`)
3. Register a new user → redirected to `/calibration`
4. Click **Start Calibration** — navigates immediately to the calibration reader
5. Read the text naturally; timer counts up
6. **Done** button is enabled after 10 s (grace period only)
7. Click **Done** when finished → baseline summary overlay appears
8. Stats shown: WPM, scroll velocity, idle %, duration
9. Click **Export CSV** (dev mode) to download telemetry
10. Click **Go to Dashboard** → normal sessions now available
11. On subsequent logins: `has_baseline=true` → dashboard loads directly

To reset and re-calibrate:
```sql
DELETE FROM user_baselines WHERE user_id = {id};
```

---

## Known Limitations / Deferred to Phase 7

| Issue | Status |
|---|---|
| `wpm_std` is always 0.0 | v1 single-pass; per-window WPM needs direction-aware scroll tracking |
| `regress_rate_mean` = same as jitter | v1 proxy; Phase 7 will use signed scroll delta to distinguish backtrack from fidget |
| Content-type idle budgets (figure/table) | Designed above, implemented in Phase 7 drift scorer |
| Extrapolation to multi-section documents | Formula documented above, wired in Phase 7 |
| Calibration re-run (intentional reset) | Works via SQL delete; no UI reset button yet |
| Calibration text word count is hardcoded as 330 in some comments | Live count from DB chunks is used at runtime; comment is cosmetic only |
