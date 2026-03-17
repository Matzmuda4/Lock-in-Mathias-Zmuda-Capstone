# Phase 8 — Training Data Export Pipeline (`classify` branch)

## Overview

Phase 8 introduces a privacy-safe, reproducible export pipeline that collects all model training data locally from any reading session. The goal is to build a labelled dataset for the future LLM attentional-state classifier that will map user behavioural signals to state labels (focused / drifting / hyperfocused / cognitive_overload).

No labels are applied automatically in this phase. The drift score is deliberately **not** used as a training label because it has not been independently validated. All state packets are exported raw for manual or semi-automated labelling in a later step.

---

## What was added

### 1. `session_state_packets` — new TimescaleDB hypertable

**File:** `services/api/app/db/models.py`

A new append-only hypertable stores a full snapshot of the drift pipeline output approximately every 10 seconds during any active session. Each row captures:

- `session_id` — FK to sessions
- `created_at` — partition column (TimescaleDB)
- `packet_json` (JSONB) — serialised snapshot containing:
  - `drift` — `drift_level`, `drift_ema`, `beta_ema`, `beta_effective`, `disruption_score`, `engagement_score`, `confidence`, `pace_ratio`, `pace_available`
  - `features` — the full `WindowFeatures` dataclass (scroll velocity, idle ratio, jitter, regress rate, stagnation, pace, mouse efficiency, focus loss rate, progress velocity, data-quality fault rates)
  - `z_scores` — all normalised deviations from the calibration baseline (`z_idle`, `z_focus_loss`, `z_jitter`, `z_regress`, `z_pause`, `z_stagnation`, `z_mouse`, `z_pace`, `z_skim`, `z_burstiness`)
  - `debug` — `beta_components` dict with per-term contributions

The composite primary key `(session_id, created_at)` is required for TimescaleDB compatibility (same pattern as `session_drift_history`).

**Why a separate table from `session_drift_history`?**  
`session_drift_history` stores only the scalar drift trajectory (6 columns) for lightweight querying. `session_state_packets` stores the full high-dimensional feature vector needed for training. Keeping them separate avoids bloating the history table.

---

### 2. Drift store — automatic packet persistence

**File:** `services/api/app/services/drift/store.py`

`upsert_drift_state()` now inserts a `SessionStatePacket` row every `_HISTORY_EVERY_N` calls (5 calls = ~10 seconds) alongside the existing `SessionDriftHistory` row. The packet is built by `_build_packet_json(result: DriftResult)` using `dataclasses.asdict()` to serialise `WindowFeatures` and `ZScores`.

This means every active session automatically accumulates state packets in the DB with no additional API calls required.

---

### 3. `ExportService` — file writing layer

**File:** `services/api/app/services/exports/service.py`

A pure service module (no HTTP, no auth) that accepts `(user_id, session_id, db)` and writes a complete bundle to `training/exports/user_{uid}/session_{sid}/`. Key design decisions:

**Deterministic CSV flattener** (`flatten_telemetry_batch`):  
Activity payload fields vary between sessions and over time. The flattener uses a fixed ordered column list (`TELEMETRY_COLUMNS`, 24 columns) so CSV files always have the same schema regardless of which optional fields were populated. Missing fields default to `""`. The raw JSONB payload is always preserved in a final `payload_json` column so no data is ever silently dropped.

**Telemetry columns (in order):**  
`created_at`, `session_id`, `scroll_delta_sum`, `scroll_delta_abs_sum`, `scroll_delta_pos_sum`, `scroll_delta_neg_sum`, `scroll_event_count`, `scroll_direction_changes`, `scroll_pause_seconds`, `idle_seconds`, `idle_since_interaction_seconds`, `mouse_path_px`, `mouse_net_px`, `window_focus_state`, `current_paragraph_id`, `current_chunk_index`, `viewport_progress_ratio`, `viewport_height_px`, `viewport_width_px`, `reader_container_height_px`, `telemetry_fault`, `scroll_capture_fault`, `paragraph_missing_fault`, `ui_context`, `interaction_zone`, `payload_json`

**Output files per session:**

| File | Contents |
|------|----------|
| `session_meta.json` | Session + document metadata, export timestamp, app/schema version |
| `baseline.json` | Full `user_baselines.baseline_json` v2 payload + `baseline_valid` flag |
| `state_packets.jsonl` | One JSON object per line per ~10 s, ordered by `created_at` |
| `telemetry_batches.csv` | Every 2-second telemetry batch with deterministic columns |
| `events.csv` | Non-batch events: blur, focus, progress_marker, interventions |
| `document_chunks.csv` | Chunk metadata (chunk_type, word_count, page range, text preview) — written only if the document has chunks |

**`baseline_valid` computation:**  
`baseline_valid = True` when `wpm_effective > 0` AND `calibration_duration_seconds >= 60`. This flag is stored in `baseline.json` so downstream labelling scripts can filter out sessions with broken baselines.

**ZIP creation** (`zip_folder`):  
Creates a `session_{id}.zip` in the parent folder when the `?download=1` query parameter is used, enabling one-click download of the full bundle.

---

### 4. API endpoints

**File:** `services/api/app/routers/exports.py`  
**Registered in:** `services/api/app/main.py`

All endpoints enforce JWT authentication and strict session ownership.

#### `GET /sessions/{session_id}/export/bundle`

Writes the bundle for a single session.

- Returns a JSON summary: `{ session_id, user_id, folder, files, state_packet_count, telemetry_batch_count, event_count }`
- Add `?download=1` to receive the bundle as `application/zip` instead.
- 404 if the session does not belong to the authenticated user.

#### `POST /exports/sessions`

Batch-export multiple sessions.

```json
// Request body
{ "session_ids": [42, 43, 44] }

// Response
{
  "results": [ { "session_id": 42, ... }, ... ],
  "errors": [ { "session_id": 99, "error": "Session not found" } ]
}
```

Partial success is intentional: failed sessions are listed in `errors` rather than aborting the entire batch.

#### `GET /exports/users/me/baseline`

Returns the current user's calibration baseline as JSON for quick inspection without writing any files.

---

### 5. Config

**File:** `services/api/app/core/config.py`

Added `training_exports_dir: Path` setting. The default is computed at import time using `__file__` resolution (not the process working directory) so it always resolves to `{repo_root}/training/exports/` regardless of where `uvicorn` is launched from.

```python
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
training_exports_dir: Path = _REPO_ROOT / "training" / "exports"
```

Override via `TRAINING_EXPORTS_DIR` environment variable. The directory is created automatically at startup alongside the other storage directories.

---

### 6. Database engine + conftest

**Files:** `services/api/app/db/engine.py`, `services/api/tests/conftest.py`

- `engine.py` converts `session_state_packets` to a TimescaleDB hypertable at startup (idempotent — uses `if_not_exists => TRUE`).
- `conftest.py` initialises the same hypertable in the test database and cleans `session_state_packets` and `session_drift_history` rows after every test.

---

### 7. Frontend — dev-only Export Bundle button

**File:** `apps/desktop/src/pages/ReaderPage.tsx`

A **📦 Export Bundle** button is rendered at the bottom of the reader page when `import.meta.env.DEV` is true. It calls `GET /sessions/{id}/export/bundle` and displays the local folder path and written file list inline after a successful export. No UI is shown in production builds.

---

### 8. Gitignore

**File:** `.gitignore`

Added entries to prevent training artifacts from being committed:
- `training/exports/`
- `training/data/`
- `training/adapters/`
- `*.zip`

---

### 9. Training README

**File:** `training/README.md`

Documents how to:
- Export a single session via HTTP or the UI button
- Batch-export multiple sessions
- Understand each output file's schema
- Use the exported data for labelling and LLM fine-tuning

---

### 10. State taxonomy

**State labels** used throughout docs and any future enum placeholders:

| Code | Meaning |
|------|---------|
| `focused` | On-task, pace and signals near baseline |
| `drifting` | Measurable behavioural deviation from baseline (early warning) |
| `hyperfocused` | Extremely fast pace, low regress, may indicate skimming/rushing |
| `cognitive_overload` | Slow pace, high regress/jitter, high stagnation — reader appears stuck or confused |

> **Note:** "Fatigued" was retired as a label. Cognitive overload is the preferred term because it maps more precisely to identifiable behavioural signals (high regress, stagnation, slow pace) rather than an unmeasurable physiological state.

---

### 11. `ui_context` + `interaction_zone` fields in telemetry (Phase 8 update)

**Backend:** `services/api/app/schemas/activity.py`, `services/api/app/services/exports/service.py`  
**Frontend:** `apps/desktop/src/services/activityService.ts`, `apps/desktop/src/hooks/useTelemetry.ts`

Two new optional fields are emitted in every telemetry batch and exported as deterministic CSV columns:

#### `ui_context` — reader UI state at flush time

| Value | Meaning |
|-------|---------|
| `READ_MAIN` | Default: user is reading, no panel present (baseline / calibration sessions always emit this) |
| `PANEL_OPEN` | Adaptive session: the assistant panel is visible and mounted |
| `PANEL_INTERACTING` | Adaptive session: user clicked, moved the mouse, or typed inside the panel during this 2 s window |
| `USER_PAUSED` | Session is paused (overrides panel state) |

Priority: `USER_PAUSED` > `PANEL_INTERACTING` > `PANEL_OPEN` > `READ_MAIN`.

#### `interaction_zone` — DOM zone of the last interaction in this window

| Value | Meaning |
|-------|---------|
| `reader` | Default: event occurred inside the reader scroll container |
| `panel` | Event occurred inside the adaptive assistant panel |
| `other` | Event occurred outside both known zones |

**Why these fields matter for training:**  
When the LLM classifier processes state packets it must not treat a "click in the panel" the same as "reader idle". These fields allow labelling scripts and the model to exclude or discount panel-interaction windows from reading-behaviour analysis.

**Baseline sessions are unaffected:** no panel renders, so every batch gets `ui_context="READ_MAIN"` and `interaction_zone="reader"` by default.

---

### 12. Self-contained state packets (Phase 8 update)

**File:** `services/api/app/services/drift/store.py`

Each `session_state_packets` row now embeds a `baseline_snapshot` inside `packet_json`:

```json
{
  "drift": { ... },
  "features": { ... },
  "z_scores": { ... },
  "debug": { ... },
  "baseline_snapshot": {
    "baseline_json": { ... },      // full v2 baseline or null
    "baseline_updated_at": "...",  // ISO timestamp or null
    "baseline_valid": true         // wpm_effective > 0 AND duration >= 60 s
  }
}
```

This makes each packet **self-contained**: the training pipeline does not need to join `user_baselines` to use a packet. The baseline is embedded at write time so it reflects what the drift model actually used.

---

### 13. Packet sequencing + window bounds (Phase 8 update)

**Backend:** `services/api/app/db/models.py`, `services/api/app/db/engine.py`

Three new columns added to `session_state_packets` (idempotent `ALTER TABLE ADD COLUMN IF NOT EXISTS` at startup):

| Column | Type | Description |
|--------|------|-------------|
| `packet_seq` | `INTEGER` | Monotonically increasing counter per session (0-indexed). Computed as `MAX(packet_seq) + 1` at insert time. |
| `window_start_at` | `TIMESTAMPTZ` | Start of the 30 s telemetry window = `window_end_at − 30 s` |
| `window_end_at` | `TIMESTAMPTZ` | Timestamp of the newest telemetry batch in the window used to compute this packet |

These fields appear in `state_packets.jsonl` export rows:
```json
{
  "session_id": 42,
  "created_at": "2026-03-15T10:05:00Z",
  "packet_seq": 3,
  "window_start_at": "2026-03-15T10:04:30Z",
  "window_end_at": "2026-03-15T10:05:00Z",
  "packet": { ... }
}
```

**Why:** The LLM classifier needs to know *which* telemetry batches contributed to each packet. `window_start_at` / `window_end_at` enable precise alignment of JSONL packets with CSV rows.

---

### 14. Adaptive assistant panel shell (Phase 8 update)

**File:** `apps/desktop/src/pages/ReaderPage.tsx`

A minimal `<aside id="assistant-panel-root">` panel is rendered **only** when `session.mode === "adaptive"`. Baseline and calibration sessions are completely unaffected — they see no panel and no change to the reader layout.

**Behaviour:**
- The panel is mounted for the lifetime of an adaptive session (never unmounted, only visually collapsed) so DOM zone detection is stable.
- A collapse toggle minimises the panel to ~44 px wide without unmounting it.
- `panelOpen` state is passed to `useTelemetry` to compute `ui_context`.
- A `panelRef` is passed to `useTelemetry` so click/mousemove/keydown events inside the panel are detected and recorded as `interaction_zone="panel"`.

**Panel CSS tokens used:** `var(--bg-surface)`, `var(--border)`, `var(--text-muted)`, `var(--text)` — consistent with existing reader styling.

**ID / data-testid contracts:**
- Reader scroll container: `id="reader-root"`, `data-testid="reader-root"`
- Panel: `id="assistant-panel-root"`, `data-testid="assistant-panel-root"`

---

### 15. `protocol_tag` query parameter (Phase 8 update)

**File:** `services/api/app/routers/exports.py`

Both export endpoints now accept an optional `protocol_tag` that is recorded in `session_meta.json`:

```
GET /sessions/{id}/export/bundle?protocol_tag=pilot_2026
POST /exports/sessions  { "session_ids": [...], "protocol_tag": "adhd_cohort_1" }
```

`protocol_tag` is `null` in `session_meta.json` when not provided. Use it to distinguish data collection runs (e.g., different participant cohorts, labelling protocols, or experimental conditions) in downstream analysis.

---

## Test coverage

**Backend:** `services/api/tests/test_exports.py` (30 tests)

| Test class | What is verified |
|------------|-----------------|
| `TestExportAuth` | All 3 endpoints return 401 without a token |
| `TestExportOwnership` | 404 when accessing another user's session |
| `TestSingleSessionExport` | All required files written; non-empty; correct headers; `session_meta.json` keys; `state_packets.jsonl` exists |
| `TestZipDownload` | `Content-Type: application/zip`; ZIP is valid and contains `session_meta.json` |
| `TestBatchExport` | Single ID succeeds; non-existent ID appears in `errors`; empty list returns 400 |
| `TestBaselineEndpoint` | 404 when no baseline exists |
| `TestCsvFlattener` | Known columns extracted; missing columns default to `""`; `payload_json` always present; all `TELEMETRY_COLUMNS` in output; ISO timestamp; `ui_context` and `interaction_zone` columns present and populated |
| `TestStatePacketFields` | `state_packets.jsonl` rows include `packet_seq`, `window_start_at`, `window_end_at`, and `baseline_snapshot` |
| `TestProtocolTag` | `protocol_tag` recorded in `session_meta.json`; null when absent; works in batch export |

**Frontend:** `apps/desktop/src/test/`

| Test file | What is verified |
|-----------|-----------------|
| `telemetry.test.ts` | `ui_context` priority rules (USER_PAUSED > PANEL_INTERACTING > PANEL_OPEN > READ_MAIN); `InteractionZone` type values; defaults |
| `assistantPanel.test.ts` | Panel only renders for `adaptive` mode; baseline/calibration sessions emit `READ_MAIN` + `interaction_zone="reader"`; paused sessions emit `USER_PAUSED` regardless of mode |

All 302 backend tests pass. All 90 frontend tests pass.

---

## Architecture summary

```
POST /activity/batch
    │
    ▼
activity.py  ──►  drift/_recompute_and_save()
                       │
                       ├── loads user baseline (UserBaseline row)
                       ├── fetch_window(session_id)  → payload dicts (last 30 s)
                       ├── query MAX(created_at) for window_end_at
                       │
                       ▼
                  drift/model.py  ──►  DriftResult
                       │
                       ▼
                  drift/store.py  ──►  session_drift_states  (latest snapshot)
                                   ──►  session_drift_history (trajectory, every ~10 s)
                                   ──►  session_state_packets (full features, every ~10 s)
                                        ├── packet_seq   (monotonic counter)
                                        ├── window_start_at / window_end_at
                                        └── packet_json.baseline_snapshot (embedded)

Telemetry batch (every 2 s)
    ui_context ─────────────────────────────────────────────────► stored in payload JSONB
    interaction_zone ───────────────────────────────────────────► stored in payload JSONB
                                                                  exported in telemetry_batches.csv

GET /sessions/{id}/export/bundle?protocol_tag=...
    │
    ▼
exports/service.py  ──►  queries: sessions, user_baselines,
                                  activity_events, session_state_packets,
                                  document_chunks
                     ──►  writes: training/exports/user_{uid}/session_{sid}/
                                  session_meta.json   (includes protocol_tag)
                                  baseline.json
                                  state_packets.jsonl (includes packet_seq, window bounds)
                                  telemetry_batches.csv (includes ui_context, interaction_zone)
                                  events.csv
                                  document_chunks.csv   (if available)
```

---

## Phase 9 candidates

- **Manual labelling tool** — a lightweight Python CLI that presents each `state_packets.jsonl` row and accepts a keyboard shortcut label.
- **LLM classifier implementation** — fine-tune or prompt-engineer a small model on the labelled packets; integrate with `GET /sessions/{id}/state-packet`.
- **Automatic label heuristics** — bootstrap a label distribution using rule-based priors (e.g. `focusloss_share > 0.5 AND drift_ema > 0.6 → drifting`) for pre-labelling before manual review.
- **Intervention engine** — act on the classifier output to trigger adaptive interventions (pacing hints, break prompts, etc.) within the assistant panel shell.
