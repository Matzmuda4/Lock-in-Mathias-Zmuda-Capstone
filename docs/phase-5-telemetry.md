# Phase 5 — Telemetry Logging v1

> **Scope:** Minimal, high-performance telemetry ingestion.  
> No calibration, no drift math, no interventions.  
> Raw signals are stored in TimescaleDB for later analysis.

---

## Overview

Every 2 seconds the reader frontend aggregates user interaction signals and sends a single batch to the backend.  The backend validates ownership, then inserts **one row** into the existing `activity_events` TimescaleDB hypertable.  This is the only new database activity; no new tables are added.

---

## New Endpoint

### `POST /activity/batch`

| Field | Value |
|---|---|
| Auth | Bearer JWT required |
| Session constraint | Session must be **active** (not paused, not ended) |
| DB write | One row in `activity_events` (`event_type = "telemetry_batch"`) |
| Response | `201 Created` with `{id, session_id, event_type, created_at}` |

#### Request payload shape

```json
{
  "session_id": 42,

  // Scroll signals
  "scroll_delta_sum": 240.0,          // net signed displacement (px, down = +)
  "scroll_delta_abs_sum": 240.0,      // total absolute displacement (px, >= 0)
  "scroll_event_count": 4,            // raw number of scroll events in window
  "scroll_direction_changes": 1,      // direction reversals (backtrack proxy)
  "scroll_pause_seconds": 0.5,        // seconds since last scroll (capped 60 s)

  // Engagement signals
  "idle_seconds": 0.5,                // seconds since any interaction (capped 60 s)

  // Mouse signals
  "mouse_path_px": 320.0,             // total cursor path length (px)
  "mouse_net_px": 200.0,              // straight-line net displacement (px)

  // Window / focus
  "window_focus_state": "focused",    // "focused" | "blurred"

  // Reading-position
  "current_paragraph_id": "chunk-7", // data-paragraph-id of most-visible element
  "current_chunk_index": 7,          // chunk array index of most-visible element
  "viewport_progress_ratio": 0.35,   // scrollTop / (scrollHeight - clientHeight), 0..1

  // Timestamp (ISO-8601; server uses now() if absent)
  "client_timestamp": "2026-02-25T12:00:00Z"
}
```

All numeric fields default to `0` / `null` — the client only needs to send non-zero values.

#### Validation rules (Pydantic)

| Field | Constraint |
|---|---|
| `scroll_delta_abs_sum` | `≥ 0` |
| `scroll_event_count` | `≥ 0`, must be `int` |
| `scroll_direction_changes` | `≥ 0`, must be `int` |
| `scroll_pause_seconds` | `≥ 0` |
| `idle_seconds` | `≥ 0` |
| `mouse_path_px` | `≥ 0` |
| `mouse_net_px` | `≥ 0` |
| `viewport_progress_ratio` | `0 ≤ x ≤ 1` |

Violations → `422 Unprocessable Entity`.

---

## New / Changed Files

### Backend

| File | Change |
|---|---|
| `services/api/app/schemas/activity.py` | Added `ActivityBatchCreate` and `ActivityBatchResponse` schemas |
| `services/api/app/routers/activity.py` | Added `POST /activity/batch` handler |
| `services/api/tests/test_activity_batch.py` | **New** — 12 tests (insert, auth, ownership, validation) |

The existing `POST /activity` (single-event endpoint) is untouched.

### Frontend

| File | Change |
|---|---|
| `apps/desktop/src/hooks/useTelemetry.ts` | **New** — main telemetry hook + exported pure helpers |
| `apps/desktop/src/services/activityService.ts` | **New** — `activityService.postBatch()` |
| `apps/desktop/src/pages/ReaderPage.tsx` | Wired `useTelemetry`; added `data-paragraph-id` / `data-word-count` / `data-chunk-index` attrs; debug panel; telemetry indicator; larger typography |
| `apps/desktop/src/test/telemetry.test.ts` | **New** — 13 unit tests for `computeMouseStats` and `selectCurrentParagraph` |

---

## How Current Paragraph ID Works

The `current_paragraph_id` field is populated by an `IntersectionObserver` running inside `useTelemetry`.

1. Every `TextChunk` element rendered in `ReaderPage` gets three HTML data attributes:

   ```html
   <div
     data-paragraph-id="chunk-{chunk.id}"
     data-word-count="{wordCount}"
     data-chunk-index="{arrayIndex}"
   >
   ```

2. The hook registers all `[data-paragraph-id]` elements with an `IntersectionObserver` (root = the scrollable `<main>` container, thresholds = `[0, 0.25, 0.5, 0.6, 0.75, 1.0]`).

3. A `MutationObserver` watches for new children so that lazily-loaded chunks (after "Load more") are automatically observed.

4. The **pure function** `selectCurrentParagraph(entries)` picks the element with the **highest intersection ratio ≥ 0.6**.  If no element clears 0.6 the last known value is retained.

5. The selected `paragraphId` and `chunkIndex` are stored in refs and included in every telemetry batch.

---

## ReaderPage Changes

### Typography (reading-optimised)

| Element | Before | After |
|---|---|---|
| Body paragraph font size | 15 px | 19 px |
| Line height | 1.8 | 1.75 |
| Max width | 740 px | 72 ch |
| Heading font size | 18 px | 22 px |

### Telemetry indicator (dev-only)

A small badge `⊙ Telemetry: ON / OFF` appears in the top-right bar when `import.meta.env.DEV` is true.  It turns green when the session is active and telemetry is flowing.

### Debug panel (dev-only)

A collapsible panel fixed to the bottom-right corner shows the **last batch JSON** that was sent.  Use it to verify signals in real time without opening DevTools.

---

## Architecture

```
ReaderPage
  └── useTelemetry(sessionId, token, active, containerRef)
        ├── scroll listener → accumulates delta_sum, abs_sum, event_count,
        │                      direction_changes
        ├── mousemove listener → pushes to mousePoints[]
        ├── keydown listener → updates lastInteraction
        ├── blur / focus listeners → windowFocused ref
        ├── IntersectionObserver → currentParagraphId, currentChunkIndex
        └── setInterval(2s) → flush()
              └── activityService.postBatch(token, batch)
                    └── POST /activity/batch
                          └── activity_events hypertable (one row)
```

Key design principles:
- **Refs only** for mutable accumulators — avoids unnecessary re-renders.
- Event listeners attached once; `activeRef` checked on each event so they stay stable even when the `active` prop changes.
- `activityService.postBatch` **silently swallows errors** — telemetry must never crash the reader.

---

## Manual Verification Steps

1. Start the backend: `uvicorn app.main:app --reload` (from `services/api/`)
2. Open the frontend: `npm run dev` (from `apps/desktop/`)
3. Log in, upload a PDF, parse it, start a session.
4. In the Reader, open DevTools → Network tab, filter for `activity/batch`.
5. Every 2 seconds you should see a `201` POST.  Inspect the request body.
6. In the **debug panel** (bottom-right corner, dev mode only), expand "Last telemetry batch" to see the latest JSON without DevTools.
7. Scroll through the document — `scroll_event_count`, `scroll_delta_abs_sum`, and `current_paragraph_id` should update.
8. Move the mouse — `mouse_path_px` should increase.
9. Switch tabs — the next batch should show `window_focus_state: "blurred"`.
10. Pause the session — telemetry stops (no more POSTs).  Resume — telemetry restarts.

To query stored batches directly in the DB:

```sql
SELECT created_at, payload
FROM activity_events
WHERE event_type = 'telemetry_batch'
ORDER BY created_at DESC
LIMIT 10;
```

---

## Test Coverage

### Backend (`pytest`)

| Test | Description |
|---|---|
| `test_batch_inserts_row` | Row inserted with `event_type=telemetry_batch` |
| `test_batch_payload_keys_stored` | Each call creates a new distinct row |
| `test_batch_minimal_payload` | Only `session_id` required; all others default |
| `test_requires_auth` | No token → 401 |
| `test_wrong_bearer_returns_401` | Invalid token → 401 |
| `test_enforces_ownership` | User B cannot post for User A session → 404 |
| `test_unknown_session_returns_404` | Non-existent session → 404 |
| `test_missing_session_id_returns_422` | Missing required field → 422 |
| `test_invalid_scroll_count_type_returns_422` | Wrong type → 422 |
| `test_negative_abs_sum_returns_422` | Negative constrained field → 422 |
| `test_viewport_ratio_out_of_range_returns_422` | Out-of-range 0..1 → 422 |
| `test_ended_session_returns_404` | Ended session → 404 |

### Frontend (`vitest`)

| Test | Description |
|---|---|
| `computeMouseStats` — 6 tests | zeros for empty/single, horizontal, diagonal (3-4-5), zig-zag > straight, negative coords |
| `selectCurrentParagraph` — 7 tests | empty, all below threshold, single above, best of many, exactly 0.6, null chunkIndex |

---

## Known Limitations / Deferred

- **No drift model yet.** Raw batches are stored; drift computation is Phase 6.
- **No calibration.** Scroll and mouse magnitudes are absolute pixel values — normalization (per viewport height etc.) will be added when the drift model is designed.
- **Paused sessions excluded.** The backend rejects batches for paused sessions.  If this needs to change for partial tracking, update the `status == "active"` check in `activity.py`.
