# Lock-In — Training Data Pipeline

This folder holds the training-data export pipeline for the Lock-In LLM attentional-state classifier.

> **Privacy note** — All exports contain only anonymised behavioural signals (scroll velocity, idle time, drift scores, etc.).  No document content is exported.  Exports are git-ignored and must never be committed.

---

## Folder layout

```
training/
├── README.md            ← this file
└── exports/             ← git-ignored; auto-created on first export
    └── user_{uid}/
        └── session_{sid}/
            ├── session_meta.json       session + document metadata
            ├── baseline.json           user calibration baseline
            ├── state_packets.jsonl     periodic drift packets (~1 per 10 s)
            ├── telemetry_batches.csv   2-second telemetry aggregates
            ├── events.csv              non-batch activity events
            └── document_chunks.csv     chunk metadata (if available)
```

---

## Exporting one session

### Via HTTP (recommended)

```bash
# Obtain a bearer token first
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -d "username=YOUR_USER&password=YOUR_PASS" | jq -r .access_token)

# Export session 42
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/sessions/42/export/bundle | jq

# Download as a ZIP instead
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/sessions/42/export/bundle?download=1" \
  -o session_42.zip
```

### Via the UI (dev mode only)

Open a session in the reader page while running with `VITE_DEV=true` (default in dev).
Click **📦 Export Bundle** in the bottom area.  The folder path is shown inline after export.

---

## Exporting multiple sessions

```bash
curl -X POST http://localhost:8000/exports/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_ids": [42, 43, 44]}' | jq
```

Returns a summary of successes and errors.  Files are written to `training/exports/`.

---

## Output file reference

### `session_meta.json`
Key fields: `session_id`, `user_id`, `document_id`, `document_is_calibration`, `mode`, `status`, `started_at`, `ended_at`, `duration_seconds`.

### `baseline.json`
Full `user_baselines.baseline_json` v2 payload.  Includes:
- `wpm_gross`, `wpm_effective`
- `scroll_velocity_norm_mean/std`
- `scroll_jitter_mean`, `regress_rate_mean/std`
- `idle_ratio_mean/std`, `para_dwell_median_s/iqr_s`
- `presentation_profile` (viewport dimensions)

Set `baseline_valid: true` only when `wpm_effective > 0` and calibration duration ≥ 60 s.

### `state_packets.jsonl`
One JSON object per line, one per ~10 seconds during the session.  Each row:
```json
{
  "session_id": 42,
  "created_at": "2026-02-25T14:00:10Z",
  "packet": {
    "drift": { "drift_level": 0.08, "drift_ema": 0.07, "beta_ema": 0.021, ... },
    "features": { "idle_ratio_mean": 0.2, "scroll_velocity_norm_mean": 0.3, ... },
    "z_scores": { "z_idle": 0.4, "z_jitter": 0.1, ... },
    "debug": { ... }
  }
}
```

**Do not use `drift_ema` as a training label** — it is a model output and not independently verified.  These packets will be manually or intelligently labelled with attentional-state distributions (Focused / Drifting / Hyperfocused / Fatigued) in a later step.

### `telemetry_batches.csv`
One row per 2-second telemetry batch.  Columns (in order):
`created_at`, `session_id`, `scroll_delta_sum`, `scroll_delta_abs_sum`, `scroll_delta_pos_sum`, `scroll_delta_neg_sum`, `scroll_event_count`, `scroll_direction_changes`, `scroll_pause_seconds`, `idle_seconds`, `idle_since_interaction_seconds`, `mouse_path_px`, `mouse_net_px`, `window_focus_state`, `current_paragraph_id`, `current_chunk_index`, `viewport_progress_ratio`, `viewport_height_px`, `viewport_width_px`, `reader_container_height_px`, `telemetry_fault`, `scroll_capture_fault`, `paragraph_missing_fault`, `payload_json`.

### `events.csv`
Non-batch events: `progress_marker`, `blur`, `focus`, `intervention`, etc.  Columns: `created_at`, `session_id`, `event_type`, `payload_json`.

### `document_chunks.csv`
Optional.  Columns: `chunk_index`, `chunk_type`, `word_count`, `page_start`, `page_end`, `asset_id`, `text_preview`.

---

## Using outputs for training

1. **Collect sessions** — run calibration + several reading sessions in the app.
2. **Export** — use the batch endpoint or the UI button.
3. **Label** — open `state_packets.jsonl` and assign attentional-state labels (Focused / Drifting / Hyperfocused / Fatigued) based on the signal context.  Suggested tool: a simple Python labelling script that presents each packet and asks for a label.
4. **Verify alignment** — cross-reference `telemetry_batches.csv` timestamps with `state_packets.jsonl` for consistency.
5. **Train** — feed labelled packets into your LLM fine-tuning pipeline.  The `baseline.json` should be included as system context so the model understands each user's personal reading baseline.

---

## Inspect your baseline

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/exports/users/me/baseline | jq
```
