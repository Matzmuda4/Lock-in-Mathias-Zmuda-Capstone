# Lock-In — Training Data Pipeline (classify branch)

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

---

## Consolidated training export (NEW — classify branch)

The `/training/packets/export` endpoints produce a **single flat file** across
all sessions — the primary input for PEFT/QLoRA fine-tuning.

### Folder layout

```
training/
├── data/                         ← git-ignored; consolidated export files
│   ├── packets_user_1_20260225T120000.csv
│   └── packets_user_1_20260225T120000.jsonl
└── exports/                      ← git-ignored; per-session bundles
    └── user_{uid}/session_{sid}/
        ├── session_meta.json
        ├── baseline.json
        ├── state_packets.jsonl
        ├── telemetry_batches.csv
        ├── events.csv
        └── document_chunks.csv
```

### Quick-start

```bash
# Export all your packets to a CSV (default)
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/training/packets/export" | jq

# Export as JSONL with debug fields
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/training/packets/export?format=jsonl&include_debug=true" | jq

# Download the CSV directly
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/training/packets/export?download=true" \
  -o training_data.csv

# Export specific sessions only
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/training/packets/export?session_ids=42,43,44" | jq

# Filter by session mode
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/training/packets/export?mode=adaptive" | jq

# POST variant (body-driven, good for many session IDs)
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_ids":[42,43],"format":"jsonl","include_debug":false}' \
  http://localhost:8000/training/packets/export | jq
```

### Packet cadence

A state packet is written to `session_state_packets` every **exactly 10 seconds**
of active reading (every 5th telemetry batch at 2-second cadence).  Paused
sessions do not generate packets.  Each packet is self-contained:

- `baseline_snapshot` — calibration baseline at packet creation time
- `features` — 30-second rolling feature vector (scroll, idle, pace, etc.)
- `z_scores` — normalised deviations vs baseline
- `drift` — model output (NOT training labels — must be labelled separately)
- `ui_aggregates` — UI context distributions (panel vs reader, etc.)
- `session_id`, `user_id`, `document_id`, `session_mode` — identity

### Stable CSV columns (schema v2.0.0)

| Group | Example columns |
|---|---|
| Identifiers | `session_id`, `user_id`, `document_id`, `packet_seq`, `session_mode` |
| Baseline | `bl_wpm_effective`, `bl_idle_ratio_mean`, `bl_regress_rate_mean` |
| Features | `feat_idle_ratio_mean`, `feat_scroll_burstiness`, `feat_pace_available` |
| Z-scores | `z_idle`, `z_focus_loss`, `z_skim`, `z_burstiness` |
| Drift | `drift_ema`, `disruption_score`, `engagement_score` (not labels) |
| UI | `panel_share_30s`, `reader_share_30s`, `iz_panel_share_30s` |
| Lossless | `packet_json` (full JSON of the packet row) |

### Using in Colab

```python
import pandas as pd
import json

# Load CSV
df = pd.read_csv("packets_user_1_20260225T120000.csv")

# Expand baseline from lossless JSON if you need extra fields
df["_pj"] = df["packet_json"].apply(json.loads)
df["bl_full"] = df["_pj"].apply(lambda x: x.get("baseline_snapshot", {}).get("baseline_json", {}))

# The drift_ema column is a model output — do NOT use as labels.
# Add your own attentional-state labels before fine-tuning.
df["label"] = None  # TODO: label manually or with a weak-supervision approach

print(df[["session_id","packet_seq","feat_idle_ratio_mean","z_idle","drift_ema","label"]].head())
```
