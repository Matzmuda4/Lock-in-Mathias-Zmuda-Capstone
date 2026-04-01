# TrainingData

This directory contains the **master unlabelled training dataset** for the Lock-in ADHD reading assistant LLM fine-tuning pipeline.

All files in this directory (except this README) are **git-ignored** and are generated locally by running export sessions through the app.

---

## Directory layout

```
TrainingData/
├── README.md                      ← this file (tracked)
├── unlabelled.jsonl               ← master append-only dataset (git-ignored)
├── baselines/
│   ├── user_1_baseline.json       ← latest calibration baseline per user
│   └── user_2_baseline.json
├── _append_index.json             ← deduplication index (session_id → max_packet_seq)
└── .append.lock                   ← ephemeral exclusive lock file
```

---

## How to populate

### Via the UI (recommended for dev)

1. Run an **adaptive** reading session to completion.
2. Click **📦 Export Bundle** in the reader toolbar.
3. The button automatically calls:
   ```
   GET /sessions/{id}/export/bundle?append_to_master=1
   ```
4. The response shows:
   - **Bundle folder** — per-session telemetry files
   - **Master JSONL path** — where packets were appended
   - **Baseline path** — the per-user baseline JSON written/updated
   - **appended_packet_count** — how many new packets were added

### Via `curl`

```bash
TOKEN="<your JWT>"
SESSION_ID=163

curl -s -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/sessions/${SESSION_ID}/export/bundle?append_to_master=1" \
  | python3 -m json.tool
```

---

## Master file format: `unlabelled.jsonl`

The file is **append-only** and uses a mixed format:

- **Comment/sentinel lines** start with `# ` and carry structural metadata (not valid JSON on their own).
- **Data lines** are valid JSON objects, one per line.

### Sentinel lines

| Prefix | When | Payload |
|---|---|---|
| `# SESSION_START` | Before first packet of a session | `{user_id, session_id, document_id, session_mode, started_at, ended_at, protocol_tag, baseline_valid, baseline_updated_at, baseline_ref}` |
| `# CHUNK_BREAK` | Before every 20th packet in a session | `{session_id, after_packet_seq, chunk_index}` |
| `# SESSION_END` | After last packet of a session | `{session_id, packet_count, first_packet_seq, last_packet_seq}` |

### Data line schema

Each data line is a JSON object with these top-level keys:

| Key | Type | Description |
|---|---|---|
| `key` | string | Stable unique ID: `u{user_id}_s{session_id}_p{packet_seq}` |
| `user_id` | int | Owning user |
| `session_id` | int | Session this packet belongs to |
| `packet_seq` | int | 0-based sequence number within the session |
| `created_at` | ISO datetime | When the packet was generated |
| `window_start_at` | ISO datetime | Start of the 30-second rolling window |
| `window_end_at` | ISO datetime | End of the 30-second rolling window |
| `drift` | object | Drift model output: `ema`, `level`, `beta_ema`, etc. |
| `features` | object | Raw window features: `idle_ratio_mean`, `pace_ratio`, etc. |
| `z_scores` | object | Normalised z-scores: `z_idle`, `z_focus_loss`, `z_jitter`, etc. |
| `ui_aggregates` | object | UI context shares over the 30s window |
| `baseline_snapshot` | object or null | Calibration baseline embedded in packet (null if no calibration) |
| `baseline_ref` | string | Relative path to the per-user baseline file |
| `packet_raw` | object | Full original packet JSON — lossless fallback |

### Example

```jsonl
# SESSION_START {"user_id": 1, "session_id": 163, "session_mode": "adaptive", ...}
{"key": "u1_s163_p0", "user_id": 1, "session_id": 163, "packet_seq": 0, "drift": {"ema": 0.12}, ...}
{"key": "u1_s163_p1", "user_id": 1, "session_id": 163, "packet_seq": 1, "drift": {"ema": 0.15}, ...}
...
# CHUNK_BREAK {"session_id": 163, "after_packet_seq": 19, "chunk_index": 1}
{"key": "u1_s163_p20", ...}
...
# SESSION_END {"session_id": 163, "packet_count": 61, "first_packet_seq": 0, "last_packet_seq": 60}
```

---

## Deduplication

`_append_index.json` tracks the **maximum `packet_seq` appended per session**:

```json
{
  "163": 60,
  "164": 29
}
```

Re-exporting a session only appends packets with `packet_seq > last_recorded`. This means the file is always append-only and safe to re-export without creating duplicates.

---

## Parsing in Python

Strip sentinel lines before loading as JSONL:

```python
import json
from pathlib import Path

def load_master(path: str = "TrainingData/unlabelled.jsonl"):
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("# "):
            records.append(json.loads(line))
    return records

packets = load_master()
print(f"Loaded {len(packets)} packets")
```

Or with pandas:

```python
import pandas as pd

rows = load_master()
df = pd.json_normalize(rows)
print(df[["key", "drift.ema", "z_scores.z_idle", "z_scores.z_focus_loss"]].head())
```

---

## Labelling with ChatGPT / LLM

Each data line is self-contained: it includes drift state, features, z-scores, UI context, and an embedded baseline snapshot.

To prepare a chunk for labelling, paste sentinel + data lines directly into your LLM context. The `# SESSION_START` line gives the labeller full session context, and `# CHUNK_BREAK` markers signal natural labelling boundaries.

See `docs/phase-8-training-export.md` for the full labelling system prompt.

---

## Per-user baseline files

`baselines/user_{user_id}_baseline.json` is **overwritten** on every export. It mirrors the `baseline.json` in the per-session bundle, making it easy for training scripts to load a user's calibration parameters without scanning session folders.

```bash
cat TrainingData/baselines/user_1_baseline.json | python3 -m json.tool
```
