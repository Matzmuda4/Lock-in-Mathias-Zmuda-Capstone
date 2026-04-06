"""
extract_batches.py
──────────────────
Splits the pending-label rows from intervention_training_v2_skeletons.jsonl
into numbered batch files ready for ChatGPT labelling.

Usage:
    python3 TrainingData/extract_batches.py

    Options (edit constants below):
      BATCH_SIZE   — rows per batch (default 20, fits comfortably in ChatGPT)
      SKIP_ROWS    — skip the first N pending rows (use when some batches are
                     already labelled and you don't want to overwrite them)
      START_BATCH  — number of the first batch file to write (e.g. 2 to start
                     at batch_002.json when batch_001 is already done)

Outputs:
    TrainingData/batches/batch_NNN.json

After ChatGPT returns labels, paste the JSON response DIRECTLY into the
batch file (overwriting it), then run merge_labels.py.
"""

import json
import math
from pathlib import Path

SKEL_PATH   = Path("TrainingData/intervention_training_v2_skeletons.jsonl")
BATCH_DIR   = Path("TrainingData/batches")
BATCH_SIZE  = 20   # smaller batches fit easily in ChatGPT
SKIP_ROWS   = 40   # batch_001 (40 rows) is already labelled — skip those
START_BATCH = 2    # start writing at batch_002.json

BATCH_DIR.mkdir(parents=True, exist_ok=True)

# Load only the pending rows
all_pending = []
with open(SKEL_PATH) as f:
    for line in f:
        row = json.loads(line)
        if row.get("output", {}).get("pending_label"):
            all_pending.append(row)

print(f"Total pending rows: {len(all_pending)}")

# Skip already-labelled rows
pending = all_pending[SKIP_ROWS:]
print(f"Skipping first {SKIP_ROWS} rows (already labelled).")
print(f"Rows remaining to batch: {len(pending)}")

n_batches = math.ceil(len(pending) / BATCH_SIZE)
print(f"Will create {n_batches} batches of up to {BATCH_SIZE} rows each "
      f"(starting at batch_{START_BATCH:03d}.json)\n")

for i in range(n_batches):
    batch      = pending[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    batch_num  = START_BATCH + i
    batch_path = BATCH_DIR / f"batch_{batch_num:03d}.json"

    trimmed = []
    for row in batch:
        trimmed.append({
            "id": row["id"],
            "output": {
                "type": row["output"]["type"],
                "tier": row["output"]["tier"],
                "rationale": row["output"].get("rationale", ""),
            },
            "input": {
                "session_context": row["input"].get("session_context", {}),
                "attentional_state_window": row["input"].get("attentional_state_window", []),
                "drift_progression": row["input"].get("drift_progression", {}),
                "reading_context": row["input"].get("reading_context", {}),
            },
        })

    with open(batch_path, "w") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {batch_path} ({len(trimmed)} rows)")

print(f"\nDone. Label each batch file using LABELLING_PROMPT_V2.md.")
print(f"Paste ChatGPT's JSON response directly into the batch file, then run:")
print(f"    python3 TrainingData/merge_labels.py")
