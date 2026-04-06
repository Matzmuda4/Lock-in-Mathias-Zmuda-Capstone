"""
merge_labels.py
───────────────
Merges ChatGPT label files back into the full skeleton dataset,
producing intervention_training_v2_labelled.jsonl ready for format_for_training.py.

Usage:
    python3 TrainingData/merge_labels.py

Paste ChatGPT's JSON response directly into the batch_NNN.json file (replacing
all existing content), then run this script. Full input context is always
restored from intervention_training_v2_skeletons.jsonl — batch files only need
to carry the id + label fields.

Supported ChatGPT output formats (all handled automatically):

  Format A — flat, with intervene key (batch_001 style):
    [{"id": "...", "intervene": true, "tier": "moderate", "content": {...}}, ...]

  Format B — nested under output key (batch_002 style):
    [{"id": "...", "output": {"type": "...", "tier": "...", "content": "..."}}, ...]

  Format C — flat, no intervene key (minimal):
    [{"id": "...", "tier": "subtle", "content": "..."}, ...]

Output:
    TrainingData/intervention_training_v2_labelled.jsonl  — complete merged dataset
    (prints a QA report on any missing/invalid labels)
"""

import json
import collections
from pathlib import Path

SKEL_PATH = Path("TrainingData/intervention_training_v2_skeletons.jsonl")
BATCH_DIR = Path("TrainingData/batches")
OUT_PATH  = Path("TrainingData/intervention_training_v2_labelled.jsonl")

VALID_TYPES = {
    "focus_point", "section_summary", "comprehension_check", "re_engagement",
    "ambient_sound", "chime", "text_reformat", "break_suggestion", "gamification", "none",
}
VALID_TIERS = {"subtle", "moderate", "strong", "special", "none"}


def is_skeleton_file(data: list) -> bool:
    """
    Skeleton/batch files always have an 'input' key (session context,
    attentional window, etc.). Label-only responses from ChatGPT never do —
    they only carry id + tier + content (+ optionally intervene).
    """
    if not data or not isinstance(data, list):
        return False
    first = data[0]
    return "input" in first


def extract_label_fields(entry: dict) -> tuple:
    """
    Normalise a label entry regardless of which format ChatGPT used.
    Returns (intervene, tier, content).

    Format A: {"id": ..., "intervene": bool, "tier": str, "content": ...}
    Format B: {"id": ..., "output": {"type": str, "tier": str, "content": ...}}
    Format C: {"id": ..., "tier": str, "content": ...}   (intervene inferred)
    """
    # Format B — content nested under "output"
    if "output" in entry and isinstance(entry["output"], dict):
        inner    = entry["output"]
        tier     = inner.get("tier", "")
        content  = inner.get("content")
        intervene = entry.get("intervene", inner.get("intervene", True))
        return intervene, tier, content

    # Format A / C — flat
    tier      = entry.get("tier", "")
    content   = entry.get("content")
    intervene = entry.get("intervene", True)
    return intervene, tier, content


# ─── Load all label data ───────────────────────────────────────────────────────

label_map: dict[str, dict] = {}
all_batch_files = sorted(BATCH_DIR.glob("batch_*.json"))

for bf in all_batch_files:
    with open(bf) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ERROR parsing {bf.name}: {e}")
            continue

    if not isinstance(data, list) or not data:
        continue

    if is_skeleton_file(data):
        print(f"  SKIP {bf.name} — still contains skeleton data (not yet labelled)")
        continue

    count = 0
    for entry in data:
        row_id = entry.get("id")
        if row_id:
            label_map[row_id] = entry
            count += 1
    print(f"  {bf.name}: {count} labels loaded")

if not label_map:
    print("\nNo label data found in TrainingData/batches/")
    print("Paste ChatGPT responses into the batch_NNN.json files and re-run.")
    raise SystemExit(1)

print(f"\nTotal labels loaded: {len(label_map)}")

# ─── Load skeletons ────────────────────────────────────────────────────────────

skeletons: list[dict] = []
with open(SKEL_PATH) as f:
    for line in f:
        skeletons.append(json.loads(line))

print(f"Skeletons  : {len(skeletons)}")

# ─── Merge ────────────────────────────────────────────────────────────────────

merged: list[dict] = []
stats = collections.Counter()
qa_issues: list[str] = []

for skel in skeletons:
    row_id = skel["id"]
    out    = skel.get("output", {})

    # Pre-labelled rows (chime, text_reformat, ambient_sound, none) — pass through
    if not out.get("pending_label"):
        merged_row = dict(skel)
        merged_row["output"] = {
            "intervene": out.get("intervene", True),
            "type"     : out["type"],
            "tier"     : out["tier"],
            "content"  : out.get("content"),
            "rationale": out.get("rationale", ""),
        }
        merged.append(merged_row)
        stats["pre_labelled"] += 1
        continue

    # Pending rows — must have a matching label
    if row_id not in label_map:
        stats["missing_label"] += 1
        qa_issues.append(f"MISSING label for {row_id} (type={out.get('type')})")
        continue

    label = label_map[row_id]
    l_intv, l_tier, l_content = extract_label_fields(label)

    # Type is always taken from the skeleton (ChatGPT cannot change it)
    l_type = out.get("type")

    # Tier validation — fall back to skeleton tier if invalid
    if l_tier not in VALID_TIERS:
        qa_issues.append(f"INVALID tier '{l_tier}' for {row_id} — using skeleton tier")
        l_tier = out.get("tier", "subtle")

    # Content validation
    if l_type == "none":
        l_content = None                              # force null for none-type
    elif l_content is None:
        qa_issues.append(f"NULL content for {l_type} in {row_id}")
        stats["null_content"] += 1

    merged_row = dict(skel)
    merged_row["output"] = {
        "intervene": bool(l_intv),
        "type"     : l_type,
        "tier"     : l_tier,
        "content"  : l_content,
        "rationale": out.get("rationale", ""),
    }
    merged.append(merged_row)
    stats["labelled"] += 1

# ─── Write output ──────────────────────────────────────────────────────────────

with open(OUT_PATH, "w") as f:
    for row in merged:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nMerge complete → {OUT_PATH}")
print(f"  pre_labelled : {stats['pre_labelled']}")
print(f"  labelled     : {stats['labelled']}")
print(f"  missing_label: {stats['missing_label']}")
print(f"  null_content : {stats['null_content']}")

# ─── QA Report ────────────────────────────────────────────────────────────────

if qa_issues:
    print(f"\nQA Issues ({len(qa_issues)}):")
    for issue in qa_issues[:30]:
        print(f"  {issue}")
    if len(qa_issues) > 30:
        print(f"  ... and {len(qa_issues) - 30} more")
else:
    print("\nNo QA issues — dataset looks clean!")

# ─── Final distribution ────────────────────────────────────────────────────────

type_dist = collections.Counter(r["output"]["type"] for r in merged)
print("\nFinal type distribution:")
for t, n in sorted(type_dist.items()):
    print(f"  {t:28s} {n}")
print(f"  {'TOTAL':28s} {len(merged)}")

print(f"\nNext step: python3 TrainingData/format_for_training.py")
