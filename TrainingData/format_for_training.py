"""
format_for_training.py
──────────────────────
Converts intervention_training_final.jsonl (Alpaca-style) into two files
ready for QLoRA fine-tuning on Qwen 2.5 7B:

  intervention_train.jsonl   (~90%, 1309 examples)
  intervention_eval.jsonl    (~10%,  146 examples)

Each line contains a single key "text" whose value is a fully-formatted
Qwen ChatML string that SFTTrainer / Unsloth can consume directly.

Format:
  <|im_start|>system
  {instruction}<|im_end|>
  <|im_start|>user
  {json(input)}<|im_end|>
  <|im_start|>assistant
  {json(output)}<|im_end|>

Run:
  python TrainingData/format_for_training.py

Outputs land in the same TrainingData/ directory.
"""

from __future__ import annotations
import json
import pathlib
import random

random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = pathlib.Path(__file__).parent
SRC         = HERE / "intervention_training_final.jsonl"
TRAIN_OUT   = HERE / "intervention_train.jsonl"
EVAL_OUT    = HERE / "intervention_eval.jsonl"

# Qwen 2.5 uses ChatML; these special tokens must match exactly.
BOS = "<|im_start|>"
EOS = "<|im_end|>"

EVAL_FRACTION = 0.10


def format_example(ex: dict) -> str:
    """Collapse one Alpaca record into a single ChatML training string."""
    instruction = ex["instruction"]
    inp         = ex["input"]
    out         = ex["output"]

    # Compact JSON — fewer tokens, cleaner for structured-output training
    user_msg = json.dumps(inp, ensure_ascii=False, separators=(",", ":"))
    asst_msg = json.dumps(out, ensure_ascii=False, separators=(",", ":"))

    return (
        f"{BOS}system\n{instruction}{EOS}\n"
        f"{BOS}user\n{user_msg}{EOS}\n"
        f"{BOS}assistant\n{asst_msg}{EOS}"
    )


def main() -> None:
    # ── Load source ──────────────────────────────────────────────────────────
    records: list[dict] = []
    with SRC.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} examples from {SRC.name}")

    # ── Format ───────────────────────────────────────────────────────────────
    formatted = [{"text": format_example(r)} for r in records]

    # ── Stratified shuffle then split ────────────────────────────────────────
    # Stratify by intervention type so eval covers all 10 types.
    from collections import defaultdict
    by_type: dict[str, list] = defaultdict(list)
    for item, rec in zip(formatted, records):
        by_type[rec["output"]["type"]].append(item)

    train_items: list[dict] = []
    eval_items:  list[dict] = []

    for itype, items in by_type.items():
        random.shuffle(items)
        n_eval = max(1, round(len(items) * EVAL_FRACTION))
        eval_items.extend(items[:n_eval])
        train_items.extend(items[n_eval:])

    random.shuffle(train_items)
    random.shuffle(eval_items)

    # ── Write ─────────────────────────────────────────────────────────────────
    for path, items in [(TRAIN_OUT, train_items), (EVAL_OUT, eval_items)]:
        with path.open("w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Written: {path.name}  ({len(items)} examples)")

    # ── Sanity-check first example ────────────────────────────────────────────
    print("\n── First training example preview ───────────────────────────────")
    first = train_items[0]["text"]
    print(first[:600] + "\n  ...[truncated]")

    # ── Token-length estimate ─────────────────────────────────────────────────
    all_lens = [len(item["text"]) // 4 for item in train_items + eval_items]
    print(f"\n── Approx token lengths ─────────────────────────────────────────")
    print(f"  min={min(all_lens)}  avg={sum(all_lens)//len(all_lens)}  max={max(all_lens)}")
    print(f"  All examples fit within 2048 tokens: {all(l <= 2048 for l in all_lens)}")


if __name__ == "__main__":
    main()
