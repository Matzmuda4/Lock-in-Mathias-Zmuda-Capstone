#!/usr/bin/env python3
"""
merge_supervised.py
───────────────────
Merge unlabelled.jsonl + labelled.jsonl into supervised.jsonl.

Only NEW entries are appended — packets whose key already appears in
supervised.jsonl are skipped, so this script is safe to run at any time
without duplicating data.

Usage (from TrainingData/ directory):
    python merge_supervised.py

Typical workflow:
    1. python generate_synthetic.py   ← adds new sessions to unlabelled + labelled
    2. python merge_supervised.py     ← folds them into supervised.jsonl
"""

import json
from pathlib import Path

BASE        = Path(__file__).parent
UNLABELLED  = BASE / "unlabelled.jsonl"
LABELLED    = BASE / "labelled.jsonl"
SUPERVISED  = BASE / "supervised.jsonl"


def _load_supervised_keys() -> set:
    """Return the set of 'key' values already written to supervised.jsonl."""
    keys: set[str] = set()
    if not SUPERVISED.exists():
        return keys
    with open(SUPERVISED) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "key" in row:
                    keys.add(row["key"])
            except Exception:
                pass
    return keys


def _load_unlabelled() -> dict[str, dict]:
    """Return a dict keyed by 'key' containing packet data from unlabelled.jsonl."""
    packets: dict[str, dict] = {}
    if not UNLABELLED.exists():
        print(f"ERROR: {UNLABELLED} not found.")
        return packets
    with open(UNLABELLED) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
                packets[row["key"]] = row
            except Exception as exc:
                print(f"  WARN: could not parse unlabelled line: {exc}")
    return packets


def _load_labelled() -> dict[tuple, dict]:
    """Return a dict keyed by (session_id, packet_seq) from labelled.jsonl."""
    labels: dict[tuple, dict] = {}
    if not LABELLED.exists():
        print(f"ERROR: {LABELLED} not found.")
        return labels
    with open(LABELLED) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = (row["session_id"], row["packet_seq"])
                labels[key] = row
            except Exception as exc:
                print(f"  WARN: could not parse labelled line: {exc}")
    return labels


def main():
    existing_keys = _load_supervised_keys()
    print(f"supervised.jsonl: {len(existing_keys)} existing entries")

    packets = _load_unlabelled()
    labels  = _load_labelled()
    print(f"unlabelled.jsonl: {len(packets)} packets")
    print(f"labelled.jsonl  : {len(labels)} label rows")

    new_rows: list[str] = []
    matched = skipped = unmatched = 0

    for pk, packet in sorted(packets.items(),
                              key=lambda x: (x[1]["session_id"], x[1]["packet_seq"])):
        if pk in existing_keys:
            skipped += 1
            continue

        sid  = packet["session_id"]
        seq  = packet["packet_seq"]
        lkey = (sid, seq)

        if lkey not in labels:
            unmatched += 1
            continue

        lbl = labels[lkey]
        merged = {
            "key":              pk,
            "user_id":          packet["user_id"],
            "session_id":       sid,
            "packet_seq":       seq,
            "created_at":       packet["created_at"],
            "window_start_at":  packet["window_start_at"],
            "window_end_at":    packet["window_end_at"],
            "drift":            packet["drift"],
            "features":         packet["features"],
            "z_scores":         packet["z_scores"],
            "ui_aggregates":    packet["ui_aggregates"],
            "baseline_snapshot": packet["baseline_snapshot"],
            "baseline_ref":     packet.get("baseline_ref", ""),
            "packet_raw":       packet.get("packet_raw", {}),
            "labels":           lbl["labels"],
            "primary_state":    lbl["primary_state"],
            "notes":            lbl.get("notes", ""),
        }
        new_rows.append(json.dumps(merged, ensure_ascii=False))
        matched += 1

    if new_rows:
        with open(SUPERVISED, "a") as fh:
            fh.write("\n".join(new_rows) + "\n")
        print(f"\nAppended {matched} new entries to supervised.jsonl")
    else:
        print("\nNothing new to append — supervised.jsonl is already up-to-date.")

    if skipped:
        print(f"Skipped {skipped} packet(s) already in supervised.jsonl.")
    if unmatched:
        print(f"WARNING: {unmatched} packet(s) had no matching label row in labelled.jsonl.")


if __name__ == "__main__":
    main()
