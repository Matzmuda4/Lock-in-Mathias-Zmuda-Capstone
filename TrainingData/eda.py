"""
EDA — Classification Spread Analysis
TrainingData/eda.py

Run from the TrainingData directory:
    python eda.py

Or from the repo root:
    python TrainingData/eda.py

Reads supervised.jsonl and prints a full breakdown of:
  - Primary-state hard label distribution (count + %)
  - Soft label mass per state (sum of probability weights across all packets)
  - Per-session breakdown
  - Per-user breakdown
  - Signal averages per primary state (drift, z_scores)
  - Class imbalance warning
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# ── Locate data file ─────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA = HERE / "supervised.jsonl"

if not DATA.exists():
    raise FileNotFoundError(f"Expected {DATA} — run from repo root or TrainingData/")

STATES = ["focused", "drifting", "hyperfocused", "cognitive_overload"]

# ── Load ─────────────────────────────────────────────────────────────────────
records = []
with open(DATA) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

N = len(records)

# ── Helpers ──────────────────────────────────────────────────────────────────
SEP    = "─" * 64
SUBSEP = "  " + "·" * 60

def bar(value: float, total: float, width: int = 30) -> str:
    filled = int(round(value / total * width)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)

def mean(values):
    return sum(values) / len(values) if values else 0.0

def pct(n, total):
    return (n / total * 100) if total > 0 else 0.0

# ── 1. Primary-state hard label distribution ─────────────────────────────────
print(f"\n{SEP}")
print(f"  Lock-In EDA — Classification Spread")
print(f"  Total labelled packets : {N}")
print(SEP)

primary_counts = defaultdict(int)
for r in records:
    primary_counts[r["primary_state"]] += 1

print("\n  1. PRIMARY STATE DISTRIBUTION (hard labels)")
print(SUBSEP)
print(f"  {'State':<22} {'Count':>6}  {'%':>6}  {'Bar'}")
print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*30}")
for s in STATES:
    c = primary_counts.get(s, 0)
    print(f"  {s:<22} {c:>6}  {pct(c,N):>5.1f}%  {bar(c, N)}")

# ── 2. Soft label mass per state ──────────────────────────────────────────────
soft_mass = defaultdict(float)
for r in records:
    for s in STATES:
        soft_mass[s] += r["labels"].get(s, 0)

total_mass = sum(soft_mass.values())

print(f"\n  2. SOFT LABEL MASS (sum of % weights across all {N} packets)")
print(SUBSEP)
print(f"  {'State':<22} {'Total mass':>11}  {'Mean/packet':>12}  {'% of total mass':>16}")
print(f"  {'─'*22}  {'─'*11}  {'─'*12}  {'─'*16}")
for s in STATES:
    m = soft_mass[s]
    print(f"  {s:<22} {m:>11.1f}  {m/N:>11.1f}%  {pct(m, total_mass):>15.1f}%")

# ── 3. Per-session breakdown ──────────────────────────────────────────────────
sessions = defaultdict(list)
for r in records:
    sessions[r["session_id"]].append(r)

print(f"\n  3. PER-SESSION BREAKDOWN")
print(SUBSEP)
for sid in sorted(sessions):
    recs = sessions[sid]
    uid = recs[0]["user_id"]
    counts = defaultdict(int)
    for r in recs:
        counts[r["primary_state"]] += 1
    spread = "  ".join(f"{s[:3]}={counts[s]}" for s in STATES if counts[s] > 0)
    print(f"  Session {sid:>4}  (user {uid})  [{len(recs):>3} pkts]   {spread}")

# ── 4. Per-user breakdown ─────────────────────────────────────────────────────
users = defaultdict(list)
for r in records:
    users[r["user_id"]].append(r)

print(f"\n  4. PER-USER BREAKDOWN")
print(SUBSEP)
for uid in sorted(users):
    recs = users[uid]
    counts = defaultdict(int)
    for r in recs:
        counts[r["primary_state"]] += 1
    spread = "  ".join(f"{s[:3]}={counts[s]}" for s in STATES if counts[s] > 0)
    sids   = sorted({r["session_id"] for r in recs})
    print(f"  User {uid:>4}  [{len(recs):>3} pkts, {len(sids)} session(s)]   {spread}")

# ── 5. Signal averages per primary state ─────────────────────────────────────
print(f"\n  5. SIGNAL AVERAGES PER PRIMARY STATE")
print(SUBSEP)

SIGNAL_KEYS = [
    ("drift.drift_ema",           lambda r: r["drift"]["drift_ema"]),
    ("drift.disruption_score",    lambda r: r["drift"]["disruption_score"]),
    ("drift.engagement_score",    lambda r: r["drift"]["engagement_score"]),
    ("drift.pace_ratio",          lambda r: r["drift"].get("pace_ratio") or 0.0),
    ("z_scores.z_idle",           lambda r: r["z_scores"]["z_idle"]),
    ("z_scores.z_skim",           lambda r: r["z_scores"]["z_skim"]),
    ("z_scores.z_stagnation",     lambda r: r["z_scores"]["z_stagnation"]),
    ("z_scores.z_focus_loss",     lambda r: r["z_scores"]["z_focus_loss"]),
    ("z_scores.z_regress",        lambda r: r["z_scores"]["z_regress"]),
    ("features.idle_ratio_mean",  lambda r: r["features"]["idle_ratio_mean"]),
    ("features.pace_ratio",       lambda r: r["features"].get("pace_ratio") or 0.0),
]

# Header
col_w = 16
print(f"  {'Signal':<30}", end="")
for s in STATES:
    print(f"  {s[:col_w]:<{col_w}}", end="")
print()
print(f"  {'─'*30}", end="")
for _ in STATES:
    print(f"  {'─'*col_w}", end="")
print()

# Group records by primary state
by_state = defaultdict(list)
for r in records:
    by_state[r["primary_state"]].append(r)

for label, extractor in SIGNAL_KEYS:
    print(f"  {label:<30}", end="")
    for s in STATES:
        vals = []
        for r in by_state.get(s, []):
            try:
                v = extractor(r)
                if v is not None:
                    vals.append(float(v))
            except (KeyError, TypeError):
                pass
        avg = f"{mean(vals):.3f}" if vals else "  n/a"
        print(f"  {avg:<{col_w}}", end="")
    print()

# ── 6. Soft-label entropy per state ──────────────────────────────────────────
import math

def entropy(dist):
    total = sum(dist.values())
    if total == 0:
        return 0.0
    e = 0.0
    for v in dist.values():
        p = v / total
        if p > 0:
            e -= p * math.log2(p)
    return e

print(f"\n  6. LABEL CERTAINTY (mean soft-label entropy per primary state)")
print(f"     Entropy=0 → labeller was certain; Entropy=2 → fully uncertain")
print(SUBSEP)
for s in STATES:
    recs = by_state.get(s, [])
    if not recs:
        print(f"  {s:<22}   n/a")
        continue
    entropies = [entropy(r["labels"]) for r in recs]
    print(f"  {s:<22}   mean entropy = {mean(entropies):.3f}  "
          f"(min {min(entropies):.3f}, max {max(entropies):.3f})")

# ── 7. Class imbalance warning ────────────────────────────────────────────────
print(f"\n  7. CLASS IMBALANCE CHECK")
print(SUBSEP)
max_count = max(primary_counts.values()) if primary_counts else 1
for s in STATES:
    c = primary_counts.get(s, 0)
    ratio = c / max_count if max_count > 0 else 0
    flag = ""
    if ratio < 0.2:
        flag = "  ⚠  severely under-represented (< 20% of majority class)"
    elif ratio < 0.4:
        flag = "  ⚠  under-represented (< 40% of majority class)"
    print(f"  {s:<22}  {c:>4} pkts  ratio={ratio:.2f}{flag}")

print(f"\n{SEP}\n")
