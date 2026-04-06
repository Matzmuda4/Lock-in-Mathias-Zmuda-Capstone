"""
build_v2_skeletons.py
─────────────────────
Produces intervention_training_v2_skeletons.jsonl:

  • Reads intervention_training_raw.jsonl
  • Quality-filters rows (real text, logically coherent state↔type pairs)
  • Deduplicates: keeps the best 1 row per (text_hash, type_suggestion)
  • Caps at TARGET_PER_TYPE rows per type (prioritising text diversity)
  • Adds fully pre-labelled synthetic rows for:
      chime, text_reformat, ambient_sound (system-driven — no prose to write)
  • Outputs two files:
      → intervention_training_v2_skeletons.jsonl   (all rows, ready for labelling)
      → intervention_training_v2_prelabelled.jsonl (system-driven rows already done)

Each skeleton row contains:
  id, instruction, input (full signal block), output (type/tier/rationale + pending_label)

The labeller only needs to fill: intervene, tier, content
for TEXT-GENERATIVE types: focus_point, section_summary, comprehension_check,
                            re_engagement, break_suggestion, gamification, none
"""

import json
import hashlib
import random
import uuid
import collections
from pathlib import Path
from typing import Optional

random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────

RAW_PATH   = Path("TrainingData/intervention_training_raw.jsonl")
OUT_SKEL   = Path("TrainingData/intervention_training_v2_skeletons.jsonl")
OUT_PRELBL = Path("TrainingData/intervention_training_v2_prelabelled.jsonl")

# Types that require ChatGPT prose labelling
TEXT_GEN_TYPES = {
    "focus_point", "section_summary", "comprehension_check",
    "re_engagement", "break_suggestion", "gamification",
}

# System-driven types — content is fully deterministic, no labelling needed.
# 'none' and 'ambient_sound' are also included: none=null content, ambient=track+fade only.
SYSTEM_TYPES = {"chime", "ambient_sound", "text_reformat", "none"}

TARGET_PER_TYPE = 80   # aim for 80 examples per type

# Minimum text quality bar
MIN_TEXT_CHARS  = 80
MAX_SHORT_RATIO = 0.40  # if >40% of words are ≤2 chars → OCR garbage


# ─── Helpers ──────────────────────────────────────────────────────────────────

def text_hash(tw: list[str]) -> str:
    return hashlib.md5(tw[0][:300].encode()).hexdigest() if tw else ""


def is_clean_text(tw: list[str]) -> bool:
    if not tw or len(tw[0]) < MIN_TEXT_CHARS:
        return False
    words = tw[0].split()
    short = sum(1 for w in words if len(w) <= 2)
    return short / max(len(words), 1) <= MAX_SHORT_RATIO


def is_section_header(text: str) -> bool:
    """Detect bare section-header text like '2.2 Comparison' or 'Text .'"""
    t = text.strip()
    if len(t) < 50:
        return True
    if t.lower().startswith("text ."):
        return True
    import re
    return bool(re.match(r"^\d+(\.\d+)*\s+\w", t))


def last_state(row: dict) -> str:
    window = row.get("input", {}).get("attentional_state_window", [])
    return window[-1].get("primary_state", "") if window else ""


def drift_ema(row: dict) -> float:
    return row.get("input", {}).get("drift_progression", {}).get("drift_ema", 0.0)


def state_pattern(row: dict) -> tuple:
    window = row.get("input", {}).get("attentional_state_window", [])
    return tuple(w.get("primary_state", "?") for w in window)


def type_suggestion(row: dict) -> str:
    return row.get("output", {}).get("type_suggestion", "")


def tier_suggestion(row: dict) -> str:
    return row.get("output", {}).get("tier_suggestion", "")


def is_logically_coherent(row: dict) -> bool:
    """Drop rows where the type↔state pairing is clearly wrong."""
    ts = type_suggestion(row)
    ps = last_state(row)
    ema = drift_ema(row)
    if ts == "break_suggestion" and ps == "focused" and ema < 0.30:
        return False
    if ts == "section_summary" and ps == "focused" and ema < 0.12:
        return False
    if ts == "gamification" and ps == "cognitive_overload":
        return False
    if ts == "comprehension_check" and ps == "cognitive_overload" and ema > 0.65:
        return False
    return True


def quality_score(row: dict) -> float:
    """Higher = prefer this row when deduplicating."""
    tw = row.get("input", {}).get("reading_context", {}).get("text_window", [])
    text_len = len(tw[0]) if tw else 0
    n_packets = len(row.get("input", {}).get("attentional_state_window", []))
    ema = drift_ema(row)
    # Prefer longer text, 3 packets, non-trivial drift
    return text_len * 0.001 + n_packets * 0.5 + min(ema, 0.9) * 0.3


SYSTEM_INSTRUCTION = (
    "You are an adaptive reading assistant engine embedded in a digital reading tool called Lock-in. "
    "Every 10 seconds you receive a 30-second window of signals about a student's attentional state, "
    "drift trajectory, and the text they are currently reading. Your task is to decide:\n"
    "  (1) whether to intervene and which type + tier is most appropriate;\n"
    "  (2) generate the exact content for that intervention based on what the student is reading.\n\n"
    "Output a single JSON object with exactly these fields:\n"
    "  intervene  : true | false\n"
    "  type       : one of [focus_point, section_summary, comprehension_check, re_engagement,\n"
    "               ambient_sound, chime, text_reformat, break_suggestion, gamification, none]\n"
    "  tier       : subtle | moderate | strong | special | none\n"
    "  content    : object whose shape depends on type (see schema)\n\n"
    "Always ground text-generative content in the text_window. Never copy sentences verbatim."
)


# ─── Step 1: Load and filter raw rows ─────────────────────────────────────────

print("Loading raw data …")
raw_rows: list[dict] = []
with open(RAW_PATH) as f:
    for line in f:
        try:
            raw_rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

print(f"  Loaded {len(raw_rows)} rows")

clean: list[dict] = []
drop_reasons = collections.Counter()

for row in raw_rows:
    tw = row.get("input", {}).get("reading_context", {}).get("text_window", [])

    if not is_clean_text(tw):
        drop_reasons["bad_text"] += 1
        continue
    if tw and is_section_header(tw[0]):
        drop_reasons["section_header"] += 1
        continue
    if not is_logically_coherent(row):
        drop_reasons["wrong_pairing"] += 1
        continue
    if type_suggestion(row) in ("", "MISSING"):
        drop_reasons["no_type"] += 1
        continue

    clean.append(row)

print(f"  After quality filter: {len(clean)} rows")
print(f"  Dropped: {dict(drop_reasons)}")


# ─── Step 2: Deduplicate — 1 best row per (text_hash, type) ───────────────────

print("\nDeduplicating (text_hash × type) …")
best: dict[tuple, dict] = {}

for row in clean:
    tw = row.get("input", {}).get("reading_context", {}).get("text_window", [])
    key = (text_hash(tw), type_suggestion(row))
    if key not in best or quality_score(row) > quality_score(best[key]):
        best[key] = row

deduped = list(best.values())
print(f"  Unique (text, type) pairs: {len(deduped)}")

by_type: dict[str, list[dict]] = collections.defaultdict(list)
for row in deduped:
    by_type[type_suggestion(row)].append(row)

for t, rows in sorted(by_type.items()):
    print(f"    {t:30s} {len(rows)}")


# ─── Step 3: Cap per type, maximise text diversity ────────────────────────────

print(f"\nSelecting from raw data, maximising text diversity …")
# chime and text_reformat have zero real examples — they stay fully synthetic.
# ambient_sound (52 real) and none (42 real) are included from raw data;
# their content is assigned programmatically and they are pre-labelled.
SKIP_FROM_RAW: set[str] = set()   # nothing skipped now
selected: list[dict] = []

for itype, rows in by_type.items():
    rows.sort(key=quality_score, reverse=True)
    cap = TARGET_PER_TYPE  # synthetics top up the rest

    seen_hashes: set[str] = set()
    picked: list[dict] = []
    for row in rows:
        tw = row.get("input", {}).get("reading_context", {}).get("text_window", [])
        h = text_hash(tw)
        if h not in seen_hashes:
            seen_hashes.add(h)
            picked.append(row)
        if len(picked) >= cap:
            break

    print(f"  {itype:30s} selected {len(picked)}/{len(rows)} available")
    selected.extend(picked)

print(f"\nTotal selected from raw: {len(selected)}")


# ─── Step 4: Build unique-text pool for synthetic generation ──────────────────

all_text_pool: list[dict] = []
seen_tw: set[str] = set()
for row in deduped:
    tw = row.get("input", {}).get("reading_context", {}).get("text_window", [])
    h = text_hash(tw)
    if h not in seen_tw:
        seen_tw.add(h)
        all_text_pool.append({
            "text_window": tw,
            "user_baseline": row.get("input", {}).get("user_baseline", {}),
        })

random.shuffle(all_text_pool)
print(f"Text pool for synthetics: {len(all_text_pool)} unique windows")


# ─── Step 5: Synthetic rows for system-driven types ───────────────────────────
#
# chime       — fires on any early-drift or transition moment
# ambient_sound — drifting / overload with 2+ negative packets
# text_reformat — cognitive_overload sustained ≥2 packets, higher drift
#
# Content is fully pre-decided (no labelling needed).

def make_drift_window(ema: float, primary: str, n: int = 3) -> list[dict]:
    """Create a plausible 3-packet attentional state window for given state/drift."""
    states_map = {
        "focused":           {"focused": 0.80, "drifting": 0.12, "hyperfocused": 0.04, "cognitive_overload": 0.04},
        "drifting":          {"focused": 0.15, "drifting": 0.72, "hyperfocused": 0.02, "cognitive_overload": 0.11},
        "cognitive_overload":{"focused": 0.08, "drifting": 0.20, "hyperfocused": 0.00, "cognitive_overload": 0.72},
        "hyperfocused":      {"focused": 0.10, "drifting": 0.03, "hyperfocused": 0.84, "cognitive_overload": 0.03},
    }
    dist = states_map.get(primary, states_map["focused"])
    packets = []
    for i in range(n):
        conf = round(random.uniform(0.62, 0.91), 2)
        d = {k: round(v + random.uniform(-0.05, 0.05), 3) for k, v in dist.items()}
        total = sum(d.values())
        d = {k: round(v / total, 3) for k, v in d.items()}
        packets.append({
            "primary_state": primary,
            "confidence": conf,
            "distribution": d,
        })
    return packets


def make_drift_prog(ema: float) -> dict:
    base = max(0.0, ema - 0.1)
    levels = [round(base + i * 0.03 + random.uniform(0, 0.02), 4) for i in range(3)]
    eng = [round(max(0.1, 1.0 - l), 2) for l in levels]
    return {
        "drift_level": levels,
        "engagement_score": eng,
        "drift_ema": round(ema, 4),
    }


def make_elapsed(stage: str) -> float:
    stages = {"early": (0.5, 3.0), "mid": (3.0, 9.0), "late": (9.0, 15.0)}
    lo, hi = stages.get(stage, (1.0, 8.0))
    return round(random.uniform(lo, hi), 1)


def make_session_context(stage: str, elapsed: float, itype: str, tier: str) -> dict:
    return {
        "elapsed_minutes": elapsed,
        "session_stage": stage,
        "last_intervention": {
            "type": "none",
            "tier": "none",
            "seconds_ago": random.randint(30, 120),
        },
        "cooldown_status": "clear",
    }


def baseline() -> dict:
    return {
        "wpm_effective": round(random.uniform(160, 280), 1),
        "idle_ratio_mean": round(random.uniform(0.15, 0.45), 2),
        "regress_rate_mean": round(random.uniform(0.0, 0.15), 2),
        "para_dwell_median_s": round(random.uniform(8, 20), 1),
    }


def build_synthetic_row(itype: str, tier: str, primary_state: str,
                        ema: float, stage: str, text_entry: dict,
                        content: dict, rationale: str) -> dict:
    elapsed = make_elapsed(stage)
    window = make_drift_window(ema, primary_state)
    dp = make_drift_prog(ema)
    sc = make_session_context(stage, elapsed, itype, tier)

    row_input = {
        "session_context": sc,
        "attentional_state_window": window,
        "drift_progression": dp,
        "user_baseline": text_entry.get("user_baseline") or baseline(),
        "reading_context": {
            "current_paragraph_index": random.randint(1, 20),
            "text_window": text_entry["text_window"],
        },
    }

    row_output = {
        "intervene": True,
        "type": itype,
        "tier": tier,
        "content": content,
        "rationale": rationale,
        "pending_label": False,
    }

    return {
        "id": f"syn_{itype}_{uuid.uuid4().hex[:8]}",
        "source": "synthetic_system_driven",
        "instruction": SYSTEM_INSTRUCTION,
        "input": row_input,
        "output": row_output,
    }


# ─── Chime scenarios (10 scenarios × 8 = 80) ──────────────────────────────────
CHIME_SCENARIOS = [
    ("focused",           (0.10, 0.22), "subtle",
     {"sound": "gentle_bell", "message": "Stay with it."},
     "Drift creeping from low base. Gentle chime anchors before slide begins."),
    ("drifting",          (0.20, 0.38), "moderate",
     {"sound": "gentle_bell", "message": "Refocus."},
     "Single drifting packet with rising EMA. Chime is lightest redirect."),
    ("drifting",          (0.35, 0.55), "moderate",
     {"sound": "double_tap",  "message": "Come back."},
     "Two drifting packets. Double-tap escalates signal without interrupting."),
    ("cognitive_overload",(0.28, 0.52), "moderate",
     {"sound": "gentle_bell", "message": "Take a breath."},
     "Overload onset. Soft chime signals pause before heavier intervention."),
    ("focused",           (0.05, 0.16), "subtle",
     {"sound": "gentle_bell", "message": "Good — keep going."},
     "Low drift, sustained focus. Positive-reinforcement chime."),
    ("drifting",          (0.42, 0.66), "strong",
     {"sound": "double_tap",  "message": "Eyes back on the text."},
     "Sustained drifting + high EMA. Fast signal before text intervention fires."),
    ("focused",           (0.18, 0.30), "moderate",
     {"sound": "gentle_bell", "message": "Almost there."},
     "Focused with mid-session drift rising. Encouraging chime before moderate prompt."),
    ("drifting",          (0.55, 0.75), "strong",
     {"sound": "double_tap",  "message": "Re-anchor now."},
     "High-drift drifting 3 consecutive packets. Urgent double-tap."),
    ("hyperfocused",      (0.02, 0.12), "subtle",
     {"sound": "gentle_bell", "message": "You're in the zone."},
     "Hyperfocused, very low drift. Subtle affirmation chime — no disruption."),
    ("cognitive_overload",(0.55, 0.80), "strong",
     {"sound": "double_tap",  "message": "Slow down."},
     "Sustained overload + high EMA. Double-tap before break suggestion."),
]

# ─── Ambient sound scenarios (10 scenarios × 8 = 80) ─────────────────────────
AMBIENT_SCENARIOS = [
    ("drifting",          (0.20, 0.38), "moderate",
     {"track": "nature",      "fade_in_seconds": 5},
     "Drifting moderate EMA. Nature sounds reduce ambient distraction."),
    ("drifting",          (0.30, 0.52), "moderate",
     {"track": "pink_noise",  "fade_in_seconds": 4},
     "Drifting 2 packets. Pink noise masks environmental distractors."),
    ("cognitive_overload",(0.25, 0.48), "moderate",
     {"track": "brown_noise", "fade_in_seconds": 6},
     "Overload detected. Brown noise lowers arousal to workable level."),
    ("cognitive_overload",(0.42, 0.70), "strong",
     {"track": "nature",      "fade_in_seconds": 3},
     "Sustained overload. Sound overlay provides attentional anchor."),
    ("focused",           (0.08, 0.18), "subtle",
     {"track": "pink_noise",  "fade_in_seconds": 8},
     "Focused, drift creeping. Proactive soundscape cements focus."),
    ("drifting",          (0.50, 0.75), "strong",
     {"track": "brown_noise", "fade_in_seconds": 3},
     "High-drift drifting. Brown noise — strongest masking for re-entry."),
    ("drifting",          (0.18, 0.35), "subtle",
     {"track": "nature",      "fade_in_seconds": 7},
     "Early drift signal, focused→drifting transition. Gentle nature soundscape."),
    ("cognitive_overload",(0.60, 0.85), "strong",
     {"track": "brown_noise", "fade_in_seconds": 3},
     "Extreme overload. Brown noise + fast fade as calming anchor."),
    ("drifting",          (0.35, 0.58), "moderate",
     {"track": "pink_noise",  "fade_in_seconds": 5},
     "Drifting mid-session. Pink noise consistent masking layer."),
    ("focused",           (0.12, 0.25), "moderate",
     {"track": "nature",      "fade_in_seconds": 6},
     "Focused but drift climbing. Proactive moderate ambient sound."),
]

# ─── Text reformat scenarios (10 scenarios × 8 = 80) ─────────────────────────
REFORMAT_SCENARIOS = [
    ("cognitive_overload",(0.38, 0.60), "moderate",
     {"line_spacing": 1.6, "chunk_size": 3},
     "Overload 2 packets. Increased spacing reduces visual crowding."),
    ("cognitive_overload",(0.55, 0.78), "strong",
     {"line_spacing": 1.8, "chunk_size": 2},
     "Sustained high-EMA overload. 2-paragraph chunking lowers load."),
    ("cognitive_overload",(0.65, 0.90), "strong",
     {"line_spacing": 2.0, "chunk_size": 1},
     "Extreme overload. Single-paragraph + max spacing = minimal demand."),
    ("drifting",          (0.44, 0.65), "strong",
     {"line_spacing": 1.6, "chunk_size": 3},
     "High-drift drifting. Reformat reduces density to ease re-entry."),
    ("cognitive_overload",(0.32, 0.52), "moderate",
     {"line_spacing": 1.5, "chunk_size": 4},
     "Early overload signal. Mild reformat pre-empts cognitive deterioration."),
    ("cognitive_overload",(0.70, 0.92), "strong",
     {"line_spacing": 2.0, "chunk_size": 2},
     "Severe overload 3 consecutive packets. Maximum spacing with small chunks."),
    ("drifting",          (0.52, 0.72), "strong",
     {"line_spacing": 1.7, "chunk_size": 2},
     "Drifting + overload mix. Reformat + chunking lowers cognitive barrier."),
    ("cognitive_overload",(0.45, 0.68), "strong",
     {"line_spacing": 1.8, "chunk_size": 3},
     "Overload persisting 3 packets. Large spacing + moderate chunking."),
    ("drifting",          (0.38, 0.55), "moderate",
     {"line_spacing": 1.5, "chunk_size": 4},
     "Drifting + rising EMA. Gentle reformat before escalating."),
    ("cognitive_overload",(0.30, 0.48), "moderate",
     {"line_spacing": 1.4, "chunk_size": 5},
     "Early overload, mild EMA. Minimal reformat as preventive measure."),
]

# ─── None scenarios (10 scenarios × 8 = 80) ──────────────────────────────────
NONE_SCENARIOS = [
    ("focused",           (0.00, 0.08), "none",  None,
     "All three packets focused, drift near zero. No intervention warranted."),
    ("hyperfocused",      (0.00, 0.05), "none",  None,
     "Sustained hyperfocus. System holds back — any intervention would disrupt flow."),
    ("hyperfocused",      (0.02, 0.10), "none",  None,
     "Hyperfocused + very low drift. Leave the reader alone."),
    ("focused",           (0.05, 0.12), "none",  None,
     "Focused, early session, drift minimal. No reason to intervene yet."),
    ("focused",           (0.08, 0.15), "none",  None,
     "Focused mid-session, drift low and stable. Session on track."),
    ("hyperfocused",      (0.00, 0.06), "none",  None,
     "Deep hyperfocus, very early session. Hold all interventions."),
    ("focused",           (0.02, 0.10), "none",  None,
     "Late session, focused, drift flat. Reader finishing strong — no interruption."),
    ("hyperfocused",      (0.03, 0.12), "none",  None,
     "Hyperfocused + low drift throughout window. Optimal state — no action."),
    ("focused",           (0.00, 0.07), "none",  None,
     "All three packets focused, engagement high. Intervention would be noise."),
    ("focused",           (0.06, 0.14), "none",  None,
     "Focused with slight upward drift but still well below threshold. Wait."),
]


def expand_scenario(scenarios, itype, text_pool, n_per_scenario):
    synthetics = []
    pool_idx = 0
    for scenario in scenarios:
        state, ema_range, tier, content, rationale = scenario
        for _ in range(n_per_scenario):
            ema = round(random.uniform(*ema_range), 4)
            stage = random.choice(["early", "mid", "mid", "late"])
            text_entry = text_pool[pool_idx % len(text_pool)]
            pool_idx += 1
            is_pending = itype in TEXT_GEN_TYPES  # none/ambient/chime/reformat are pre-labelled
            row = build_synthetic_row(itype, tier, state, ema, stage, text_entry,
                                      content if content is not None else None, rationale)
            if is_pending:
                # Mark as pending so ChatGPT labels it
                row["output"]["pending_label"] = True
                row["output"]["content"] = None
                row["output"]["intervene"] = None
                row["source"] = "synthetic_context"
            synthetics.append(row)
    return synthetics


# ─── Text-generative synthetic gap rows ───────────────────────────────────────
# For types where raw data falls short of TARGET_PER_TYPE, we generate synthetic
# context rows (real text + new state pattern). ChatGPT writes the content.

TEXT_GEN_SYNTHETIC_SCENARIOS: dict[str, list] = {
    "focus_point": [
        ("focused",           (0.10, 0.22), "subtle",
         "All focused, low drift. Curiosity spark to sustain engagement."),
        ("drifting",          (0.15, 0.35), "moderate",
         "Transitioning from focused to drifting. Focus point to re-anchor."),
        ("focused",           (0.05, 0.18), "subtle",
         "Hyperfocus cooling. Early focus point to maintain interest."),
        ("hyperfocused",      (0.02, 0.10), "subtle",
         "Hyperfocused; subtle curiosity spark reinforces deep engagement."),
    ],
    "re_engagement": [
        ("drifting",          (0.25, 0.50), "moderate",
         "2+ drifting packets, moderate EMA. Re-engagement before further slide."),
        ("drifting",          (0.45, 0.70), "strong",
         "3 drifting packets, high EMA. Direct re-engagement prompt needed."),
        ("focused",           (0.35, 0.65), "moderate",
         "Focused state but very high drift — unusual combination. Re-engage."),
    ],
    "gamification": [
        ("focused",           (0.05, 0.15), "subtle",
         "Early session, all focused. Journey launch anchors motivation."),
        ("focused",           (0.10, 0.22), "subtle",
         "Mid-session sustained focus. Milestone gamification reward."),
        ("drifting",          (0.20, 0.40), "moderate",
         "Drifting but recoverable. XP boost as positive reinforcement."),
        ("focused",           (0.00, 0.10), "subtle",
         "Consistent hyperfocused/focused window. Badge milestone trigger."),
    ],
    "break_suggestion": [
        ("cognitive_overload",(0.60, 0.85), "strong",
         "3 overload packets + very high EMA. Break strongly recommended."),
        ("drifting",          (0.70, 0.95), "strong",
         "3 drifting packets, extreme drift EMA. Break as reset."),
        ("cognitive_overload",(0.75, 0.95), "strong",
         "Extreme overload late session. Break before cognitive shutdown."),
        ("drifting",          (0.60, 0.82), "strong",
         "Late session sustained drifting. Break to restore capacity."),
        ("cognitive_overload",(0.55, 0.78), "strong",
         "Heavy overload mid-session. 5-minute break to consolidate."),
        ("drifting",          (0.50, 0.72), "strong",
         "Drifting throughout window + high EMA. Break resets attention."),
    ],
    "section_summary": [
        ("cognitive_overload",(0.30, 0.55), "moderate",
         "Overload 2 packets, moderate EMA. Summary provides re-entry point."),
        ("cognitive_overload",(0.55, 0.80), "strong",
         "Sustained high-EMA overload. Summary digests dense section."),
    ],
    "comprehension_check": [
        ("focused",           (0.05, 0.15), "subtle",
         "Sustained focus 9+ min. Higher-intensity subtle check to verify understanding."),
        ("hyperfocused",      (0.00, 0.08), "special",
         "Sustained hyperfocus 10+ min. Comprehension check to confirm depth."),
    ],
}


def build_text_gen_synthetics(itype: str, target_count: int,
                               existing_rows: list[dict],
                               text_pool: list[dict]) -> list[dict]:
    """Generate pending-label synthetic rows for text-generative types.
    Avoids any text window already used by existing rows of the same type."""
    existing_count = len(existing_rows)
    needed = target_count - existing_count
    if needed <= 0:
        return []
    scenarios = TEXT_GEN_SYNTHETIC_SCENARIOS.get(itype, [])
    if not scenarios:
        return []

    # Build a set of text hashes already occupied by this type
    used_hashes: set[str] = set()
    for row in existing_rows:
        tw = row.get("input", {}).get("reading_context", {}).get("text_window", [])
        used_hashes.add(text_hash(tw))

    # Filter pool to only unused texts
    free_pool = [e for e in text_pool if text_hash(e["text_window"]) not in used_hashes]
    if len(free_pool) < needed:
        free_pool = text_pool  # fallback — pool exhausted

    synthetics = []
    pool_idx = 0
    per_scenario = max(1, needed // len(scenarios) + 1)

    for (state, ema_range, tier, rationale) in scenarios:
        for _ in range(per_scenario):
            if len(synthetics) >= needed:
                break
            ema = round(random.uniform(*ema_range), 4)
            stage = random.choice(["early", "mid", "mid", "late"])
            text_entry = free_pool[pool_idx % len(free_pool)]
            pool_idx += 1
            row = build_synthetic_row(itype, tier, state, ema, stage, text_entry, None, rationale)
            row["output"]["pending_label"] = True
            row["output"]["content"] = None
            row["output"]["intervene"] = None
            row["source"] = "synthetic_context"
            synthetics.append(row)
        if len(synthetics) >= needed:
            break

    return synthetics[:needed]


# ─── Content assignment: derive (content, tier, rationale) from signals ──────
# All system-driven types — content is deterministic, no prose needed.

def assign_ambient_content(row: dict) -> tuple[dict, str, str]:
    window = row.get("input", {}).get("attentional_state_window", [])
    ps = window[-1].get("primary_state", "drifting") if window else "drifting"
    e = row.get("input", {}).get("drift_progression", {}).get("drift_ema", 0.3)
    if ps == "cognitive_overload" or e > 0.55:
        return {"track": "brown_noise", "fade_in_seconds": 3}, "strong", \
               f"Overload/high-drift (ema={e:.3f}). Brown noise lowers arousal."
    elif ps == "drifting" and e >= 0.30:
        return {"track": "pink_noise",  "fade_in_seconds": 4}, "moderate", \
               f"Drifting mid-drift (ema={e:.3f}). Pink noise masks distractors."
    elif ps == "drifting":
        return {"track": "nature",      "fade_in_seconds": 6}, "moderate", \
               f"Early drift (ema={e:.3f}). Nature soundscape gentle re-anchor."
    elif ps == "focused" and e >= 0.15:
        return {"track": "pink_noise",  "fade_in_seconds": 7}, "subtle", \
               f"Focused but drift creeping (ema={e:.3f}). Proactive pink noise."
    else:
        return {"track": "nature",      "fade_in_seconds": 8}, "subtle", \
               f"Focused, low drift (ema={e:.3f}). Gentle nature soundscape."


def assign_none_content(row: dict) -> tuple[None, str, str]:
    window = row.get("input", {}).get("attentional_state_window", [])
    ps = window[-1].get("primary_state", "focused") if window else "focused"
    e = row.get("input", {}).get("drift_progression", {}).get("drift_ema", 0.0)
    rationale = (f"State={ps}, ema={e:.3f}. " +
                 ("Hyperfocus sustained — any intervention would disrupt flow."
                  if ps == "hyperfocused" else "Focused and on track — no intervention warranted."))
    return None, "none", rationale


def assign_chime_content(row: dict) -> tuple[dict, str, str]:
    window = row.get("input", {}).get("attentional_state_window", [])
    ps = window[-1].get("primary_state", "drifting") if window else "drifting"
    e = row.get("input", {}).get("drift_progression", {}).get("drift_ema", 0.3)
    if ps == "hyperfocused":
        return {"sound": "gentle_bell", "message": "You're in the zone."}, "subtle", \
               f"Hyperfocused (ema={e:.3f}). Affirmation chime — no disruption."
    elif ps == "focused" and e < 0.18:
        return {"sound": "gentle_bell", "message": "Good — keep going."}, "subtle", \
               f"Focused, low drift (ema={e:.3f}). Positive reinforcement chime."
    elif ps == "focused" and e < 0.35:
        return {"sound": "gentle_bell", "message": "Stay with it."}, "subtle", \
               f"Focused, drift creeping (ema={e:.3f}). Gentle chime before slide."
    elif ps == "focused" and e < 0.55:
        return {"sound": "gentle_bell", "message": "Almost there."}, "moderate", \
               f"Focused, rising drift (ema={e:.3f}). Encouraging chime before prompt."
    elif ps == "drifting" and e < 0.35:
        return {"sound": "gentle_bell", "message": "Refocus."}, "moderate", \
               f"Drifting, low EMA (ema={e:.3f}). Chime is lightest redirect."
    elif ps == "drifting" and e < 0.55:
        return {"sound": "double_tap",  "message": "Come back."}, "moderate", \
               f"Drifting, moderate EMA (ema={e:.3f}). Double-tap escalates signal."
    elif ps == "drifting":
        return {"sound": "double_tap",  "message": "Eyes back on the text."}, "strong", \
               f"Sustained drifting, high EMA (ema={e:.3f}). Urgent double-tap chime."
    elif ps == "cognitive_overload" and e < 0.45:
        return {"sound": "gentle_bell", "message": "Take a breath."}, "moderate", \
               f"Overload onset (ema={e:.3f}). Soft chime before heavier intervention."
    else:
        return {"sound": "double_tap",  "message": "Slow down."}, "strong", \
               f"Sustained overload, high EMA (ema={e:.3f}). Double-tap before break."


def assign_reformat_content(row: dict) -> tuple[dict, str, str]:
    window = row.get("input", {}).get("attentional_state_window", [])
    ps = window[-1].get("primary_state", "cognitive_overload") if window else "cognitive_overload"
    e = row.get("input", {}).get("drift_progression", {}).get("drift_ema", 0.5)
    if ps == "cognitive_overload" and e < 0.45:
        return {"line_spacing": 1.5, "chunk_size": 4}, "moderate", \
               f"Early overload (ema={e:.3f}). Mild reformat pre-empts deterioration."
    elif ps == "cognitive_overload" and e < 0.60:
        return {"line_spacing": 1.6, "chunk_size": 3}, "moderate", \
               f"Overload moderate EMA (ema={e:.3f}). Spacing reduces visual crowding."
    elif ps == "cognitive_overload" and e < 0.75:
        return {"line_spacing": 1.8, "chunk_size": 2}, "strong", \
               f"High-EMA overload (ema={e:.3f}). 2-paragraph chunking lowers load."
    elif ps == "cognitive_overload":
        return {"line_spacing": 2.0, "chunk_size": 1}, "strong", \
               f"Extreme overload (ema={e:.3f}). Max spacing + single-paragraph chunks."
    elif ps == "drifting" and e < 0.60:
        return {"line_spacing": 1.5, "chunk_size": 4}, "moderate", \
               f"Drifting high EMA (ema={e:.3f}). Mild reformat reduces text density."
    else:
        return {"line_spacing": 1.7, "chunk_size": 2}, "strong", \
               f"High-drift drifting (ema={e:.3f}). Reformat + chunking lowers barrier."


# ─── Helper: pick best real rows for a type from the full quality-filtered pool ──

def pick_real_rows_for_type(all_clean: list[dict], eligible_fn,
                             assign_fn, itype: str,
                             target: int, exclude_hashes: set[str]) -> list[dict]:
    """
    From all quality-filtered raw rows, find ones eligible for `itype`,
    deduplicate by text_hash (best quality score per unique text),
    exclude hashes already used in the dataset, and return up to `target` rows
    with content pre-assigned and source marked as 'db_jsonl_reused'.
    """
    eligible = [r for r in all_clean if eligible_fn(r)]

    # Deduplicate by text hash — keep best quality per text
    best: dict[str, dict] = {}
    for row in eligible:
        tw = row.get("input", {}).get("reading_context", {}).get("text_window", [])
        h = text_hash(tw)
        if h in exclude_hashes:
            continue
        if h not in best or quality_score(row) > quality_score(best[h]):
            best[h] = row

    # Sort by quality, take up to target
    candidates = sorted(best.values(), key=quality_score, reverse=True)[:target]

    # Build normalised pre-labelled rows
    result = []
    for row in candidates:
        content, tier, rationale = assign_fn(row)
        out_row = {
            "id": f"{itype}_{row.get('id', uuid.uuid4().hex[:8])}",
            "source": "db_jsonl_reused",
            "instruction": SYSTEM_INSTRUCTION,
            "input": row.get("input", {}),
            "output": {
                "intervene": True,
                "type": itype,
                "tier": tier,
                "content": content,
                "rationale": rationale,
                "pending_label": False,
            },
        }
        result.append(out_row)
    return result


# ─── Calculate and generate all synthetics ────────────────────────────────────
print(f"\nGenerating synthetics (target {TARGET_PER_TYPE}/type) …")

# Use all text pool for synthetics — maximise variety
syst_pool = list(all_text_pool)
random.shuffle(syst_pool)

# ─── Build chime and text_reformat from REAL signal blocks ────────────────────
#
# We repurpose real session signal blocks that have appropriate state/drift
# profiles for chime and text_reformat. The original type_suggestion of those
# rows doesn't matter — what matters is that the signals (RF output, drift,
# last_intervention history, transitions) are genuine session data.
# Content is assigned deterministically based on state + drift.

# For chime/text_reformat, the only constraint is no same-(text,type) duplicate.
# Since neither type appears in raw data, the excluded set is empty — we are free
# to reuse text windows that appear under other types. The model benefits from
# seeing the same paragraph paired with different interventions based on state.
CHIME_EXCLUDED: set[str] = set()    # no prior chime rows to avoid
REFORMAT_EXCLUDED: set[str] = set() # no prior text_reformat rows to avoid

def chime_eligible(row: dict) -> bool:
    ps, e = last_state(row), drift_ema(row)
    return (
        (ps == "focused"           and 0.08 <= e <= 0.55) or
        (ps == "drifting"          and 0.10 <= e <= 0.75) or
        (ps == "cognitive_overload" and 0.20 <= e <= 0.60) or
        (ps == "hyperfocused"      and e <= 0.15)
    )

def reformat_eligible(row: dict) -> bool:
    ps, e = last_state(row), drift_ema(row)
    return (
        (ps == "cognitive_overload" and e >= 0.30) or
        (ps == "drifting"           and e >= 0.45)
    )

print("  Building chime/text_reformat from real signal blocks …")

chime_real    = pick_real_rows_for_type(clean, chime_eligible,    assign_chime_content,
                                        "chime",         TARGET_PER_TYPE, CHIME_EXCLUDED)
reformat_real = pick_real_rows_for_type(clean, reformat_eligible, assign_reformat_content,
                                        "text_reformat", TARGET_PER_TYPE, REFORMAT_EXCLUDED)

print(f"    chime from real:        {len(chime_real)}/{TARGET_PER_TYPE}")
print(f"    text_reformat from real: {len(reformat_real)}/{TARGET_PER_TYPE}")

# Track used hashes from real chime/reformat to avoid synthetic collision
used_chime_hashes   = {text_hash(r["input"]["reading_context"]["text_window"]) for r in chime_real}
used_reform_hashes  = {text_hash(r["input"]["reading_context"]["text_window"]) for r in reformat_real}

# Only generate synthetic rows for the remaining gap (usually 0 if enough real data)
chime_gap   = max(0, TARGET_PER_TYPE - len(chime_real))
reform_gap  = max(0, TARGET_PER_TYPE - len(reformat_real))

# ─── Ambient sound and none: same pattern (real first, synthetic fills gap) ───

real_system_counts: dict[str, int] = collections.Counter(
    r.get("output", {}).get("type_suggestion", "")
    for r in selected
    if r.get("output", {}).get("type_suggestion", "") in {"ambient_sound", "none"}
)
print(f"  Real rows for ambient_sound/none: {dict(real_system_counts)}")

def n_per(existing: int, scenarios: list) -> int:
    g = max(0, TARGET_PER_TYPE - existing)
    return max(1, (g + len(scenarios) - 1) // len(scenarios))

def gap_n(existing: int) -> int:
    return max(0, TARGET_PER_TYPE - existing)

used_ambient_hashes = {
    text_hash(r.get("input", {}).get("reading_context", {}).get("text_window", [""]))
    for r in selected if r.get("output", {}).get("type_suggestion") == "ambient_sound"
}
used_none_hashes = {
    text_hash(r.get("input", {}).get("reading_context", {}).get("text_window", [""]))
    for r in selected if r.get("output", {}).get("type_suggestion") == "none"
}

ambient_pool = [e for e in syst_pool if text_hash(e["text_window"]) not in used_ambient_hashes] or syst_pool
none_pool    = [e for e in syst_pool if text_hash(e["text_window"]) not in used_none_hashes]    or syst_pool

n_amb  = real_system_counts.get("ambient_sound", 0)
n_none = real_system_counts.get("none", 0)

# Synthetic fill for chime and text_reformat (only if real data fell short)
chime_pool  = [e for e in syst_pool if text_hash(e["text_window"]) not in used_chime_hashes]  or syst_pool
reform_pool = [e for e in syst_pool if text_hash(e["text_window"]) not in used_reform_hashes] or syst_pool

def expand_gap(scenarios, itype, pool, gap_count):
    if gap_count <= 0:
        return []
    n = max(1, (gap_count + len(scenarios) - 1) // len(scenarios))
    return expand_scenario(scenarios, itype, pool, n)[:gap_count]

chime_syn    = expand_gap(CHIME_SCENARIOS,    "chime",         chime_pool,  chime_gap)
ambient_syn  = expand_scenario(AMBIENT_SCENARIOS,  "ambient_sound", ambient_pool,
                               n_per(n_amb, AMBIENT_SCENARIOS))[:gap_n(n_amb)]
reformat_syn = expand_gap(REFORMAT_SCENARIOS, "text_reformat", reform_pool, reform_gap)
none_syn     = expand_scenario(NONE_SCENARIOS,     "none",          none_pool,
                               n_per(n_none, NONE_SCENARIOS))[:gap_n(n_none)]

print(f"  Synthetic fill — chime: {len(chime_syn)}, ambient: {len(ambient_syn)}, "
      f"reformat: {len(reformat_syn)}, none: {len(none_syn)}")


# ─── Step 7: Normalise existing rows to output schema ─────────────────────────
#
# Text-gen types: pending_label=True (ChatGPT writes the content)
# System-driven types from raw (ambient_sound, none): pre-labelled with assigned content
# System-driven types without real data (chime, text_reformat): fully synthetic

def normalise_raw_row(row: dict) -> dict:
    out = row.get("output", {})
    itype = out.get("type_suggestion", "")

    # System-driven types from real data: assign content now, mark pre-labelled
    if itype == "ambient_sound":
        content, tier, rationale = assign_ambient_content(row)
        new_out = {
            "intervene": True,
            "type": itype,
            "tier": tier,
            "content": content,
            "rationale": rationale,
            "pending_label": False,
        }
    elif itype == "none":
        content, tier, rationale = assign_none_content(row)
        new_out = {
            "intervene": False,
            "type": itype,
            "tier": tier,
            "content": content,
            "rationale": rationale,
            "pending_label": False,
        }
    else:
        # Text-generative types: pending for ChatGPT labelling
        new_out = {
            "intervene": None,
            "type": itype,
            "tier": out.get("tier_suggestion", ""),
            "content": None,
            "rationale": out.get("rationale", ""),
            "pending_label": True,
        }

    return {
        "id": row.get("id", str(uuid.uuid4())),
        "source": row.get("source", "db_jsonl"),
        "instruction": SYSTEM_INSTRUCTION,
        "input": row.get("input", {}),
        "output": new_out,
    }


normalised = [normalise_raw_row(r) for r in selected]

# Text-generative shortfall synthetics (pending — need ChatGPT labelling)
# Must be built AFTER normalised so we can dedup against it
selected_by_type: dict[str, list[dict]] = collections.defaultdict(list)
for row in normalised:
    selected_by_type[row["output"]["type"]].append(row)

text_gen_syn: list[dict] = []
for itype in TEXT_GEN_TYPES:
    existing_rows = selected_by_type.get(itype, [])
    gap_rows = build_text_gen_synthetics(itype, TARGET_PER_TYPE, existing_rows, syst_pool)
    if gap_rows:
        print(f"  {itype}: +{len(gap_rows)} synthetic context rows (pending label)")
    text_gen_syn.extend(gap_rows)


# ─── Step 7: Merge and write ───────────────────────────────────────────────────

all_rows = (normalised
            + chime_real    + chime_syn
            + reformat_real + reformat_syn
            + ambient_syn
            + none_syn
            + text_gen_syn)
random.shuffle(all_rows)

# Assign clean sequential IDs
for i, row in enumerate(all_rows):
    if not row.get("id"):
        row["id"] = f"v2_{i:04d}"

print(f"\nTotal dataset size: {len(all_rows)}")

type_dist = collections.Counter(r["output"]["type"] for r in all_rows)
print("Final type distribution:")
for t, n in sorted(type_dist.items()):
    pending = sum(1 for r in all_rows if r["output"]["type"] == t and r["output"].get("pending_label"))
    done    = n - pending
    print(f"  {t:28s} {n:4d}  (needs labelling: {pending:3d} | pre-labelled: {done:3d})")

# Write all skeletons (both pending and pre-labelled)
with open(OUT_SKEL, "w") as f:
    for row in all_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"\nWrote {OUT_SKEL}")

# Write only the pre-labelled (system-driven) rows separately for reference
prelabelled = [r for r in all_rows if not r["output"].get("pending_label")]
with open(OUT_PRELBL, "w") as f:
    for row in prelabelled:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Wrote {OUT_PRELBL} ({len(prelabelled)} pre-labelled rows)")

needs_labelling = [r for r in all_rows if r["output"].get("pending_label")]
pre_labelled    = [r for r in all_rows if not r["output"].get("pending_label")]
print(f"\nRows that need ChatGPT labelling: {len(needs_labelling)}")
print(f"Rows already pre-labelled:        {len(pre_labelled)}")
lbl_by_type = collections.Counter(r["output"]["type"] for r in needs_labelling)
print("  Pending by type:")
for t, n in sorted(lbl_by_type.items()):
    print(f"    {t:28s} {n}")
