#!/usr/bin/env python3
"""
Intervention LLM Training Dataset Builder
==========================================

Builds TrainingData/intervention_training_raw.jsonl — the Alpaca-format dataset
used to fine-tune the Qwen 2.5-7B intervention engine via QLoRA.

Every row in supervised.jsonl is a real session packet exported from the
Lock-in system (sessions 166–211, documents 182–220).  The packets carry
real timestamps, real document_id references, real z-scores, and real drift
signals.  This script reconstructs the 30-second context windows from those
packets and pairs them with the actual document paragraph text queried from
the PostgreSQL database.

Output schema (one JSON object per line, Alpaca-style):
───────────────────────────────────────────────────────
{
  "id":          str,      # traceability key, e.g. "s166_w48"
  "source":      str,      # "db_jsonl" (real session + real text)
                           # "db_jsonl_synthetic_text" (real session + fallback text)
  "instruction": str,      # fixed system prompt (identical for all examples)
  "input": {
    "session_context": {
      "elapsed_minutes":   float,   # derived from real created_at timestamps
      "session_stage":     str,     # "early" <8min | "mid" 8-25min | "late" >25min
      "last_intervention": dict | null,  # previous intervention in this session
      "cooldown_status":   str,     # "clear" | "cooling"
      "xp":                int,     # simulated session XP
      "badges_earned":     list[str]
    },
    "attentional_state_window": [   # 3 consecutive RF classifications (oldest → newest)
      {
        "primary_state": str,       # focused | drifting | hyperfocused | cognitive_overload
        "confidence":    float,     # probability of the winning class
        "distribution":  {          # full 4-class probability vector
          "focused": float, "drifting": float,
          "hyperfocused": float, "cognitive_overload": float
        }
      },
      ...  (×3)
    ],
    "drift_progression": {          # concurrent drift model outputs for the 3 packets
      "drift_level":     [float, float, float],   # 0-1, ascending = worsening
      "engagement_score":[float, float, float],   # 0-1, descending = disengaging
      "drift_ema":       float                    # exponential moving average (latest)
    },
    "user_baseline": {
      "wpm_effective":       float,  # calibrated reading speed
      "idle_ratio_mean":     float,  # typical pause fraction per window
      "regress_rate_mean":   float,  # typical backward-scroll rate
      "para_dwell_median_s": float   # typical time spent per paragraph
    },
    "reading_context": {
      "current_paragraph_index": int | null,  # estimated chunk index in document
      "text_window": [str, str, str]          # ±1 paragraphs around current position
    }
  },
  "output": {                       # to be filled by chat labelling pass
    "intervene":       null,        # true=fire now; false=cooling suppressed OR tier=none
    "tier":            str,         # tier of the generated intervention (or "none")
    "content":         null,        # ALWAYS generated when tier != "none", even if cooling
    "tier_suggestion": str,         # rule-based hint for labeller
    "type_suggestion": str,
    "pending_label":   true,
    "rationale":       str
    // Labelling rule:
    //   tier_suggestion=="none"  → intervene:false, content:null
    //   cooldown=="clear"        → intervene:true,  content:<full>
    //   cooldown=="cooling"      → intervene:false, content:<full> (suppressed but generated)
  }
}

Usage
─────
  # Default — tries DB for real text, falls back to synthetic templates:
  python TrainingData/build_intervention_dataset.py

  # Explicit DB URL:
  python TrainingData/build_intervention_dataset.py \\
      --db-url "postgresql://lockin:lockin@localhost:5433/lockin"

  # Skip DB connection (always use synthetic templates):
  python TrainingData/build_intervention_dataset.py --no-db
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPT_DIR     = Path(__file__).resolve().parent
_DEFAULT_JSONL  = _SCRIPT_DIR / "supervised.jsonl"
_DEFAULT_OUTPUT = _SCRIPT_DIR / "intervention_training_raw.jsonl"
_DEFAULT_DB_URL = "postgresql://lockin:lockin@localhost:5433/lockin"

# ── Fixed system instruction (identical at training and inference time) ────────
INSTRUCTION = (
    "You are an adaptive reading assistant engine embedded in a digital reading tool "
    "called Lock-in. Every 10 seconds you receive a 30-second window of signals about "
    "a student's attentional state, drift trajectory, and the text they are currently "
    "reading. Your task is to: (1) identify the most appropriate intervention type and "
    "tier (subtle | moderate | strong | special) based on the signals, and always "
    "generate its content; (2) set 'intervene' to true only when cooldown_status is "
    "'clear' — if cooldown_status is 'cooling', set 'intervene' to false but still "
    "output the full content of what you would have fired, so the system can schedule "
    "it for the next clear window; (3) if no intervention is warranted at all "
    "(tier='none'), set 'intervene' to false and 'content' to null. "
    "Respond with a single valid JSON object only — no prose, no markdown fences."
)

# ── Synthetic paragraph templates (fallback when DB unreachable) ───────────────
# Four genres × 8 paragraphs. Genre assignment is keyed to document type when
# known; otherwise matched to attentional state (academic → overload/focused,
# technical → hyperfocused, narrative → drifting, argumentative → mixed).
SYNTHETIC_PARAGRAPHS: dict[str, list[str]] = {
    "academic": [
        "The concept of working memory capacity has been studied extensively since "
        "Baddeley and Hitch proposed their tripartite model in 1974. Their framework "
        "posits a central executive alongside two slave systems: the phonological loop "
        "and the visuospatial sketchpad, each serving distinct representational functions.",
        "Neuroimaging evidence suggests that the dorsolateral prefrontal cortex plays a "
        "pivotal role in maintaining goal-relevant information under attentional load. "
        "Damage to this region impairs working memory capacity without affecting long-term "
        "declarative memory, suggesting functional dissociation.",
        "Cognitive load theory proposes three distinct categories of mental demand: "
        "intrinsic load (inherent task complexity), extraneous load (instructional design "
        "overhead), and germane load (schema formation). Effective reading environments "
        "minimise extraneous load while preserving germane load.",
        "The dual-process framework distinguishes between System 1 (fast, automatic, "
        "heuristic) and System 2 (slow, deliberate, analytical) cognition. Deep reading "
        "comprehension relies heavily on System 2, which is metabolically costly and "
        "susceptible to depletion over sustained sessions.",
        "Metacognitive monitoring refers to the learner's real-time assessment of their "
        "own comprehension. Skilled readers are distinguished not merely by vocabulary or "
        "background knowledge but by their capacity to detect comprehension failures and "
        "apply repair strategies before continuing.",
        "Research on spaced repetition demonstrates that distributing learning episodes "
        "across time yields superior long-term retention compared with massed practice. "
        "The spacing effect is robust across domains and material types, suggesting a "
        "fundamental property of memory consolidation.",
        "Attention is not a monolithic faculty but a collection of distinct sub-processes "
        "including alerting, orienting, and executive control. These components are "
        "subserved by partially separable neural networks, as evidenced by patient studies "
        "showing selective impairments following focal lesions.",
        "The construct of reading fluency encompasses accuracy, rate, and prosody. Fluency "
        "serves as a bridge between decoding and comprehension: automatised word recognition "
        "frees cognitive resources for higher-order inference and meaning construction.",
    ],
    "technical": [
        "A transformer architecture processes input sequences by mapping them to query, "
        "key, and value projections. The scaled dot-product attention mechanism computes "
        "a weighted sum of values where the weights represent the alignment between each "
        "query vector and all key vectors in the sequence.",
        "Gradient descent optimisation iteratively updates model parameters in the direction "
        "that minimises the loss function. The learning rate controls step size; too large "
        "a value causes divergence while too small a value leads to slow convergence or "
        "entrapment in saddle points.",
        "Isotonic calibration fits a monotonically non-decreasing step function to map raw "
        "classifier scores to calibrated probabilities. Unlike Platt scaling, isotonic "
        "regression is non-parametric and therefore more flexible in multi-class settings "
        "with irregular score distributions.",
        "A random forest aggregates predictions from an ensemble of decorrelated decision "
        "trees trained on bootstrap samples. Feature randomisation at each split reduces "
        "variance without proportional bias increase, yielding strong out-of-bag "
        "generalisation on tabular data.",
        "TimescaleDB extends PostgreSQL with native time-series capabilities by "
        "automatically partitioning hypertables along the time dimension. This enables "
        "efficient range scans over recent data while preserving full SQL compatibility "
        "for complex analytical queries across partitions.",
        "QLoRA fine-tuning quantises the base language model to 4-bit NF4 precision and "
        "injects trainable low-rank adapter matrices at each attention projection. This "
        "reduces GPU memory requirements by approximately 65% compared with full-precision "
        "fine-tuning while preserving the adapter's expressive capacity.",
        "Hyperparameter optimisation via RandomizedSearchCV evaluates a random subset of "
        "the parameter space using cross-validation, trading exhaustive coverage for "
        "computational tractability. Combined with StratifiedGroupKFold, it prevents "
        "data leakage across session boundaries in grouped datasets.",
        "The expected calibration error (ECE) measures the gap between predicted "
        "probabilities and empirical accuracy across probability bins. A well-calibrated "
        "classifier produces ECE values near zero, meaning a prediction of 70% confidence "
        "is correct approximately 70% of the time.",
    ],
    "argumentative": [
        "Critics of standardised testing argue that such assessments measure "
        "socioeconomic privilege as much as academic ability. Students from lower "
        "socioeconomic backgrounds consistently score lower, not because they lack "
        "capability but because they have had less access to test preparation resources.",
        "Proponents of open-source software contend that code transparency enables broader "
        "peer review, ultimately producing more secure and reliable systems. The Heartbleed "
        "vulnerability demonstrated, however, that openness alone does not guarantee timely "
        "detection of critical security flaws.",
        "The introduction of social media into adolescent life has been correlated with "
        "rising rates of anxiety and depression. However, establishing causality is "
        "methodologically fraught: it remains unclear whether social media causes distress "
        "or whether distressed individuals are more likely to seek connection online.",
        "Universal basic income has been proposed as a policy response to labour "
        "displacement by automation. Pilot programmes in Finland and Kenya have yielded "
        "promising results on wellbeing, though critics question whether these results "
        "generalise to larger, more heterogeneous populations.",
        "The precautionary principle holds that when an action risks harm to the public, "
        "the burden of proof falls on those taking the action rather than those potentially "
        "harmed. This principle underpins EU regulatory frameworks but is criticised by "
        "economists as a barrier to beneficial innovation.",
        "Remote work advocates argue that location-independent employment increases "
        "productivity by eliminating commutes and granting workers autonomy over their "
        "environments. Counterarguments focus on coordination costs, innovation loss, "
        "and the mental health effects of social isolation.",
    ],
    "narrative": [
        "She had always believed the library held answers, not questions. But standing "
        "before the locked cabinet in the basement, she realised every answer was a door "
        "with a different key, and she had been collecting the wrong ones for years.",
        "The morning fog settled over the harbour like a held breath. From the lighthouse "
        "window, Marcus watched the fishing boats emerge and disappear, their horns calling "
        "out to one another across the grey water as though asking questions no one was "
        "willing to answer.",
        "Three weeks after the funeral, Yemi finally opened her grandmother's sewing box. "
        "Inside, wrapped in ankara cloth, was a letter addressed to Adaeze — a name she "
        "had never heard spoken aloud in their family, though she had always felt its "
        "absence.",
        "The city had changed while he was away. The corner café where he had spent entire "
        "summers writing letters no one would read had become a chain pharmacy. Even the "
        "light seemed different — more diffuse, as though the sky had negotiated a "
        "settlement with the new glass towers.",
        "Rain drummed against the windows as she spread the documents across the kitchen "
        "table. She had known this moment was coming — had known it in the way you know "
        "a season is turning before the leaves change, before anything visible shifts.",
        "He remembered being told, as a child, that maps were not the territory. He had "
        "nodded as though he understood. Now, holding the faded survey of land that no "
        "longer existed, he finally did.",
    ],
}

_ALL_PARAGRAPHS: list[str] = [
    p for paras in SYNTHETIC_PARAGRAPHS.values() for p in paras
]

_RNG = random.Random(42)  # fixed seed for reproducibility


# ─────────────────────────────────────────────────────────────────────────────
#  Database: fetch real paragraph text and exact chunk positions
# ─────────────────────────────────────────────────────────────────────────────


async def _try_fetch_db_data(
    db_url: str,
    doc_ids: list[int],
    session_ids: list[int],
) -> tuple[dict[int, dict[int, str]], dict[int, list[tuple[datetime, int]]]]:
    """
    Open ONE database connection and fetch:
      1. document_chunks → {document_id: {chunk_index: text}}
         Text paragraphs for the LLM text_window.

      2. activity_events telemetry positions →
         {session_id: [(created_at, chunk_index), ...]} sorted by created_at.
         This gives the EXACT chunk_index the user was reading at every 2-second
         telemetry tick.  We use bisect to find the closest tick at or before
         each 10-second packet's window_end_at.

    Returns (doc_chunks, session_chunk_positions).
    Returns ({}, {}) silently if the database is unreachable.

    Why activity_events instead of a linear estimate?
    Linear estimation (packet_seq / max_seq * num_chunks) assumes constant
    reading speed and that the session starts at chunk 0.  Real reading
    involves variable pace, backtracking (negative progress_velocity), skipped
    sections, and mid-document session starts.  The activity_events table
    records the exact paragraph the user's viewport was centred on at every
    2-second telemetry tick — this is authoritative ground truth.
    """
    try:
        import asyncpg  # type: ignore
    except ImportError:
        return {}, {}

    try:
        conn = await asyncpg.connect(db_url, timeout=6)
    except Exception as e:
        print(f"  [DB] Cannot connect ({e}) — using synthetic paragraph templates.")
        return {}, {}

    try:
        # ── 1. Document chunks ───────────────────────────────────────────────
        chunk_rows = await conn.fetch(
            "SELECT document_id, chunk_index, text "
            "FROM document_chunks "
            "WHERE document_id = ANY($1) "
            "  AND text IS NOT NULL AND text != '' "
            "ORDER BY document_id, chunk_index",
            doc_ids,
        )
        doc_chunks: dict[int, dict[int, str]] = defaultdict(dict)
        for r in chunk_rows:
            doc_chunks[int(r["document_id"])][int(r["chunk_index"])] = r["text"]
        print(
            f"  [DB] Fetched {len(chunk_rows)} text chunks "
            f"for {len(doc_chunks)} documents."
        )

        # ── 2. Activity events: exact reading positions per session ──────────
        # We select the current_chunk_index from the JSONB payload of every
        # telemetry batch event.  The cast to int filters out NULL / non-integer
        # values (images and tables report chunk_index = -1 and are excluded).
        pos_rows = await conn.fetch(
            """
            SELECT
                session_id,
                created_at,
                (payload->>'current_chunk_index')::int AS chunk_index
            FROM activity_events
            WHERE session_id = ANY($1)
              AND event_type = 'telemetry_batch'
              AND (payload->>'current_chunk_index') ~ '^[0-9]+$'
            ORDER BY session_id, created_at
            """,
            session_ids,
        )
        session_positions: dict[int, list[tuple[datetime, int]]] = defaultdict(list)
        for r in pos_rows:
            chunk_idx = int(r["chunk_index"])
            if chunk_idx >= 0:  # skip -1 sentinel for images/tables
                session_positions[int(r["session_id"])].append(
                    (r["created_at"], chunk_idx)
                )

        total_pos = sum(len(v) for v in session_positions.values())
        print(
            f"  [DB] Fetched {total_pos} telemetry position ticks "
            f"for {len(session_positions)} sessions."
        )
        return dict(doc_chunks), dict(session_positions)

    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Elapsed time from real timestamps
# ─────────────────────────────────────────────────────────────────────────────


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _elapsed_minutes(packet: dict, session_start: datetime | None) -> float:
    """
    Compute elapsed minutes from session start to this packet's creation time.
    Uses real created_at timestamps.  Falls back to packet_seq * 10s if
    timestamps are unavailable.
    """
    if session_start:
        ts = _parse_ts(packet.get("created_at") or packet.get("window_end_at"))
        if ts:
            delta = (ts.replace(tzinfo=None) - session_start.replace(tzinfo=None)).total_seconds()
            return max(0.0, delta / 60.0)
    return packet.get("packet_seq", 0) * 10.0 / 60.0


def _session_stage(elapsed_min: float) -> str:
    if elapsed_min < 8:
        return "early"
    if elapsed_min < 25:
        return "mid"
    return "late"


# ─────────────────────────────────────────────────────────────────────────────
#  Paragraph text window
# ─────────────────────────────────────────────────────────────────────────────


def _lookup_chunk_index(
    session_positions: list[tuple[datetime, int]],
    window_end_at: datetime | None,
) -> int | None:
    """
    Find the DocumentChunk.chunk_index the user was reading at or just before
    ``window_end_at``, using the sorted list of (created_at, chunk_index) ticks
    retrieved from activity_events.

    Uses bisect for O(log n) lookup.  Returns None if no ticks are available
    before window_end_at.

    This replaces the discredited linear estimate
    (packet_seq / max_seq * num_chunks) which assumed constant reading speed
    and a document-start session origin — neither of which holds in practice.
    """
    import bisect

    if not session_positions or window_end_at is None:
        return None

    # Build a parallel list of naive timestamps for bisect comparison
    # (activity_events created_at may be timezone-aware; strip tz for comparison)
    def _strip_tz(dt: datetime) -> datetime:
        return dt.replace(tzinfo=None) if dt.tzinfo else dt

    target = _strip_tz(window_end_at)
    timestamps = [_strip_tz(ts) for ts, _ in session_positions]

    # Find insertion point — the index of the first tick AFTER window_end_at
    pos = bisect.bisect_right(timestamps, target)
    if pos == 0:
        return None  # all ticks are after the window end

    # The tick just before or at window_end_at
    _, chunk_idx = session_positions[pos - 1]
    return chunk_idx


def _get_text_window(
    doc_chunks: dict[int, str],
    chunk_index: int,
    window_size: int = 3,
) -> list[str]:
    """
    Return up to ``window_size`` non-empty text paragraphs centred on
    ``chunk_index`` from the pre-loaded ``doc_chunks`` dict.

    Mirrors paragraph_fetcher.text_window_from_dict — scans forward from
    (chunk_index - half) collecting only non-empty text, skipping image/table
    chunk slots (which have empty strings in the DB).  Pads with synthetic
    text only if fewer than window_size real paragraphs are found.
    """
    half = window_size // 2
    start = max(0, chunk_index - half)
    texts: list[str] = []
    idx = start
    while len(texts) < window_size and idx < start + window_size + 4:
        text = (doc_chunks.get(idx) or "").strip()
        if text:
            texts.append(text)
        idx += 1
    # Pad with synthetic text only if we genuinely ran out of document text
    while len(texts) < window_size:
        texts.append(_RNG.choice(_ALL_PARAGRAPHS))
    return texts


def _synthetic_text_window(state: str, window_size: int = 3) -> list[str]:
    """Return a genre-matched synthetic text window when no DB chunks are available."""
    genre_map = {
        "focused":            "academic",
        "drifting":           "narrative",
        "hyperfocused":       "technical",
        "cognitive_overload": "academic",
    }
    genre = genre_map.get(state, "academic")
    pool = SYNTHETIC_PARAGRAPHS[genre]
    start = _RNG.randint(0, max(0, len(pool) - window_size))
    window = pool[start : start + window_size]
    while len(window) < window_size:
        window.append(_RNG.choice(_ALL_PARAGRAPHS))
    return window


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-based tier / type suggestion
# ─────────────────────────────────────────────────────────────────────────────


def _suggest(
    window: list[dict[str, Any]],
    elapsed_min: float,
    prev: dict[str, Any] | None,
) -> dict[str, str]:
    """
    Apply the Lock-in intervention framework rules to the 3-packet window and
    return {tier, type, rationale}.  This becomes the pending label hint in the
    output field, which the GPT-4 labelling pass can validate or override.

    Cooldown is NOT mechanically enforced here — the `last_intervention` field
    in session_context already provides the LLM with everything it needs to
    respect cooldowns at inference time.  Enforcing hard cooldown in the tier
    suggestion logic would cause ~90% of training examples to be labelled "none
    (cooling)", which would starve the model of positive training signal for
    intervention generation.  Instead, ~15% of examples are explicitly
    constructed as cooldown examples by the caller.
    """
    states     = [p.get("primary_state", "focused") for p in window]
    drift_emas = [float((p.get("drift") or {}).get("drift_ema", 0.0)) for p in window]
    avg_ema    = sum(drift_emas) / len(drift_emas)
    last_ema   = drift_emas[-1]

    focused_n  = states.count("focused")
    drifting_n = states.count("drifting")
    overload_n = states.count("cognitive_overload")
    hyper_n    = states.count("hyperfocused")
    negative_n = drifting_n + overload_n

    prev_tier     = (prev or {}).get("tier", "none")
    prev_secs_ago = int((prev or {}).get("seconds_ago", 9999))
    cooldown_map  = {"subtle": 60, "moderate": 120, "strong": 300}

    # ── Sustained hyperfocus ─────────────────────────────────────────────
    # All three packets hyperfocused.  Early in session: preserve flow.
    # Mid/late session (>= 8 min): lightweight comprehension check confirms
    # the user is actually understanding, not just speed-reading in a tunnel.
    # This is a subtle, higher-intensity intervention — it does not disrupt
    # flow but adds a pedagogical safety net (confirmed by educational research
    # on monitoring comprehension during sustained reading episodes).
    if hyper_n == 3:
        if elapsed_min >= 8:
            return {"tier": "subtle", "type": "comprehension_check",
                    "rationale": (
                        f"Sustained hyperfocus for 3 consecutive packets at {elapsed_min:.1f} min. "
                        "Lightweight comprehension check verifies understanding without breaking flow."
                    )}
        return {"tier": "none", "type": "none",
                "rationale": "Hyperfocused early in session — preserve undisturbed flow state."}

    # Majority hyperfocused → do not interrupt under any circumstance
    if hyper_n >= 2:
        return {"tier": "none", "type": "none",
                "rationale": "Majority hyperfocused — interruption would disrupt flow state (Csikszentmihalyi 1990)."}

    # ── All focused — three distinct subtle interventions by session depth ──
    # Early (< 3 min): award gamification to positively reinforce a clean
    #   session start ("great start!" reward — anchors motivation early).
    # Mid-early (3–5 min): curiosity spark (focus_point) sustains engagement
    #   without disrupting the reading rhythm.
    # Established session (>= 5 min): comprehension check at the higher
    #   intensity of the subtle tier.  After several minutes of focus the user
    #   has consumed enough text for a meaningful T/F or highlight question.
    #   This is the key pedagogical scenario: retain focus AND verify learning.
    if focused_n == 3:
        if elapsed_min < 3:
            return {"tier": "subtle", "type": "gamification",
                    "rationale": (
                        f"All three packets focused, only {elapsed_min:.1f} min elapsed. "
                        "Early-session gamification reward anchors motivation."
                    )}
        if elapsed_min < 5:
            return {"tier": "subtle", "type": "focus_point",
                    "rationale": (
                        f"All three packets focused at {elapsed_min:.1f} min. "
                        "Curiosity spark sustains engagement without disruption."
                    )}
        return {"tier": "subtle", "type": "comprehension_check",
                "rationale": (
                    f"All three packets focused for {elapsed_min:.1f} min. "
                    "Higher-intensity subtle: comprehension check retains engagement and verifies understanding."
                )}

    # ── Cognitive overload — break suggestion scenarios ──────────────────
    # Scenario A: complete overload (all 3 packets) with high drift EMA.
    #   User is cognitively exhausted — a break is the only appropriate action.
    # Scenario B: sustained overload mid-session (2+ overload packets, >= 8 min
    #   elapsed, moderately elevated drift).  A break prevents compounding fatigue
    #   and is more effective than content-level interventions at this stage.
    if overload_n == 3 and avg_ema > 0.65:
        return {"tier": "strong", "type": "break_suggestion",
                "rationale": (
                    f"All three packets cognitive_overload, avg_ema={avg_ema:.2f}. "
                    "User is cognitively exhausted — break is the only appropriate response."
                )}

    if overload_n >= 2 and elapsed_min >= 8 and avg_ema > 0.55:
        return {"tier": "strong", "type": "break_suggestion",
                "rationale": (
                    f"overload_n={overload_n}, elapsed={elapsed_min:.1f} min, avg_ema={avg_ema:.2f}. "
                    "Sustained mid-session overload — break prevents compounding cognitive fatigue."
                )}

    # ── Escalation: previous moderate had no effect ───────────────────────
    if prev_tier == "moderate" and prev_secs_ago > cooldown_map["moderate"] and negative_n >= 2:
        return {"tier": "strong", "type": "re_engagement",
                "rationale": "Previous moderate intervention had no measurable effect. Escalating to strong re-engagement."}

    # ── Heavy negative signal → strong section summary ───────────────────
    if overload_n >= 2 or (negative_n == 3 and avg_ema > 0.62):
        return {"tier": "strong", "type": "section_summary",
                "rationale": (
                    f"overload_n={overload_n}, negative_n={negative_n}, avg_ema={avg_ema:.2f}. "
                    "Strong: inline section summary provides re-entry point."
                )}

    # ── All 3 drifting, moderate drift → comprehension check re-anchors ──
    # This is the moderate-tier comprehension check scenario: three consecutive
    # drifting windows suggest the user's mind has wandered from the content.
    # A T/F question on the last paragraph forces re-engagement with the text.
    if drifting_n == 3 and avg_ema > 0.50:
        return {"tier": "moderate", "type": "comprehension_check",
                "rationale": (
                    f"Three consecutive drifting packets, avg_ema={avg_ema:.2f}. "
                    "Moderate comprehension check re-anchors attention to content."
                )}

    # ── 2 negative + drift rising → moderate re-engagement ───────────────
    if negative_n >= 2 and last_ema >= avg_ema:
        return {"tier": "moderate", "type": "re_engagement",
                "rationale": (
                    f"negative_n={negative_n}, drift_ema rising ({drift_emas[0]:.2f}→{last_ema:.2f}). "
                    "Moderate re-engagement prompt before further deterioration."
                )}

    # ── 2 negative (stable or falling drift) → moderate section summary ──
    if negative_n >= 2:
        return {"tier": "moderate", "type": "section_summary",
                "rationale": "Two negative-state packets. Collapsible summary gives structured re-entry point."}

    # ── Single negative packet ────────────────────────────────────────────
    # Drifting: alternate between chime (immediate audio nudge) and focus_point
    #   (curiosity spark). Both are subtle-tier; the mix ensures the model learns
    #   both intervention options for the same trigger context.
    # Cognitive overload: alternate between ambient sound (long-term background
    #   shift) and text_reformat (visual restructuring of the dense passage).
    if negative_n == 1:
        # Use packet_seq parity (derived from avg_ema hash) to alternate types
        _parity = int(avg_ema * 1000) % 2
        if drifting_n == 1:
            if _parity == 0:
                return {"tier": "subtle", "type": "chime",
                        "rationale": "Single drifting packet. Soft chime re-anchors attention without disrupting reading rhythm."}
            return {"tier": "subtle", "type": "focus_point",
                    "rationale": "Single drifting packet in otherwise positive window. Curiosity spark re-engages without disruption."}
        # overload packet
        if _parity == 0:
            return {"tier": "moderate", "type": "text_reformat",
                    "rationale": "Single overload packet. Text reformat reduces visual density to ease cognitive load."}
        return {"tier": "subtle", "type": "ambient_sound",
                "rationale": "Single overload packet. Ambient sound shift eases cognitive load gently."}

    # ── Mostly focused (focused_n >= 2, mixed with hyperfocused) ─────────
    # Positive reinforcement via gamification when the window is predominantly
    # positive but not all-focused (e.g. 2 focused + 1 hyperfocused).
    if focused_n >= 2:
        return {"tier": "subtle", "type": "gamification",
                "rationale": "Predominantly focused window. Gamification reinforces positive engagement."}

    return {"tier": "none", "type": "none",
            "rationale": "No clear positive or negative signal — no intervention needed."}


# ─────────────────────────────────────────────────────────────────────────────
#  Build one training example
# ─────────────────────────────────────────────────────────────────────────────


def _build_example(
    ex_id: str,
    source: str,
    window: list[dict[str, Any]],      # 3 packet dicts from supervised.jsonl
    session_start: datetime | None,
    prev_intervention: dict | None,
    xp: int,
    badges: list[str],
    text_window: list[str],
    para_idx: int | None,
) -> dict[str, Any]:
    """
    Assemble one Alpaca-style training example.  The input field mirrors exactly
    what the live intervention engine passes to Qwen at inference time.
    """
    mid = window[1]  # centre packet of the 30-second window

    elapsed_min = _elapsed_minutes(mid, session_start)
    suggestion  = _suggest(window, elapsed_min, prev_intervention)

    cooldown_map  = {"subtle": 60, "moderate": 120, "strong": 300}
    prev_tier     = (prev_intervention or {}).get("tier", "none")
    prev_secs_ago = int((prev_intervention or {}).get("seconds_ago", 9999))
    cooldown_status = (
        "cooling" if prev_secs_ago < cooldown_map.get(prev_tier, 0) else "clear"
    )

    # ── attentional_state_window: pure RF output (no drift in this block) ──
    state_window = []
    for pkt in window:
        labels = pkt.get("labels") or {}
        total  = sum(labels.values()) or 100
        state  = pkt.get("primary_state") or max(labels, key=labels.get, default="focused")
        conf   = round(labels.get(state, 0) / total, 4)
        state_window.append({
            "primary_state": state,
            "confidence":    conf,
            "distribution": {
                "focused":            round(labels.get("focused",            0) / total, 4),
                "drifting":           round(labels.get("drifting",           0) / total, 4),
                "hyperfocused":       round(labels.get("hyperfocused",       0) / total, 4),
                "cognitive_overload": round(labels.get("cognitive_overload", 0) / total, 4),
            },
        })

    # ── drift_progression: separate from state (as agreed) ─────────────────
    drift_data = [pkt.get("drift") or {} for pkt in window]
    drift_progression = {
        "drift_level":      [round(float(d.get("drift_level",     0.0)), 4) for d in drift_data],
        "engagement_score": [round(float(d.get("engagement_score",0.0)), 4) for d in drift_data],
        "drift_ema":        round(float(drift_data[-1].get("drift_ema", 0.0)), 4),
    }

    # ── user_baseline: from embedded baseline_snapshot ──────────────────────
    bl_snap  = mid.get("baseline_snapshot") or {}
    bl_json  = bl_snap.get("baseline_json") or {}
    baseline = {
        "wpm_effective":       round(float(bl_json.get("wpm_effective") or bl_json.get("wpm_gross") or 0.0), 1),
        "idle_ratio_mean":     round(float(bl_json.get("idle_ratio_mean",  0.08)), 4),
        "regress_rate_mean":   round(float(bl_json.get("regress_rate_mean", 0.05)), 4),
        "para_dwell_median_s": round(float(bl_json.get("para_dwell_median_s") or 10.0), 1),
    }

    return {
        "id":     ex_id,
        "source": source,
        "instruction": INSTRUCTION,
        "input": {
            "session_context": {
                "elapsed_minutes":   round(elapsed_min, 2),
                "session_stage":     _session_stage(elapsed_min),
                "last_intervention": prev_intervention,
                "cooldown_status":   cooldown_status,
                "xp":                xp,
                "badges_earned":     badges[:],
            },
            "attentional_state_window": state_window,
            "drift_progression":        drift_progression,
            "user_baseline":            baseline,
            "reading_context": {
                "current_paragraph_index": para_idx,
                "text_window":             text_window,
            },
        },
        "output": {
            "intervene":       None,
            "tier_suggestion": suggestion["tier"],
            "type_suggestion": suggestion["type"],
            "pending_label":   True,
            "rationale":       suggestion["rationale"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


async def _build(
    jsonl_path: Path,
    output_path: Path,
    db_url: str,
    use_db: bool,
) -> None:

    # ── 1. Load supervised.jsonl ────────────────────────────────────────────
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {jsonl_path} …")
    raw: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Skipping line {i}: {e}", file=sys.stderr)

    print(f"  {len(raw)} packets loaded.")

    # ── 2. Group by session, sort by packet_seq ─────────────────────────────
    sessions: dict[int, list[dict]] = defaultdict(list)
    for pkt in raw:
        sessions[int(pkt.get("session_id", 0))].append(pkt)
    for sid in sessions:
        sessions[sid].sort(key=lambda p: p.get("packet_seq", 0))

    print(f"  {len(sessions)} sessions, {len(raw)} total packets.")

    # ── 3. Collect unique document IDs and session IDs ──────────────────────
    doc_ids = sorted(set(
        int(p.get("packet_raw", {}).get("document_id") or 0)
        for p in raw
        if p.get("packet_raw", {}).get("document_id")
    ))
    session_ids_all = sorted(sessions.keys())
    print(f"  {len(doc_ids)} unique document IDs: {doc_ids}")
    print(f"  {len(session_ids_all)} unique session IDs.")

    # ── 4. Try to fetch real paragraph text AND exact positions from DB ──────
    # Two queries in a single connection:
    #   a) document_chunks   → {doc_id: {chunk_index: text}}
    #   b) activity_events   → {session_id: [(created_at, chunk_index), ...]}
    #
    # The activity_events data replaces the discredited linear estimate
    # (packet_seq / max_seq * num_chunks) with exact ground-truth positions
    # recorded by the frontend renderer's IntersectionObserver every 2 s.
    doc_chunks: dict[int, dict[int, str]] = {}
    session_chunk_positions: dict[int, list[tuple[datetime, int]]] = {}

    if use_db and doc_ids:
        print("Querying database for document chunks and telemetry positions …")
        doc_chunks, session_chunk_positions = await _try_fetch_db_data(
            db_url, doc_ids, session_ids_all
        )

    real_text_docs = set(doc_chunks.keys())
    if real_text_docs:
        print(f"  Real text available for documents: {sorted(real_text_docs)}")
    else:
        print("  No DB chunks available — all examples will use synthetic paragraph templates.")

    sessions_with_positions = set(session_chunk_positions.keys())
    if sessions_with_positions:
        print(f"  Exact reading positions available for {len(sessions_with_positions)} sessions.")
    else:
        print("  No activity_events positions available — chunk_index will be null.")

    # ── 5. Determine per-session session_start timestamp ────────────────────
    session_starts: dict[int, datetime | None] = {}
    for sid, pkts in sessions.items():
        first_ts = _parse_ts(pkts[0].get("created_at") or pkts[0].get("window_start_at"))
        session_starts[sid] = first_ts

    # ── 6. Build sliding 3-packet windows ───────────────────────────────────
    examples: list[dict[str, Any]] = []
    real_text_count = 0
    synthetic_text_count = 0

    for sid, pkts in sessions.items():
        if len(pkts) < 3:
            continue

        doc_id        = int((pkts[0].get("packet_raw") or {}).get("document_id") or 0)
        chunks        = doc_chunks.get(doc_id, {})
        session_start = session_starts[sid]

        # ── Gamification state ───────────────────────────────────────────────
        xp                    = 0
        badges: list[str]     = []
        streak_focused        = 0   # consecutive focused windows
        streak_hyperfocused   = 0   # consecutive hyperfocused windows
        streak_clean          = 0   # consecutive non-negative windows (not drifting/overload)
        prev_state: str | None = None
        first_para_idx: int | None = None   # for page_turner badge

        # ── Cooldown simulation ──────────────────────────────────────────────
        # We track the LAST INTERVENTION THAT ACTUALLY FIRED (after cooldown
        # cleared) using a simulated clock (sim_clock, in seconds).  Only
        # cooldown-clear firings update this record; suppressed suggestions
        # (tier != none but still cooling) do not.
        #
        # Each session starts with a randomised history to prevent the entire
        # dataset from being in "cooling" state:
        #   40 % — session starts with a recent (cooling) intervention
        #   60 % — session starts with no prior intervention (clear from start)
        #
        # This gives a realistic 40-60 % clear / 40-60 % cooling split after
        # accounting for interventions fired during the session itself.
        _COOLDOWN_MAP = {"subtle": 60, "moderate": 120, "strong": 300, "special": 300}

        sim_clock = 0          # simulated seconds elapsed in this session
        last_fired: dict | None  # {type, tier, fired_at (sim_clock seconds)}

        if _RNG.random() < 0.40:
            # Start with a recent fake intervention (within cooldown period)
            _fake_tier  = _RNG.choice(["subtle", "subtle", "moderate", "strong"])
            _fake_type  = _RNG.choice(["focus_point", "re_engagement",
                                       "ambient_sound", "section_summary"])
            _fake_cd    = _COOLDOWN_MAP[_fake_tier]
            _fake_age   = _RNG.randint(10, max(11, _fake_cd - 10))
            last_fired  = {
                "type":     _fake_type,
                "tier":     _fake_tier,
                "fired_at": -_fake_age,   # negative = before session start
            }
        else:
            last_fired = None

        # Sorted (created_at, chunk_index) ticks for this session from DB
        sid_positions = session_chunk_positions.get(sid, [])

        for w in range(len(pkts) - 2):
            sim_clock += 10   # each window = 10 seconds
            window = pkts[w : w + 3]
            mid    = window[1]

            # ── Exact paragraph position from activity_events ──────────────
            # Use the window_end_at of the CENTRE packet as the lookup key.
            win_end_str = mid.get("window_end_at") or mid.get("created_at")
            win_end_dt  = _parse_ts(win_end_str)
            para_idx    = _lookup_chunk_index(sid_positions, win_end_dt)

            # ── Paragraph text window ──────────────────────────────────────
            if chunks and para_idx is not None:
                text_window = _get_text_window(chunks, para_idx)
                source = "db_jsonl"
                real_text_count += 1
            elif chunks and para_idx is None:
                fallback_idx = len(chunks) // 2
                text_window  = _get_text_window(chunks, fallback_idx)
                para_idx     = fallback_idx
                source       = "db_jsonl_approx_position"
                real_text_count += 1
            else:
                para_idx    = None
                text_window = _synthetic_text_window(mid.get("primary_state", "focused"))
                source      = "db_jsonl_synthetic_text"
                synthetic_text_count += 1

            # ── Build prev_intervention view for this window ───────────────
            # Show the last ACTUALLY-FIRED intervention with elapsed seconds.
            # Two mechanisms clear the record (giving a "clear" state):
            #   1. Clock advances past 2× the cooldown threshold (long gap).
            #   2. Random 9% chance per window — simulates natural breaks,
            #      pauses, or the user closing and reopening the session.
            #      At ~66 windows/session this creates ~6 clear transitions
            #      per session, pushing the overall clear rate to ~35-40 %.
            if last_fired is not None:
                _secs_since = sim_clock - last_fired["fired_at"]
                _cd_thresh  = _COOLDOWN_MAP.get(last_fired["tier"], 60)
                _natural_clear = _secs_since > _cd_thresh * 2
                _random_clear  = _RNG.random() < 0.09
                if _natural_clear or _random_clear:
                    last_fired = None
                    prev_intervention: dict | None = None
                else:
                    prev_intervention = {
                        "type":        last_fired["type"],
                        "tier":        last_fired["tier"],
                        "seconds_ago": _secs_since,
                    }
            else:
                prev_intervention = None

            ex = _build_example(
                ex_id=f"s{sid}_w{w}",
                source=source,
                window=window,
                session_start=session_start,
                prev_intervention=prev_intervention,
                xp=xp,
                badges=badges,
                text_window=text_window,
                para_idx=para_idx,
            )
            examples.append(ex)

            # ── Update cooldown: only record if cooldown has cleared ───────
            tier = ex["output"]["tier_suggestion"]
            if tier not in ("none",):
                _cd_thresh = _COOLDOWN_MAP.get(
                    (last_fired or {}).get("tier", "none"), 0
                )
                _cooldown_clear = (
                    last_fired is None
                    or (sim_clock - last_fired["fired_at"]) >= _cd_thresh
                )
                if _cooldown_clear:
                    last_fired = {
                        "type":     ex["output"]["type_suggestion"],
                        "tier":     tier,
                        "fired_at": sim_clock,
                    }
                # If still cooling: intervention suppressed — do NOT update
                # last_fired.  The prev_intervention clock keeps advancing.

            # ── Update gamification state ──────────────────────────────────
            last_state = window[-1].get("primary_state", "focused")
            elapsed_min = ex["input"]["session_context"]["elapsed_minutes"]

            # XP awards
            if last_state == "focused":
                xp += 10
                streak_focused      += 1
                streak_hyperfocused  = 0
                streak_clean        += 1
            elif last_state == "hyperfocused":
                xp += 15
                streak_hyperfocused += 1
                streak_focused       = 0
                streak_clean        += 1
            elif last_state in ("drifting", "cognitive_overload"):
                streak_focused      = 0
                streak_hyperfocused = 0
                streak_clean        = 0
            else:
                streak_focused      = 0
                streak_hyperfocused = 0
                streak_clean       += 1

            # Comeback bonus XP: transition from negative → focused
            if (prev_state in ("drifting", "cognitive_overload")
                    and last_state == "focused"):
                xp += 20

            # Track first paragraph index for page_turner badge
            if para_idx is not None and first_para_idx is None:
                first_para_idx = para_idx

            # ── Badge awards ───────────────────────────────────────────────
            def _award(badge: str) -> None:
                if badge not in badges:
                    badges.append(badge)

            # XP milestones
            if xp >= 100:
                _award("first_focus_streak")
            if xp >= 300:
                _award("deep_reader")
            if xp >= 500:
                _award("focus_master")

            # Session duration milestone
            if elapsed_min >= 15:
                _award("reading_marathon")

            # Hyperfocus streak
            if streak_hyperfocused >= 3:
                _award("hyperfocus_detected")

            # Comeback from drift/overload
            if (prev_state in ("drifting", "cognitive_overload")
                    and last_state == "focused"):
                _award("comeback_kid")

            # Sustained clean attention
            if streak_clean >= 10:
                _award("no_distraction_10")

            # Document progress: advanced 20+ paragraphs since first position
            if (para_idx is not None
                    and first_para_idx is not None
                    and (para_idx - first_para_idx) >= 20):
                _award("page_turner")

            prev_state = last_state

    # ── 7. Shuffle deterministically, write output ───────────────────────────
    _RNG.shuffle(examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ── 8. Stats ─────────────────────────────────────────────────────────────
    from collections import Counter
    tiers     = Counter(e["output"]["tier_suggestion"] for e in examples)
    types     = Counter(e["output"]["type_suggestion"] for e in examples)
    stages    = Counter(e["input"]["session_context"]["session_stage"] for e in examples)
    states    = Counter(
        e["input"]["attentional_state_window"][-1]["primary_state"] for e in examples
    )
    sources   = Counter(e["source"] for e in examples)
    cooldowns = Counter(e["input"]["session_context"]["cooldown_status"] for e in examples)
    all_badges_flat = [b for e in examples for b in e["input"]["session_context"]["badges_earned"]]
    badge_counts = Counter(all_badges_flat)

    print(f"\n── Dataset Summary {'─'*42}")
    print(f"  Total examples         : {len(examples)}")
    print(f"  Real text (exact pos)  : {sources.get('db_jsonl', 0)}"
          f"  ({100*sources.get('db_jsonl',0)/max(len(examples),1):.0f}%)")
    print(f"  Real text (approx pos) : {sources.get('db_jsonl_approx_position', 0)}"
          f"  ({100*sources.get('db_jsonl_approx_position',0)/max(len(examples),1):.0f}%)")
    print(f"  Synthetic text         : {sources.get('db_jsonl_synthetic_text', 0)}"
          f"  ({100*sources.get('db_jsonl_synthetic_text',0)/max(len(examples),1):.0f}%)")
    print(f"\n  Session stages:")
    for s, c in stages.most_common():
        print(f"    {s:<10} {c:>5}  ({100*c/max(len(examples),1):.1f}%)")
    print(f"\n  Cooldown status:")
    for s, c in cooldowns.most_common():
        print(f"    {s:<10} {c:>5}  ({100*c/max(len(examples),1):.1f}%)")
    print(f"\n  Window-end attentional states:")
    for s, c in states.most_common():
        print(f"    {s:<25} {c:>5}  ({100*c/max(len(examples),1):.1f}%)")
    print(f"\n  Tier suggestions:")
    for t, c in tiers.most_common():
        print(f"    {t:<20} {c:>5}  ({100*c/max(len(examples),1):.1f}%)")
    print(f"\n  Type suggestions:")
    for t, c in types.most_common():
        print(f"    {t:<25} {c:>5}  ({100*c/max(len(examples),1):.1f}%)")
    print(f"\n  Badge distribution (unique badge appearances across all examples):")
    if badge_counts:
        for b, c in badge_counts.most_common():
            print(f"    {b:<25} {c:>5}  ({100*c/max(len(examples),1):.1f}%)")
    else:
        print("    (no badges awarded)")
    no_badge = sum(1 for e in examples if not e["input"]["session_context"]["badges_earned"])
    print(f"\n  Examples with no badges: {no_badge} ({100*no_badge/max(len(examples),1):.1f}%)")
    print(f"{'─'*62}\n")

    print(f"✓  Wrote {len(examples)} examples → {output_path}")
    print(
        "\nNext step: run TrainingData/label_interventions_gpt4.py to fill in "
        "output.intervene and output.content using GPT-4, then fine-tune "
        "Qwen 2.5-7B with QLoRA on the completed dataset."
    )

    # ── 9. Verify one example looks correct ──────────────────────────────────
    print("\n── Sample example (first in file) ──────────────────────────────")
    sample = examples[0]
    print(f"  id:           {sample['id']}")
    print(f"  source:       {sample['source']}")
    sc = sample["input"]["session_context"]
    print(f"  elapsed_min:  {sc['elapsed_minutes']}  stage={sc['session_stage']}")
    print(f"  xp/badges:    {sc['xp']} / {sc['badges_earned']}")
    aw = sample["input"]["attentional_state_window"]
    print(f"  state window: {[p['primary_state'] for p in aw]}")
    dp = sample["input"]["drift_progression"]
    print(f"  drift_emas:   {dp['drift_level']}  → latest_ema={dp['drift_ema']}")
    bl = sample["input"]["user_baseline"]
    print(f"  baseline wpm: {bl['wpm_effective']}")
    rc = sample["input"]["reading_context"]
    print(f"  para_idx:     {rc['current_paragraph_index']}")
    print(f"  text[0]:      {rc['text_window'][0][:80]}…")
    out = sample["output"]
    print(f"  tier/type:    {out['tier_suggestion']} / {out['type_suggestion']}")
    print(f"  rationale:    {out['rationale'][:100]}…")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the intervention LLM training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db-url", default=_DEFAULT_DB_URL,
                        help="PostgreSQL connection URL for document_chunks lookup.")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip database connection; use synthetic text throughout.")
    parser.add_argument("--jsonl", default=str(_DEFAULT_JSONL),
                        help="Path to supervised.jsonl.")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT),
                        help="Output JSONL path.")
    args = parser.parse_args()

    asyncio.run(_build(
        jsonl_path=Path(args.jsonl),
        output_path=Path(args.output),
        db_url=args.db_url,
        use_db=not args.no_db,
    ))


if __name__ == "__main__":
    main()
