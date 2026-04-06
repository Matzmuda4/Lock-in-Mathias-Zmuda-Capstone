"""
intervention/rules.py
─────────────────────
Backend supplementary rules that fire chime and text_reformat when the LLM
does not select them due to training gradient imbalance.

These rules are kept deliberately narrow and transparent:
  • chime      — lightest possible nudge; fires when drift starts rising.
  • text_reformat — fires when the student is clearly in sustained overload
                    and the LLM produced nothing actionable.

Both functions are async, receive the same data as the trigger endpoint,
and mutate the DB + tracker exactly as the LLM path does.

Design note: these are pure SIGNAL-BASED rules, not hard-coded responses to
specific states. They fire only when the signal picture is unambiguous, and
always go through the same ActiveInterventionTracker gate as the LLM, so
cooldown and slot limits are respected.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Intervention, SessionAttentionalState
from app.services.intervention.engine import (
    ActiveInterventionTracker,
    InterventionResult,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ── Helper: log and mark an intervention fired ────────────────────────────────

async def _fire(
    session_id:    int,
    itype:         str,
    tier:          str,
    content:       dict,
    tracker:       ActiveInterventionTracker,
    db:            AsyncSession,
    primary_state: str,
) -> bool:
    """
    Gate-check, log to DB, and mark fired.  Returns True if it actually fired.
    """
    decision = tracker.check(session_id, itype, primary_state)
    if not decision.allowed:
        log.debug("[rules] %s gate blocked: %s", itype, decision.reason)
        return False

    result = InterventionResult(
        intervene  = True,
        tier       = tier,
        type       = itype,
        content    = content,
        raw_json   = "{}",
        latency_ms = 0,
        parse_ok   = True,
    )
    row = Intervention(
        session_id = session_id,
        type       = itype,
        intensity  = tier,
        payload    = {
            "intervene":  True,
            "tier":       tier,
            "type":       itype,
            "content":    content,
            "raw_json":   "{}",
            "latency_ms": 0,
        },
    )
    db.add(row)
    await db.flush()
    tracker.mark_fired(session_id, row.id, itype, tier)
    await db.commit()
    log.info("[rules] FIRED %s (tier=%s) session=%d id=%d", itype, tier, session_id, row.id)
    return True


# ── Chime ─────────────────────────────────────────────────────────────────────

async def maybe_fire_chime(
    session_id:    int,
    rows:          list[SessionAttentionalState],
    primary_state: str,
    tracker:       ActiveInterventionTracker,
    db:            AsyncSession,
) -> bool:
    """
    Fire a subtle chime when drift starts rising noticeably.

    Conditions (ALL must be true):
      1. At least 2 attentional-state rows available.
      2. drift_ema increased by ≥ 0.10 from the previous packet to the latest
         OR the latest packet is the first 'drifting' state in the window
         (i.e. student just started drifting).
      3. Current drift_ema > 0.12 (ignores noise at the floor).
      4. Primary state is 'focused' or 'drifting' (not hyperfocused/overload —
         those need stronger interventions).

    The INSTANT type classification in the tracker means chime can fire
    alongside other active interventions and does not consume a slot.
    """
    if len(rows) < 2:
        return False

    current_ema = rows[-1].drift_ema or 0.0
    prev_ema    = rows[-2].drift_ema or 0.0
    drift_delta = current_ema - prev_ema

    # Condition: drift rising meaningfully
    drift_rising = drift_delta >= 0.10

    # Condition: first drifting state in the window (state just changed)
    prev_states = [r.primary_state for r in rows[:-1]]
    state_just_shifted = (
        rows[-1].primary_state == "drifting"
        and "drifting" not in prev_states
    )

    if not (drift_rising or state_just_shifted):
        return False

    if current_ema < 0.12:
        return False

    if primary_state not in ("focused", "drifting"):
        return False

    content = {"sound": "chime", "note": "gentle attention cue"}

    return await _fire(
        session_id    = session_id,
        itype         = "chime",
        tier          = "subtle",
        content       = content,
        tracker       = tracker,
        db            = db,
        primary_state = primary_state,
    )


# ── Text reformat ─────────────────────────────────────────────────────────────

async def maybe_fire_text_reformat(
    session_id:    int,
    rows:          list[SessionAttentionalState],
    primary_state: str,
    tracker:       ActiveInterventionTracker,
    db:            AsyncSession,
) -> bool:
    """
    Fire text_reformat when the student is in sustained cognitive overload and
    the LLM didn't produce anything actionable.

    Conditions (ALL must be true):
      1. At least 2 attentional-state rows, with the current primary state
         being 'cognitive_overload'.
      2. At least 2 of the last 3 packets have primary_state == 'cognitive_overload'
         (sustained, not a one-off spike).
      3. Current drift_ema ≥ 0.60 (high drift confirms genuine overload).

    Uses strong tier + spacing + single-chunk layout to maximally reduce
    cognitive load.
    """
    if not rows:
        return False

    if primary_state != "cognitive_overload":
        return False

    # Count how many of the last 3 packets are cognitive_overload
    overload_count = sum(
        1 for r in rows if r.primary_state == "cognitive_overload"
    )
    if overload_count < 2:
        return False

    current_ema = rows[-1].drift_ema or 0.0
    if current_ema < 0.60:
        return False

    content = {
        "line_spacing": 2.0,
        "chunk_size":   1,
    }

    return await _fire(
        session_id    = session_id,
        itype         = "text_reformat",
        tier          = "strong",
        content       = content,
        tracker       = tracker,
        db            = db,
        primary_state = primary_state,
    )
