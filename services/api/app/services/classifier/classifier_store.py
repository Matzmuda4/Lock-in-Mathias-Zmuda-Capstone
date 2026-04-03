"""
Persistence layer for attentional-state classifier output.

Writes one row to session_attentional_states (TimescaleDB hypertable) on
every full-window classification.  The row is self-contained: it includes
the full intervention_context JSONB so the intervention LLM can read a
sliding window of history with a single query and no joins.

Responsibility split (SOLID — Single Responsibility):
  classifier_store.py  → persistence (this file)
  rf_classifier.py     → inference
  feature_extractor.py → feature vector construction
  registry.py          → singleton + in-memory cache

The intervention LLM query pattern expected by this schema:

  SELECT *
  FROM session_attentional_states
  WHERE session_id = :sid
  ORDER BY created_at DESC
  LIMIT 5;

  → Returns the last 5 classifications in newest-first order, each row
    carrying a complete intervention_context ready for prompt assembly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SessionAttentionalState
from app.services.classifier.base import ClassificationResult


async def save_attentional_state(
    session_id: int,
    packet_seq: int,
    result: ClassificationResult,
    packet_json: dict[str, Any],
    db: AsyncSession,
) -> SessionAttentionalState:
    """
    Persist one classifier result to session_attentional_states.

    Extends ClassificationResult.as_intervention_context() with the concurrent
    drift state (drift_level, drift_ema) from the packet_json so the
    intervention LLM has a single JSONB document containing everything it needs:

      intervention_context = {
        "primary_state":  str,
        "confidence":     float,          # max(distribution)
        "distribution":   {state: float}, # all 4 probabilities
        "ambiguous":      bool,           # top-2 gap < 0.15
        "drift_level":    float,          # concurrent drift model output
        "drift_ema":      float,
        "packet_seq":     int,
        "session_id":     int,
      }

    Parameters
    ----------
    session_id   : owning session
    packet_seq   : monotonic packet counter for this session
    result       : ClassificationResult from the RF classifier
    packet_json  : raw packet as stored by store._build_packet_json()
                   used to extract the concurrent drift snapshot
    db           : active async SQLAlchemy session (caller must commit)

    Returns
    -------
    The newly added SessionAttentionalState ORM row (unflushed until caller commits).
    """
    dr: dict[str, Any] = packet_json.get("drift") or {}
    drift_level = float(dr.get("drift_level", 0.0))
    drift_ema   = float(dr.get("drift_ema",   0.0))

    # Extend the base intervention context with drift state and identifiers
    ctx = result.as_intervention_context()
    ctx["drift_level"] = drift_level
    ctx["drift_ema"]   = drift_ema
    ctx["packet_seq"]  = packet_seq
    ctx["session_id"]  = session_id

    row = SessionAttentionalState(
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        packet_seq=packet_seq,
        primary_state=result.primary_state,
        confidence=round(max(
            result.focused, result.drifting,
            result.hyperfocused, result.cognitive_overload,
        ), 6),
        prob_focused=round(result.focused, 6),
        prob_drifting=round(result.drifting, 6),
        prob_hyperfocused=round(result.hyperfocused, 6),
        prob_cognitive_overload=round(result.cognitive_overload, 6),
        drift_level=drift_level,
        drift_ema=drift_ema,
        intervention_context=ctx,
        latency_ms=result.latency_ms,
        parse_ok=result.parse_ok,
    )
    db.add(row)
    return row
