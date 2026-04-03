"""
Packet formatter — converts the internal _build_packet_json dict
into the exact input structure the LLM was trained on.

The training data's extract_inputs() function (Colab notebook) produced:
  {
    "meta":              { user_id, session_id, packet_seq, window_start_at,
                           window_end_at, session_mode, document_id, drift_included },
    "baseline_snapshot": { baseline_json, baseline_updated_at, baseline_valid },
    "features":          { ...WindowFeatures fields... },
    "z_scores":          { ...ZScores fields... },
    "ui_aggregates":     { ...ui context fractions... },
    "drift":             { drift_ema, disruption_score, engagement_score, ... },
  }

This module maps the store.py packet_json + PacketWrittenInfo to that structure
so inference is identical to training.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def format_for_llm(
    packet_json: dict[str, Any],
    packet_seq: int,
    window_start_at: datetime | None,
    window_end_at: datetime | None,
) -> dict[str, Any]:
    """
    Build the LLM user-message dict from the raw packet stored in DB.

    Parameters
    ----------
    packet_json      : dict from _build_packet_json() in store.py
    packet_seq       : monotonic packet counter for this session
    window_start_at  : start of the 30-second window (window_end_at - 30s)
    window_end_at    : timestamp of the newest batch in the window

    Returns
    -------
    dict ready to be JSON-serialised as the "user" message to the LLM.
    """
    return {
        "meta": {
            "user_id":         packet_json.get("user_id"),
            "session_id":      packet_json.get("session_id"),
            "packet_seq":      packet_seq,
            "window_start_at": window_start_at.isoformat() if window_start_at else None,
            "window_end_at":   window_end_at.isoformat()   if window_end_at   else None,
            "session_mode":    packet_json.get("session_mode"),
            "document_id":     packet_json.get("document_id"),
            # drift is always available in live inference
            "drift_included":  True,
        },
        # Baseline snapshot exactly as stored (contains baseline_json, baseline_valid, etc.)
        "baseline_snapshot": packet_json.get("baseline_snapshot"),
        # WindowFeatures dataclass dict
        "features":          packet_json.get("features"),
        # ZScores dataclass dict
        "z_scores":          packet_json.get("z_scores"),
        # UI context fractions over the 30-second window
        "ui_aggregates":     packet_json.get("ui_aggregates"),
        # Drift model secondary hints
        "drift":             packet_json.get("drift"),
    }
