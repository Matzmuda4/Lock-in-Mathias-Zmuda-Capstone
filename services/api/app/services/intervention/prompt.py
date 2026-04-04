"""
intervention/prompt.py
──────────────────────
Assembles the exact ChatML-formatted raw prompt string the intervention LLM
expects.  This is a pure function module — no network, no DB, no FastAPI.

The prompt format mirrors the training data produced by
  TrainingData/build_intervention_dataset.py
and the validated inference call confirmed at 17-19 tok/s on Apple M3 Metal.

Key design decisions (Single Responsibility / Open-Closed):
  • This module owns ONLY prompt construction.
  • All DB queries live in the router layer.
  • Changing the prompt structure never touches the engine or router.
"""

from __future__ import annotations

import json
from typing import Any

# ── System prompt ─────────────────────────────────────────────────────────────
# Must match the Modelfile SYSTEM block exactly (same wording as training).
INTERVENTION_SYSTEM_PROMPT: str = (
    "You are an adaptive reading assistant engine embedded in a digital reading "
    "tool called Lock-in. Every 10 seconds you receive a 30-second window of "
    "signals about a student's attentional state, drift trajectory, and the text "
    "they are currently reading. Your task is to: "
    "(1) identify the most appropriate intervention type and tier "
    "(subtle | moderate | strong | special) based on the signals, and always "
    "generate its content; "
    "(2) set 'intervene' to true only when cooldown_status is 'clear' — "
    "if cooldown_status is 'cooling', set 'intervene' to false but still output "
    "the full content of what you would have fired, so the system can schedule it "
    "for the next clear window; "
    "(3) if no intervention is warranted at all (tier='none'), set 'intervene' to "
    "false and 'content' to null. "
    "Respond with a single valid JSON object only — no prose, no markdown fences."
)

# Cooldown enforced between intervention fires (seconds).
COOLDOWN_SECONDS: int = 90

# Tier ordering for session-stage mapping.
_SESSION_STAGE_MINUTES: tuple[tuple[float, str], ...] = (
    (5.0,  "early"),
    (12.0, "mid"),
    (float("inf"), "late"),
)


def _session_stage(elapsed_minutes: float) -> str:
    for threshold, label in _SESSION_STAGE_MINUTES:
        if elapsed_minutes < threshold:
            return label
    return "late"


def build_intervention_input(
    *,
    elapsed_minutes: float,
    attentional_window: list[dict[str, Any]],
    drift_progression: dict[str, Any],
    user_baseline: dict[str, Any],
    text_window: list[str],
    current_paragraph_index: int | None,
    xp: int,
    badges_earned: list[str],
    last_intervention: dict[str, Any] | None,
    cooldown_status: str,
) -> dict[str, Any]:
    """
    Build the structured JSON dict that forms the user turn of the prompt.

    Parameters
    ----------
    elapsed_minutes        : how long the session has been running
    attentional_window     : last ≤3 RF classifier outputs (newest last)
    drift_progression      : drift_level list, engagement_score list, drift_ema
    user_baseline          : wpm_effective, idle_ratio_mean, etc.
    text_window            : ≤3 paragraphs of text around current position
    current_paragraph_index: DocumentChunk.chunk_index at current position
    xp                     : accumulated XP for this session
    badges_earned          : list of badge_id strings earned this session
    last_intervention      : {type, tier, seconds_ago} or None
    cooldown_status        : "clear" or "cooling"
    """
    return {
        "session_context": {
            "elapsed_minutes": round(elapsed_minutes, 2),
            "session_stage":   _session_stage(elapsed_minutes),
            "last_intervention": last_intervention,
            "cooldown_status":   cooldown_status,
            "xp":                xp,
            "badges_earned":     badges_earned,
        },
        "attentional_state_window": attentional_window,
        "drift_progression":        drift_progression,
        "user_baseline":            user_baseline,
        "reading_context": {
            "current_paragraph_index": current_paragraph_index,
            "text_window":             text_window,
        },
    }


def build_raw_chatml_prompt(input_dict: dict[str, Any]) -> str:
    """
    Wrap the structured input dict in the ChatML tokens the model was trained on.

    Using ``raw=true`` in the Ollama API call bypasses Ollama's template engine,
    so we must supply the full ChatML string ourselves.  This was validated as the
    correct inference path (17-19 tok/s, correct JSON output).
    """
    user_content = json.dumps(input_dict, ensure_ascii=False, separators=(",", ":"))
    return (
        f"<|im_start|>system\n{INTERVENTION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
