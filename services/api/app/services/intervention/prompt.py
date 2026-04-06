"""
intervention/prompt.py
──────────────────────
Assembles the exact ChatML-formatted raw prompt string the intervention LLM
expects.  This is a pure function module — no network, no DB, no FastAPI.

Key design decisions:
  • Uses raw=true on Ollama /api/generate, bypassing the baked-in Modelfile
    SYSTEM block so we can evolve the runtime system prompt independently of
    the GGUF.  Validated inference path: 17-19 tok/s, correct JSON output.
  • INTERVENTION_SYSTEM_PROMPT is the authoritative runtime system prompt.
    The Modelfile SYSTEM block is used only when GGUF was created.
"""

from __future__ import annotations

import json
from typing import Any

# ── System prompt ─────────────────────────────────────────────────────────────
#
# This prompt is the authoritative runtime anchor for the fine-tuned model.
# It matches the training-time instruction field in intervention_training_v2_skeletons.jsonl
# and adds:
#   - Per-type content schemas (JSON shapes) — reduces format errors
#   - High-signal decision rules — steers minority types (chime, ambient_sound)
#   - "Never copy verbatim" constraint — reinforces dataset-level quality rules
#   - active_interventions check — prevents redundant concurrent suggestions
#
# Target: ~300 tokens. Beyond ~400 tokens diminishing returns set in quickly
# as the system turn competes with the user message for model attention.
#
INTERVENTION_SYSTEM_PROMPT: str = """\
You are an adaptive reading assistant engine embedded in a digital reading tool called \
Lock-in. Every ~30 seconds you receive a window of signals about a student's attentional \
state, drift trajectory, and the text they are currently reading. Your task is to decide: \
(1) which intervention type and tier is most appropriate RIGHT NOW; \
(2) generate contextually specific content grounded in what the student is reading.

Output a single valid JSON object — no prose, no markdown fences, no code blocks:
{"intervene": true|false, "type": "<type>", "tier": "subtle|moderate|strong|none", "content": {...}|null}

Intervention types and required content shapes:
- none           : content null  — ONLY when drift_ema < 0.10 AND state is focused or hyperfocused
- chime          : {"sound": "chime", "note": "attention cue"}
- ambient_sound  : {"sound": "nature|pink_noise|brown_noise", "note": "<why>"}
- text_reformat  : {"mode": "spaced|chunked", "note": "<why>"}
- gamification   : {"event": "journey_start|milestone|xp_boost", "message": "<specific to text>"}
- focus_point    : {"headline": "<curiosity hook from text>", "body": "<1-2 sentences from text>", "cta": "Keep reading"}
- re_engagement  : {"headline": "<direct pull-back hook>", "body": "<what they're missing in text>", "cta": "Jump back in"}
- section_summary: {"title": "<descriptive>", "summary": "<synthesise — never quote verbatim>", "key_point": "<one key insight>"}
- comprehension_check: {"type": "true_false", "question": "<testable claim>", "answer": true|false, "explanation": "<why>"}
- break_suggestion: {"headline": "Take a breather", "message": "<what they read + why a break helps>", "duration_minutes": 5}

STRICT decision rules — follow exactly:
- drift_ema < 0.10 AND focused/hyperfocused   → none (genuine focus, do not disturb)
- drift_ema 0.10–0.25 AND focused              → chime or focus_point (earliest nudge)
- drift_ema 0.10–0.25 AND hyperfocused         → comprehension_check (verify encoding)
- drift_ema 0.25–0.50 AND focused/drifting     → focus_point, re_engagement, or ambient_sound
- drift_ema > 0.50 AND drifting                → re_engagement or section_summary
- drift_ema > 0.65 AND cognitive_overload      → text_reformat or section_summary
- drift_ema > 0.80 AND sustained overload      → break_suggestion
- gamification: fire when focused progress is sustained (drift_ema < 0.20, consecutive focus)
- NEVER fire the same type that is already in active_interventions
- NEVER return none if drift_ema > 0.15 — always choose the most appropriate type
- Always ground text-generative content in text_window. Never copy sentences verbatim.\
"""

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
    elapsed_minutes:         float,
    attentional_window:      list[dict[str, Any]],
    drift_progression:       dict[str, Any],
    user_baseline:           dict[str, Any],
    text_window:             list[str],
    current_paragraph_index: int | None,
    xp:                      int,
    badges_earned:           list[str],
    last_intervention:       dict[str, Any] | None,
    cooldown_status:         str,
    active_interventions:    list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build the structured JSON dict that forms the user turn of the prompt.

    Parameters
    ----------
    elapsed_minutes         : how long the session has been running
    attentional_window      : last ≤3 RF classifier outputs (newest last)
    drift_progression       : drift_level list, engagement_score list, drift_ema
    user_baseline           : wpm_effective, idle_ratio_mean, etc.
    text_window             : ≤3 paragraphs of text around current position
    current_paragraph_index : DocumentChunk.chunk_index at current position
    xp                      : accumulated XP for this session
    badges_earned           : list of badge_id strings earned this session
    last_intervention       : {type, tier, seconds_ago} or None
    cooldown_status         : "clear" or "cooling"
    active_interventions    : list of {type, tier, seconds_active} currently on
                              screen — lets the LLM avoid redundant suggestions
    """
    return {
        "session_context": {
            "elapsed_minutes":      round(elapsed_minutes, 2),
            "session_stage":        _session_stage(elapsed_minutes),
            "last_intervention":    last_intervention,
            "cooldown_status":      cooldown_status,
            "xp":                   xp,
            "badges_earned":        badges_earned,
            "active_interventions": active_interventions or [],
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

    The user message leads with the reading text explicitly so the model cannot
    miss it, then provides the structured JSON data.

    Using ``raw=true`` in the Ollama API call bypasses Ollama's template engine,
    so we supply the full ChatML string ourselves.
    """
    # Hoist text_window to the very top so the model grounds in it first
    reading_ctx = input_dict.get("reading_context", {})
    text_window: list[str] = reading_ctx.get("text_window") or []
    text_block = ""
    if text_window:
        paragraphs = "\n\n".join(text_window)
        text_block = (
            "READING TEXT (what the student is reading right now):\n"
            "---\n"
            f"{paragraphs}\n"
            "---\n\n"
        )

    data_content = json.dumps(input_dict, ensure_ascii=False, separators=(",", ":"))
    user_message = f"{text_block}SESSION DATA:\n{data_content}"

    # Seed the assistant turn with the opening token so the model cannot
    # echo the user-input JSON back.  All training outputs start with
    # {"intervene":  — seeding this forces the model onto the correct path.
    # engine.py prepends this prefix back before parsing the response.
    return (
        f"<|im_start|>system\n{INTERVENTION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        '<|im_start|>assistant\n{"intervene":'
    )
