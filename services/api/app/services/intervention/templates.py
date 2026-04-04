"""
intervention/templates.py
─────────────────────────
Canned payload templates for the manual trigger endpoint.

Each entry is a realistic example matching the shape the LLM produces during
training, validated against the intervention_training_final.jsonl dataset.
Used only by POST /sessions/{id}/interventions/manual (dev/test path).
"""

from __future__ import annotations

from typing import Any

MANUAL_TEMPLATES: dict[str, dict[str, Any]] = {

    "re_engagement": {
        "headline": "Still with us?",
        "body": (
            "You've been on this section for a while. "
            "Take a breath, re-read the last line, and carry on — "
            "you're closer to understanding than you think."
        ),
        "cta": "Got it",
    },

    "focus_point": {
        "prompt": (
            "There's a key idea hidden in this paragraph. "
            "Can you spot the main argument before moving on?"
        ),
    },

    "section_summary": {
        "title":     "Quick Recap",
        "summary":   (
            "This section introduced the key concepts and framed the central "
            "argument. The author's main point is that attention is not a single "
            "faculty — it is a collection of sub-processes that can be selectively "
            "impaired."
        ),
        "key_point": (
            "Attention is composed of distinct sub-processes, each subserved by "
            "partially separable neural networks."
        ),
    },

    "comprehension_check": {
        "type":        "true_false",
        "question":    (
            "True or False: The author argues that attention is a single, "
            "unified cognitive faculty."
        ),
        "answer":      False,
        "explanation": (
            "The author explicitly states attention is composed of distinct "
            "sub-processes — alerting, orienting, and executive control."
        ),
    },

    "break_suggestion": {
        "headline": "Time for a breather",
        "body":     (
            "You've been reading for a while. A short break will help "
            "consolidate what you've read. Your session is paused — "
            "come back whenever you're ready."
        ),
        "cta_take":  "Take a break",
        "cta_skip":  "Keep reading",
        "auto_pause": True,
    },

    "gamification": {
        "event_type":  "badge_unlocked",
        "badge_id":    "first_focus_streak",
        "xp_awarded":  50,
        "message":     "Focus Streak! You've maintained attention for 3 consecutive windows.",
    },

    "chime": {
        "sound": "chime",
        "note":  "gentle attention cue",
    },

    "ambient_sound": {
        "sound":    "nature",
        "profile":  "focus",
        "note":     "background audio shifted to low-stimulation profile",
    },

    "text_reformat": {
        "mode":    "spaced",
        "note":    "Paragraph spacing increased to reduce visual density.",
        "revert_after_s": 60,
    },
}
