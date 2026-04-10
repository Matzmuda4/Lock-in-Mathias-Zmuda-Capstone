"""
intervention/engine.py
──────────────────────
Calls the locally-running ``lockin-intervention`` Ollama model, parses its
JSON response, and tracks per-session active intervention state.

Single Responsibility:
  This module owns ONLY Ollama I/O + response parsing + active state tracking.
  Prompt construction lives in prompt.py.
  DB writes + session ownership checks live in the router.

Active intervention model (agreed design):
  ─────────────────────────────────────────
  Intervention types are divided into five categories:

  PASSIVE          gamification, ambient_sound
                   → Fire freely; no slot constraint; run in the background.

  INSTANT          chime
                   → Fires and is immediately done; no UI slot consumed;
                     subject only to the minimum 60-second gap.

  TEXT_PROMPT      re_engagement, focus_point, section_summary,
                   comprehension_check
                   → Visible text cards on screen; max 2 simultaneously.

  NON_TEXT_ACTIVE  text_reformat
                   → Modifies the page visually; counts toward the 3-slot
                     foreground cap but not toward the text-prompt cap.

  NUCLEAR          break_suggestion
                   → Auto-pauses the session; while active, nothing else fires.
                     After the user resumes, a 5-minute post-break cooldown
                     prevents any further fires.

  Slot rules (applied in this order):
    1. If break_suggestion is active → nothing fires (except it can be
       acknowledged to end the break).
    2. If type == break_suggestion AND time since last break_resume < 5 min
       → cannot fire.
    3. PASSIVE types fire freely (no slot check).
    4. INSTANT types fire if minimum gap (60 s) has passed.
    5. If this type is already active → skip (no duplicate types).
    6. TEXT_PROMPT types: text_active < 2 AND foreground_active < 3.
    7. NON_TEXT_ACTIVE / NUCLEAR: foreground_active < 3.
    8. Minimum gap: time since last fire >= 60 s.

  Auto-dismiss: text prompts that have been on screen unacknowledged for
  > 60 s are silently removed before each gate check so they do not block
  new interventions permanently.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ── Intervention categories ───────────────────────────────────────────────────

PASSIVE_TYPES: frozenset[str] = frozenset({
    "gamification",
    "ambient_sound",
})

INSTANT_TYPES: frozenset[str] = frozenset({
    "chime",
})

TEXT_PROMPT_TYPES: frozenset[str] = frozenset({
    "re_engagement",
    "focus_point",
    "section_summary",
    "comprehension_check",
})

NON_TEXT_ACTIVE_TYPES: frozenset[str] = frozenset({
    "text_reformat",
})

NUCLEAR_TYPES: frozenset[str] = frozenset({
    "break_suggestion",
})

# All valid types (union of all categories)
VALID_TYPES: frozenset[str] = (
    PASSIVE_TYPES | INSTANT_TYPES | TEXT_PROMPT_TYPES |
    NON_TEXT_ACTIVE_TYPES | NUCLEAR_TYPES
)

VALID_TIERS: frozenset[str] = frozenset({
    "none", "subtle", "moderate", "strong", "special",
})

# Slot limits
MAX_TEXT_PROMPTS:      int = 2
MAX_FOREGROUND_ACTIVE: int = 3

# Default timing constants (overridden by config)
# Min-gap is kept very small (10 s) just to prevent burst-firing on the same
# window.  The LLM is the primary pacing mechanism — no hard cooldown floor.
DEFAULT_MIN_GAP_SECONDS:          int = 10
DEFAULT_BREAK_COOLDOWN_SECONDS:   int = 300  # 5 minutes post-break
DEFAULT_AUTO_DISMISS_SECONDS:     int = 90   # unacknowledged text prompt TTL
CHIME_DISPLAY_SECONDS:            int = 12   # how long the chime toast stays visible


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class InterventionResult:
    """
    Parsed output from one intervention LLM call.

    ``intervene`` is the LLM's recommendation.  The router layer makes the
    final fire/skip decision based on the active-intervention gate.
    """
    intervene:  bool
    tier:       str
    type:       str | None          # None when tier=="none"
    content:    dict[str, Any] | None
    raw_json:   str
    latency_ms: int
    parse_ok:   bool = True

    def is_actionable(self) -> bool:
        """True when the LLM wants to fire and produced valid content."""
        return self.intervene and self.type is not None and self.content is not None


_PARSE_FALLBACK = InterventionResult(
    intervene=False, tier="none", type=None,
    content=None, raw_json="", latency_ms=0, parse_ok=False,
)


def _extract_first_json(raw: str) -> str | None:
    """
    Find and return the first complete, balanced JSON object in ``raw``.

    Uses brace-depth tracking so appended user-input content after the
    closing ``}`` does not corrupt the parse.  Returns None if not found.
    """
    for start in range(len(raw)):
        if raw[start] != "{":
            continue
        depth, in_str, escape = 0, False, False
        for end in range(start, len(raw)):
            ch = raw[end]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : end + 1]
    return None


def _parse_response(raw: str, latency_ms: int) -> InterventionResult:
    """
    Parse the LLM's JSON output into an InterventionResult.

    Uses balanced-brace extraction so trailing user-input echo or stray
    tokens after the closing ``}`` do not corrupt the parse.
    """
    raw = raw.strip()
    candidate = _extract_first_json(raw)
    if candidate is None:
        log.warning("[intervention_engine] No JSON object found in output: %r", raw[:200])
        return InterventionResult(**{**_PARSE_FALLBACK.__dict__, "raw_json": raw, "latency_ms": latency_ms})

    try:
        obj: dict[str, Any] = json.loads(candidate)
    except json.JSONDecodeError as exc:
        log.warning("[intervention_engine] JSON decode error — %s. Raw: %r", exc, raw[:200])
        return InterventionResult(**{**_PARSE_FALLBACK.__dict__, "raw_json": raw, "latency_ms": latency_ms})

    intervene = bool(obj.get("intervene", False))
    tier      = str(obj.get("tier", "none")).lower()
    itype     = obj.get("type")
    content   = obj.get("content")

    if tier not in VALID_TIERS:
        log.warning("[intervention_engine] Unknown tier %r — treating as 'none'", tier)
        tier = "none"

    if itype is not None:
        itype = str(itype).lower()
        if itype not in VALID_TYPES:
            log.warning("[intervention_engine] Unknown intervention type %r", itype)
            itype = None

    if tier == "none":
        intervene = False
        content   = None
        itype     = None

    return InterventionResult(
        intervene  = intervene,
        tier       = tier,
        type       = itype,
        content    = content if isinstance(content, (dict, type(None))) else {"raw": content},
        raw_json   = candidate,
        latency_ms = latency_ms,
        parse_ok   = True,
    )


# ── Active intervention tracker ───────────────────────────────────────────────

@dataclass
class ActiveIntervention:
    """Represents a single intervention that is currently shown to the user."""
    intervention_id: int
    itype:           str
    tier:            str
    fired_at:        datetime


@dataclass
class FireDecision:
    """
    Result of the gate check for a proposed intervention type.

    ``allowed``  — whether the type may fire right now.
    ``reason``   — human-readable gate outcome:
                     "clear"                  → allowed
                     "type_already_active"    → same type on screen
                     "text_slots_full"        → 2 text prompts already active
                     "all_slots_full"         → 3 foreground items already active
                     "min_gap"                → fired too recently (< 60 s ago)
                     "break_active"           → break running; nothing fires
                     "post_break_cooldown"    → within 5-min post-break window
    """
    allowed: bool
    reason:  str

    @property
    def cooldown_status(self) -> str:
        """'clear' or 'cooling' — matches the LLM training data vocabulary."""
        return "clear" if self.allowed else "cooling"


class ActiveInterventionTracker:
    """
    In-process, per-session intervention state machine.

    Tracks what is currently active on screen so the gate can enforce:
      • No duplicate types
      • Max 2 text prompts simultaneously
      • Max 3 foreground interventions simultaneously
      • Break-suggestion exclusivity + 5-min post-break cooldown
      • 60-second minimum gap between fires (safety floor)
      • Auto-dismiss stale text prompts after 60 s (unacknowledged TTL)

    Designed as a module-level singleton; asyncio single-thread safe.
    """

    def __init__(
        self,
        min_gap_seconds:        int = DEFAULT_MIN_GAP_SECONDS,
        break_cooldown_seconds: int = DEFAULT_BREAK_COOLDOWN_SECONDS,
        auto_dismiss_seconds:   int = DEFAULT_AUTO_DISMISS_SECONDS,
    ) -> None:
        self._min_gap        = min_gap_seconds
        self._break_cooldown = break_cooldown_seconds
        self._auto_dismiss   = auto_dismiss_seconds

        # session_id → {itype: ActiveIntervention}
        self._active: dict[int, dict[str, ActiveIntervention]] = {}
        # session_id → UTC datetime of last fire (any type)
        self._last_fired: dict[int, datetime] = {}
        # session_id → UTC datetime when the user resumed after a break
        self._break_resume: dict[int, datetime] = {}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _auto_dismiss_stale(self, session_id: int) -> None:
        """
        Remove stale interventions from the active registry:
          • TEXT_PROMPT types older than auto_dismiss_seconds (unacknowledged).
          • INSTANT types (chime) older than CHIME_DISPLAY_SECONDS — these are
            short-lived toasts that auto-dismiss on the frontend but we also
            clean them here so they don't block the gate.
        Called before every gate check and before returning active list.
        """
        active = self._active.get(session_id)
        if not active:
            return
        now     = self._now()
        expired = [
            itype for itype, ai in active.items()
            if (
                itype in TEXT_PROMPT_TYPES
                and (now - ai.fired_at).total_seconds() > self._auto_dismiss
            ) or (
                itype in INSTANT_TYPES
                and (now - ai.fired_at).total_seconds() > CHIME_DISPLAY_SECONDS
            )
        ]
        for itype in expired:
            del active[itype]
            log.info(
                "[tracker] AUTO-DISMISSED session=%d type=%s (stale)",
                session_id, itype,
            )

    def _foreground_active(self, session_id: int) -> dict[str, ActiveIntervention]:
        """Non-passive active interventions for this session."""
        return {
            t: ai for t, ai in self._active.get(session_id, {}).items()
            if t not in PASSIVE_TYPES
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        session_id:    int,
        itype:         str,
        primary_state: str = "focused",
    ) -> FireDecision:
        """
        Gate check — returns a FireDecision describing whether ``itype``
        may fire right now and the reason if not.

        ``primary_state`` is used to set the effective minimum gap dynamically:
          - drifting / cognitive_overload → 0 s  (intervene freely)
          - focused / hyperfocused         → self._min_gap  (preserve focus)
        """
        self._auto_dismiss_stale(session_id)
        now = self._now()

        # 1. Break active → nothing fires (break must be acknowledged first)
        if "break_suggestion" in self._active.get(session_id, {}):
            return FireDecision(False, "break_active")

        # 2. Post-break cooldown for break_suggestion itself
        if itype in NUCLEAR_TYPES:
            resume = self._break_resume.get(session_id)
            if resume is not None:
                elapsed = (now - resume).total_seconds()
                if elapsed < self._break_cooldown:
                    return FireDecision(False, "post_break_cooldown")

        # 3. Passive types — fire freely, but not if the same type is already
        #    active (gamification / ambient_sound are session-persistent widgets;
        #    re-firing them while they are already on screen is redundant).
        if itype in PASSIVE_TYPES:
            if itype in self._active.get(session_id, {}):
                return FireDecision(False, "type_already_active")
            return FireDecision(True, "clear")

        # 4. Dynamic minimum gap:
        #    • 0 s when student is drifting/overloaded — speed of response matters
        #    • self._min_gap when student is focused — avoid interrupting good flow
        if primary_state in ("drifting", "cognitive_overload"):
            effective_gap = 0
        else:
            effective_gap = self._min_gap

        last = self._last_fired.get(session_id)
        if effective_gap > 0 and last is not None and (now - last).total_seconds() < effective_gap:
            return FireDecision(False, "min_gap")

        # 5. Instant types — subject only to the global min gap, but also blocked
        #    if the same chime is still showing (within CHIME_DISPLAY_SECONDS).
        if itype in INSTANT_TYPES:
            if itype in self._active.get(session_id, {}):
                return FireDecision(False, "type_already_active")
            return FireDecision(True, "clear")

        # 6. No duplicate types
        if itype in self._active.get(session_id, {}):
            return FireDecision(False, "type_already_active")

        foreground = self._foreground_active(session_id)

        # 7. Text-prompt cap
        if itype in TEXT_PROMPT_TYPES:
            text_count = sum(1 for t in foreground if t in TEXT_PROMPT_TYPES)
            if text_count >= MAX_TEXT_PROMPTS:
                return FireDecision(False, "text_slots_full")

        # 8. Total foreground cap
        if len(foreground) >= MAX_FOREGROUND_ACTIVE:
            return FireDecision(False, "all_slots_full")

        return FireDecision(True, "clear")

    def mark_fired(
        self,
        session_id:      int,
        intervention_id: int,
        itype:           str,
        tier:            str,
    ) -> None:
        """
        Record that ``itype`` was just fired.

        PASSIVE types go into their own registry but are never subject to
        slot eviction.  INSTANT types update the last_fired clock only.
        All other types occupy a foreground slot.
        """
        now = self._now()
        self._last_fired[session_id] = now

        # All types — including INSTANT (chime) — are tracked in _active so the
        # frontend polling /active can see them.  Chime is auto-expired after
        # CHIME_DISPLAY_SECONDS by _auto_dismiss_stale; all others stay until
        # acknowledged or their own TTL.
        if session_id not in self._active:
            self._active[session_id] = {}
        self._active[session_id][itype] = ActiveIntervention(
            intervention_id = intervention_id,
            itype           = itype,
            tier            = tier,
            fired_at        = now,
        )

    def acknowledge(self, session_id: int, itype: str) -> None:
        """
        User has dismissed/acknowledged intervention of type ``itype``.

        For break_suggestion, records the resume timestamp which starts the
        5-minute post-break cooldown.
        """
        active = self._active.get(session_id, {})
        if itype in active:
            del active[itype]
            if itype in NUCLEAR_TYPES:
                self._break_resume[session_id] = self._now()
            log.info(
                "[tracker] ACKNOWLEDGED session=%d type=%s", session_id, itype,
            )

    def active_for_session(self, session_id: int) -> list[ActiveIntervention]:
        """
        Return currently active interventions (auto-dismisses stale text prompts
        first).  Passive + foreground types are included.
        """
        self._auto_dismiss_stale(session_id)
        return list(self._active.get(session_id, {}).values())

    def seconds_since_last(self, session_id: int) -> int | None:
        """Seconds since the last fired intervention (any type), or None."""
        last = self._last_fired.get(session_id)
        if last is None:
            return None
        return int((self._now() - last).total_seconds())

    def status(self, session_id: int, primary_state: str = "focused") -> str:
        """
        Simplified 'clear'/'cooling' signal for the LLM prompt.

        Computed using a representative text-prompt type so it reflects
        the most common foreground gate.
        """
        return self.check(session_id, "re_engagement", primary_state).cooldown_status

    def reset(self, session_id: int) -> None:
        """Wipe all state for this session (call on session end)."""
        self._active.pop(session_id, None)
        self._last_fired.pop(session_id, None)
        self._break_resume.pop(session_id, None)


# Module-level singleton — lazily initialised from config on first access
_tracker: ActiveInterventionTracker | None = None


def get_active_tracker() -> ActiveInterventionTracker:
    """Return the shared ActiveInterventionTracker (lazy-initialised from config)."""
    global _tracker
    if _tracker is None:
        try:
            from app.core.config import settings
            _tracker = ActiveInterventionTracker(
                min_gap_seconds        = settings.intervention_cooldown_seconds,
                break_cooldown_seconds = settings.intervention_break_cooldown_seconds,
                auto_dismiss_seconds   = settings.intervention_auto_dismiss_seconds,
            )
        except Exception:
            _tracker = ActiveInterventionTracker()
    return _tracker


# Backward-compatible alias used by existing tests and any callers that
# imported the old name.  Will be removed after full migration.
def get_cooldown_tracker() -> ActiveInterventionTracker:
    return get_active_tracker()


# ── Ollama client ─────────────────────────────────────────────────────────────

class InterventionEngine:
    """
    Calls the ``lockin-intervention`` Ollama model via /api/generate with
    ``raw=true`` and the pre-formatted ChatML prompt string.

    Open-Closed: sub-class or replace to swap the backend without touching
    the router.
    """

    def __init__(
        self,
        base_url: str  = "http://localhost:11434",
        model:    str  = "lockin-intervention",
        timeout:  float = 60.0,
    ) -> None:
        self._generate_url = f"{base_url}/api/generate"
        self._health_url   = f"{base_url}/api/tags"
        self._model        = model
        self._timeout      = timeout

    async def call(self, raw_chatml_prompt: str) -> InterventionResult:
        """
        Send the pre-built ChatML prompt and parse the JSON response.

        Uses ``raw=true`` to bypass Ollama's template engine.
        ``num_ctx=4096`` overrides the Modelfile default (2048) — required
        because the full prompt (system ~662t + dense text window ~800t +
        session JSON ~300t) can exceed 2048 tokens, causing silent failures.
        ``num_predict=220`` covers the longest trained output type (section_summary
        ~150 tokens) with headroom, and saves ~6 s vs 350 at 23 tok/s on Metal.
        ``temperature=0.5`` gives enough freedom to select minority types
        (chime, text_reformat, ambient_sound) that have lower learned priors.
        ``timeout=60s`` is conservative for ~14 s typical call (4 s prefill +
        9.5 s generation at 23 tok/s Metal); catches genuine hangs quickly.
        """
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(
                    self._generate_url,
                    json={
                        "model":  self._model,
                        "prompt": raw_chatml_prompt,
                        "stream": False,
                        "raw":    True,
                        "options": {
                            "num_ctx":        4096,
                            "temperature":    0.5,
                            "top_k":          50,
                            "top_p":          0.95,
                            "repeat_penalty": 1.05,
                            "num_predict":    220,
                        },
                    },
                )
                r.raise_for_status()
        except httpx.HTTPError as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log.warning("[intervention_engine] HTTP error — %s", exc)
            return InterventionResult(
                **{**_PARSE_FALLBACK.__dict__,
                   "raw_json": f"[http_error] {exc}",
                   "latency_ms": latency_ms}
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        raw_continuation = r.json().get("response", "")
        # The prompt seeds the assistant turn with {"intervene": so Ollama
        # returns only the continuation (e.g. `true,"type":"chime",...}`).
        # Prepend the seeded prefix to reconstruct the complete JSON object.
        raw_text = '{"intervene":' + raw_continuation
        result   = _parse_response(raw_text, latency_ms)

        log.info(
            "[intervention_engine] intervene=%s tier=%s type=%s latency=%dms parse_ok=%s",
            result.intervene, result.tier, result.type, latency_ms, result.parse_ok,
        )
        return result

    async def health_check(self) -> bool:
        """Return True if the lockin-intervention model is registered in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(self._health_url)
                if r.status_code != 200:
                    return False
                models = [m.get("name", "") for m in r.json().get("models", [])]
                return any(self._model in m for m in models)
        except Exception:
            return False


# Module-level singleton — lazily reads config on first access
_engine: InterventionEngine | None = None


def get_engine() -> InterventionEngine:
    """Return the shared InterventionEngine instance (lazy-initialised from config)."""
    global _engine
    if _engine is None:
        from app.core.config import settings
        _engine = InterventionEngine(
            base_url = settings.ollama_url,
            model    = settings.intervention_model,
        )
    return _engine
