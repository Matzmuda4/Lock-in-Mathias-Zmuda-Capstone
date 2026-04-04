"""
intervention/engine.py
──────────────────────
Calls the locally-running ``lockin-intervention`` Ollama model, parses its
JSON response, and enforces session-level cooldowns.

Single Responsibility:
  This module owns ONLY Ollama I/O + response parsing + cooldown state.
  Prompt construction lives in prompt.py.
  DB writes + session ownership checks live in the router.

Cooldown design:
  An in-process dict maps session_id → last_fired_at (UTC datetime).
  The router decides whether the cooldown is "clear" or "cooling" and passes
  that status INTO the prompt so the LLM is aware.  The engine independently
  enforces the cooldown for actual DB writes (i.e. it will return
  intervene=True payloads even during cooldown so the frontend can pre-cache
  the content, but the router only logs/fires when cooldown is truly clear).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.services.intervention.prompt import COOLDOWN_SECONDS

log = logging.getLogger(__name__)

# ── Response types ────────────────────────────────────────────────────────────

VALID_TYPES: frozenset[str] = frozenset({
    "re_engagement",
    "focus_point",
    "section_summary",
    "comprehension_check",
    "break_suggestion",
    "gamification",
    "chime",
    "ambient_sound",
    "text_reformat",
})

VALID_TIERS: frozenset[str] = frozenset({
    "none", "subtle", "moderate", "strong", "special",
})


@dataclass
class InterventionResult:
    """
    Parsed output from one intervention LLM call.

    ``intervene`` is the LLM's recommendation.  The router layer makes the
    final fire/skip decision based on cooldown and session state.
    """
    intervene:  bool
    tier:       str
    type:       str | None          # None when tier=="none"
    content:    dict[str, Any] | None
    raw_json:   str                 # original model output string
    latency_ms: int
    parse_ok:   bool = True

    def is_actionable(self) -> bool:
        """True when the LLM wants to fire and produced valid content."""
        return self.intervene and self.type is not None and self.content is not None


@dataclass
class _FallbackResult:
    """Sentinel used when parsing fails — not exported."""
    pass


_PARSE_FALLBACK = InterventionResult(
    intervene=False,
    tier="none",
    type=None,
    content=None,
    raw_json="",
    latency_ms=0,
    parse_ok=False,
)


def _parse_response(raw: str, latency_ms: int) -> InterventionResult:
    """
    Parse the LLM's single-line JSON output into an InterventionResult.

    Tolerates leading/trailing whitespace and stops at the first complete
    JSON object (handles any stray trailing tokens).
    """
    raw = raw.strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        log.warning("[intervention_engine] No JSON object found in output: %r", raw[:200])
        return InterventionResult(**{**_PARSE_FALLBACK.__dict__, "raw_json": raw, "latency_ms": latency_ms})

    try:
        obj: dict[str, Any] = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        log.warning("[intervention_engine] JSON decode error — %s. Raw: %r", exc, raw[:200])
        return InterventionResult(**{**_PARSE_FALLBACK.__dict__, "raw_json": raw, "latency_ms": latency_ms})

    intervene = bool(obj.get("intervene", False))
    tier      = str(obj.get("tier", "none")).lower()
    itype     = obj.get("type")
    content   = obj.get("content")

    # Normalise tier
    if tier not in VALID_TIERS:
        log.warning("[intervention_engine] Unknown tier %r — treating as 'none'", tier)
        tier = "none"

    # Normalise type
    if itype is not None:
        itype = str(itype).lower()
        if itype not in VALID_TYPES:
            log.warning("[intervention_engine] Unknown intervention type %r", itype)
            itype = None

    # If tier is none, force intervene=False and content=None
    if tier == "none":
        intervene = False
        content   = None
        itype     = None

    return InterventionResult(
        intervene  = intervene,
        tier       = tier,
        type       = itype,
        content    = content if isinstance(content, (dict, type(None))) else {"raw": content},
        raw_json   = raw[start:end + 1],
        latency_ms = latency_ms,
        parse_ok   = True,
    )


# ── Cooldown tracker ──────────────────────────────────────────────────────────

class CooldownTracker:
    """
    In-process, per-session cooldown state.

    Designed as a singleton (see module-level ``_cooldown_tracker``).
    Thread-safe enough for asyncio single-thread use; no locking needed.
    """

    def __init__(self, cooldown_seconds: int = COOLDOWN_SECONDS) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._last_fired: dict[int, datetime] = {}

    def status(self, session_id: int) -> str:
        """Return 'clear' or 'cooling' for the given session."""
        last = self._last_fired.get(session_id)
        if last is None:
            return "clear"
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return "clear" if elapsed >= self._cooldown_seconds else "cooling"

    def seconds_since_last(self, session_id: int) -> int | None:
        """Seconds since the last fired intervention, or None if never fired."""
        last = self._last_fired.get(session_id)
        if last is None:
            return None
        return int((datetime.now(timezone.utc) - last).total_seconds())

    def mark_fired(self, session_id: int) -> None:
        """Record that an intervention was just fired for this session."""
        self._last_fired[session_id] = datetime.now(timezone.utc)

    def reset(self, session_id: int) -> None:
        """Clear cooldown state for a session (e.g. on session end)."""
        self._last_fired.pop(session_id, None)


# Module-level singleton — shared across all request handlers
# Cooldown seconds are read from config on first access.
_cooldown_tracker: CooldownTracker | None = None


def get_cooldown_tracker() -> CooldownTracker:
    """Dependency accessor; also enables easy test injection."""
    global _cooldown_tracker
    if _cooldown_tracker is None:
        try:
            from app.core.config import settings
            secs = settings.intervention_cooldown_seconds
        except Exception:
            secs = COOLDOWN_SECONDS
        _cooldown_tracker = CooldownTracker(cooldown_seconds=secs)
    return _cooldown_tracker


# ── Ollama client ─────────────────────────────────────────────────────────────

class InterventionEngine:
    """
    Calls the ``lockin-intervention`` Ollama model via the /api/generate
    endpoint with ``raw=true`` and the pre-formatted ChatML prompt string.

    Open-Closed: sub-class or replace this class to swap the backend
    (e.g. OpenAI API, a different local model) without touching the router.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model:    str = "lockin-intervention",
        timeout:  float = 45.0,
    ) -> None:
        self._generate_url = f"{base_url}/api/generate"
        self._health_url   = f"{base_url}/api/tags"
        self._model        = model
        self._timeout      = timeout

    async def call(self, raw_chatml_prompt: str) -> InterventionResult:
        """
        Send the pre-built ChatML prompt and parse the JSON response.

        Uses ``raw=true`` to bypass Ollama's template engine — mandatory
        because the model was fine-tuned on explicit ChatML tokens.
        ``num_predict=256`` caps output to keep latency under 10 seconds.
        """
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(
                    self._generate_url,
                    json={
                        "model":       self._model,
                        "prompt":      raw_chatml_prompt,
                        "stream":      False,
                        "raw":         True,
                        "options": {
                            "temperature":    0.1,
                            "top_k":          20,
                            "top_p":          0.9,
                            "repeat_penalty": 1.1,
                            "num_predict":    256,
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
        raw_text   = r.json().get("response", "")
        result     = _parse_response(raw_text, latency_ms)

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
