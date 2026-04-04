"""
Tests for the intervention engine — Steps 1-3.

Covers:
  1. Pure unit tests (no network, no DB):
       - prompt.py — ChatML construction
       - engine.py — JSON parsing, cooldown tracker
       - templates.py — all 9 types present and valid shape

  2. Integration tests via HTTPX async client (test DB, no real Ollama):
       - GET  /intervention-engine/health
       - POST /sessions/{id}/interventions/manual  (all 9 types)
       - GET  /sessions/{id}/interventions/pending
       - POST /sessions/{id}/interventions/{id}/acknowledge
       - POST /sessions/{id}/interventions/trigger  (no attentional data → 422)

The Ollama call inside /trigger is NOT tested here (requires a live Ollama
instance with lockin-intervention loaded).  The live test is run separately
via the system check script.
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone
from httpx import AsyncClient


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UNIT TESTS — no network, no DB
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptBuilder:
    """prompt.py — pure functions."""

    def test_session_stage_early(self):
        from app.services.intervention.prompt import _session_stage
        assert _session_stage(0.0)  == "early"
        assert _session_stage(4.9)  == "early"

    def test_session_stage_mid(self):
        from app.services.intervention.prompt import _session_stage
        assert _session_stage(5.0)  == "mid"
        assert _session_stage(11.9) == "mid"

    def test_session_stage_late(self):
        from app.services.intervention.prompt import _session_stage
        assert _session_stage(12.0) == "late"
        assert _session_stage(99.0) == "late"

    def test_build_intervention_input_structure(self):
        from app.services.intervention.prompt import build_intervention_input
        result = build_intervention_input(
            elapsed_minutes        = 5.0,
            attentional_window     = [{"primary_state": "drifting", "confidence": 0.7,
                                       "distribution": {"focused": 0.3, "drifting": 0.7,
                                                        "hyperfocused": 0.0,
                                                        "cognitive_overload": 0.0}}],
            drift_progression      = {"drift_level": [0.4], "engagement_score": [0.6],
                                      "drift_ema": 0.4},
            user_baseline          = {"wpm_effective": 280.0, "idle_ratio_mean": 0.3,
                                      "regress_rate_mean": 0.03, "para_dwell_median_s": 15.0},
            text_window            = ["Attention is not a monolithic faculty."],
            current_paragraph_index= 12,
            xp                     = 0,
            badges_earned          = [],
            last_intervention      = None,
            cooldown_status        = "clear",
        )
        assert result["session_context"]["elapsed_minutes"]  == 5.0
        assert result["session_context"]["session_stage"]    == "mid"
        assert result["session_context"]["cooldown_status"]  == "clear"
        assert result["session_context"]["last_intervention"] is None
        assert len(result["attentional_state_window"]) == 1
        assert result["reading_context"]["text_window"] == ["Attention is not a monolithic faculty."]
        assert result["reading_context"]["current_paragraph_index"] == 12

    def test_build_raw_chatml_prompt_tokens(self):
        from app.services.intervention.prompt import build_intervention_input, build_raw_chatml_prompt
        inp = build_intervention_input(
            elapsed_minutes=2.0, attentional_window=[], drift_progression={},
            user_baseline={}, text_window=[], current_paragraph_index=None,
            xp=0, badges_earned=[], last_intervention=None, cooldown_status="clear",
        )
        prompt = build_raw_chatml_prompt(inp)
        assert prompt.startswith("<|im_start|>system\n")
        assert "<|im_end|>" in prompt
        assert "<|im_start|>user\n" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")
        # User content must be valid JSON
        user_start = prompt.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        user_end   = prompt.index("<|im_end|>", user_start)
        parsed     = json.loads(prompt[user_start:user_end])
        assert "session_context" in parsed
        assert "reading_context" in parsed


class TestParseResponse:
    """engine._parse_response — all edge cases."""

    def _parse(self, raw: str, latency: int = 100):
        from app.services.intervention.engine import _parse_response
        return _parse_response(raw, latency)

    def test_valid_intervene_true(self):
        raw = '{"intervene":true,"tier":"moderate","type":"re_engagement","content":{"headline":"Stay with it","body":"Take a breath.","cta":"Got it"}}'
        r = self._parse(raw)
        assert r.parse_ok
        assert r.intervene
        assert r.tier    == "moderate"
        assert r.type    == "re_engagement"
        assert r.content == {"headline": "Stay with it", "body": "Take a breath.", "cta": "Got it"}

    def test_valid_intervene_false(self):
        raw = '{"intervene":false,"tier":"none","type":null,"content":null}'
        r = self._parse(raw)
        assert r.parse_ok
        assert not r.intervene
        assert r.tier    == "none"
        assert r.type    is None
        assert r.content is None

    def test_tier_none_forces_no_action(self):
        raw = '{"intervene":true,"tier":"none","type":"chime","content":{}}'
        r = self._parse(raw)
        assert not r.intervene
        assert r.tier    == "none"
        assert r.type    is None

    def test_unknown_type_normalised(self):
        raw = '{"intervene":true,"tier":"subtle","type":"UNKNOWN_XYZ","content":{}}'
        r = self._parse(raw)
        assert r.parse_ok
        assert r.type is None

    def test_unknown_tier_normalised(self):
        raw = '{"intervene":true,"tier":"critical","type":"chime","content":{}}'
        r = self._parse(raw)
        assert r.tier == "none"

    def test_no_json_object_returns_fallback(self):
        r = self._parse("Hello, I cannot produce JSON right now.")
        assert not r.parse_ok
        assert not r.intervene

    def test_leading_whitespace_and_trailing_tokens(self):
        raw = '   \n{"intervene":true,"tier":"subtle","type":"chime","content":{"sound":"chime"}}\n extra'
        r = self._parse(raw)
        assert r.parse_ok
        assert r.intervene

    def test_latency_propagated(self):
        raw = '{"intervene":false,"tier":"none","type":null,"content":null}'
        r = self._parse(raw, latency=42)
        assert r.latency_ms == 42

    def test_is_actionable(self):
        raw = '{"intervene":true,"tier":"moderate","type":"focus_point","content":{"prompt":"find the key idea"}}'
        r = self._parse(raw)
        assert r.is_actionable()

    def test_not_actionable_when_no_content(self):
        raw = '{"intervene":true,"tier":"moderate","type":"focus_point","content":null}'
        r = self._parse(raw)
        assert not r.is_actionable()


class TestCooldownTracker:
    """CooldownTracker — state machine."""

    def test_initially_clear(self):
        from app.services.intervention.engine import CooldownTracker
        ct = CooldownTracker(cooldown_seconds=90)
        assert ct.status(1) == "clear"
        assert ct.seconds_since_last(1) is None

    def test_cooling_after_fire(self):
        from app.services.intervention.engine import CooldownTracker
        ct = CooldownTracker(cooldown_seconds=90)
        ct.mark_fired(1)
        assert ct.status(1) == "cooling"
        secs = ct.seconds_since_last(1)
        assert secs is not None and secs < 5

    def test_different_sessions_independent(self):
        from app.services.intervention.engine import CooldownTracker
        ct = CooldownTracker(cooldown_seconds=90)
        ct.mark_fired(1)
        assert ct.status(1) == "cooling"
        assert ct.status(2) == "clear"

    def test_reset_clears_state(self):
        from app.services.intervention.engine import CooldownTracker
        ct = CooldownTracker(cooldown_seconds=90)
        ct.mark_fired(5)
        assert ct.status(5) == "cooling"
        ct.reset(5)
        assert ct.status(5) == "clear"
        assert ct.seconds_since_last(5) is None

    def test_clear_after_elapsed(self):
        from app.services.intervention.engine import CooldownTracker
        from datetime import timedelta
        ct = CooldownTracker(cooldown_seconds=1)
        ct.mark_fired(7)
        # Manually backdate
        ct._last_fired[7] = ct._last_fired[7] - timedelta(seconds=2)
        assert ct.status(7) == "clear"


class TestManualTemplates:
    """templates.py — all 9 types present and non-empty."""

    EXPECTED_TYPES = {
        "re_engagement", "focus_point", "section_summary", "comprehension_check",
        "break_suggestion", "gamification", "chime", "ambient_sound", "text_reformat",
    }

    def test_all_types_present(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        assert self.EXPECTED_TYPES == set(MANUAL_TEMPLATES.keys())

    def test_re_engagement_has_required_keys(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        t = MANUAL_TEMPLATES["re_engagement"]
        assert "headline" in t and "body" in t and "cta" in t

    def test_comprehension_check_has_required_keys(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        t = MANUAL_TEMPLATES["comprehension_check"]
        assert "type" in t and "question" in t and "answer" in t

    def test_break_suggestion_has_auto_pause(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        t = MANUAL_TEMPLATES["break_suggestion"]
        assert t.get("auto_pause") is True

    def test_gamification_has_xp(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        t = MANUAL_TEMPLATES["gamification"]
        assert isinstance(t.get("xp_awarded"), int)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INTEGRATION TESTS — test DB, HTTPX, no real Ollama
# ═══════════════════════════════════════════════════════════════════════════════

# ── Shared helpers ────────────────────────────────────────────────────────────

async def _create_session(api_client: AsyncClient, auth_headers: dict) -> int:
    """Register a document, then create an adaptive session. Returns session_id."""
    # Upload a minimal document
    from io import BytesIO
    content = b"%PDF-1.4 1 0 obj<</Type/Catalog>>endobj"
    resp = await api_client.post(
        "/documents/upload",
        data={"title": "Test Document"},
        files={"file": ("test.pdf", BytesIO(content), "application/pdf")},
        headers=auth_headers,
    )
    assert resp.status_code in (200, 201), resp.text
    doc_id = resp.json()["id"]

    resp = await api_client.post(
        "/sessions/start",
        json={"document_id": doc_id, "name": "test session", "mode": "adaptive"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ── Health ────────────────────────────────────────────────────────────────────

class TestEngineHealth:
    async def test_health_endpoint_returns_shape(self, api_client: AsyncClient):
        resp = await api_client.get("/intervention-engine/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "available" in body
        assert "model" in body
        assert body["model"] == "lockin-intervention"

    async def test_health_endpoint_no_auth_required(self, api_client: AsyncClient):
        # Health check should be public
        resp = await api_client.get("/intervention-engine/health")
        assert resp.status_code == 200


# ── Manual trigger ────────────────────────────────────────────────────────────

class TestManualTrigger:
    @pytest.mark.parametrize("itype,tier", [
        ("re_engagement",      "moderate"),
        ("focus_point",        "subtle"),
        ("section_summary",    "moderate"),
        ("comprehension_check","subtle"),
        ("break_suggestion",   "strong"),
        ("gamification",       "subtle"),
        ("chime",              "subtle"),
        ("ambient_sound",      "moderate"),
        ("text_reformat",      "moderate"),
    ])
    async def test_all_intervention_types_fire(
        self, api_client: AsyncClient, auth_headers: dict, itype: str, tier: str
    ):
        session_id = await _create_session(api_client, auth_headers)

        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": itype, "tier": tier},
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"{itype}: {resp.text}"
        body = resp.json()
        assert body["intervene"]        is True
        assert body["type"]             == itype
        assert body["tier"]             == tier
        assert body["session_id"]       == session_id
        assert body["intervention_id"]  is not None
        assert body["content"]          is not None
        assert body["fired_at"]         is not None

    async def test_invalid_type_returns_422(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        session_id = await _create_session(api_client, auth_headers)
        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "FAKE_TYPE", "tier": "subtle"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_invalid_tier_returns_422(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        session_id = await _create_session(api_client, auth_headers)
        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "chime", "tier": "extreme"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_custom_content_override(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        session_id = await _create_session(api_client, auth_headers)
        custom = {"headline": "Custom headline", "body": "Custom body", "cta": "OK"}
        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "re_engagement", "tier": "moderate", "content": custom},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == custom

    async def test_requires_auth(self, api_client: AsyncClient):
        resp = await api_client.post(
            "/sessions/1/interventions/manual",
            json={"type": "chime", "tier": "subtle"},
        )
        assert resp.status_code == 401

    async def test_wrong_session_returns_404(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        resp = await api_client.post(
            "/sessions/999999/interventions/manual",
            json={"type": "chime", "tier": "subtle"},
            headers=auth_headers,
        )
        assert resp.status_code == 404


# ── Pending ───────────────────────────────────────────────────────────────────

class TestPendingIntervention:
    async def test_no_pending_returns_null(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        session_id = await _create_session(api_client, auth_headers)
        resp = await api_client.get(
            f"/sessions/{session_id}/interventions/pending",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json() is None

    async def test_after_manual_fire_pending_returns_payload(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        session_id = await _create_session(api_client, auth_headers)

        # Fire a manual intervention
        await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "focus_point", "tier": "subtle"},
            headers=auth_headers,
        )

        resp = await api_client.get(
            f"/sessions/{session_id}/interventions/pending",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body is not None
        assert body["type"]    == "focus_point"
        assert body["tier"]    == "subtle"
        assert body["content"] is not None


# ── Acknowledge ───────────────────────────────────────────────────────────────

class TestAcknowledge:
    async def test_acknowledge_success(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        session_id = await _create_session(api_client, auth_headers)

        # Fire a manual intervention to get an id
        fire_resp = await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "chime", "tier": "subtle"},
            headers=auth_headers,
        )
        intervention_id = fire_resp.json()["intervention_id"]

        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/{intervention_id}/acknowledge",
            headers=auth_headers,
        )
        assert resp.status_code == 204

    async def test_acknowledge_wrong_id_returns_404(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        session_id = await _create_session(api_client, auth_headers)
        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/999999/acknowledge",
            headers=auth_headers,
        )
        assert resp.status_code == 404


# ── Trigger (no attentional data) ────────────────────────────────────────────

class TestTriggerNoData:
    async def test_trigger_without_attentional_data_returns_422(
        self, api_client: AsyncClient, auth_headers: dict
    ):
        """Before any RF classifications exist, trigger must return 422."""
        session_id = await _create_session(api_client, auth_headers)
        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/trigger",
            headers=auth_headers,
        )
        assert resp.status_code == 422
        assert "attentional-state" in resp.json()["detail"].lower()
