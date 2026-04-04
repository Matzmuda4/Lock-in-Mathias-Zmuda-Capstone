"""
Tests for the intervention engine — Steps 1-3 (revised cooldown model).

Covers:
  1. Pure unit tests (no network, no DB):
       - prompt.py — ChatML construction, active_interventions field
       - engine.py — JSON parsing, ActiveInterventionTracker gate logic
       - templates.py — all 9 types present and valid shape

  2. Integration tests via HTTPX async client (test DB, no real Ollama):
       - GET  /intervention-engine/health
       - POST /sessions/{id}/interventions/manual  (all 9 types)
       - GET  /sessions/{id}/interventions/active
       - GET  /sessions/{id}/interventions/pending (legacy alias)
       - POST /sessions/{id}/interventions/{id}/acknowledge  (frees slot)
       - POST /sessions/{id}/interventions/trigger  (no attentional data → 422)
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UNIT TESTS — no network, no DB
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptBuilder:

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
            active_interventions   = [{"type": "chime", "tier": "subtle", "seconds_active": 5}],
        )
        assert result["session_context"]["elapsed_minutes"]    == 5.0
        assert result["session_context"]["session_stage"]      == "mid"
        assert result["session_context"]["cooldown_status"]    == "clear"
        assert result["session_context"]["active_interventions"] == [
            {"type": "chime", "tier": "subtle", "seconds_active": 5}
        ]
        assert result["reading_context"]["text_window"] == ["Attention is not a monolithic faculty."]

    def test_active_interventions_defaults_to_empty(self):
        from app.services.intervention.prompt import build_intervention_input
        result = build_intervention_input(
            elapsed_minutes=2.0, attentional_window=[], drift_progression={},
            user_baseline={}, text_window=[], current_paragraph_index=None,
            xp=0, badges_earned=[], last_intervention=None, cooldown_status="clear",
        )
        assert result["session_context"]["active_interventions"] == []

    def test_build_raw_chatml_prompt_tokens(self):
        from app.services.intervention.prompt import build_intervention_input, build_raw_chatml_prompt
        inp    = build_intervention_input(
            elapsed_minutes=2.0, attentional_window=[], drift_progression={},
            user_baseline={}, text_window=[], current_paragraph_index=None,
            xp=0, badges_earned=[], last_intervention=None, cooldown_status="clear",
        )
        prompt = build_raw_chatml_prompt(inp)
        assert prompt.startswith("<|im_start|>system\n")
        assert "<|im_end|>" in prompt
        assert "<|im_start|>user\n" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")
        user_start = prompt.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        user_end   = prompt.index("<|im_end|>", user_start)
        parsed     = json.loads(prompt[user_start:user_end])
        assert "session_context" in parsed
        assert "reading_context" in parsed
        assert "active_interventions" in parsed["session_context"]


class TestParseResponse:

    def _parse(self, raw: str, latency: int = 100):
        from app.services.intervention.engine import _parse_response
        return _parse_response(raw, latency)

    def test_valid_intervene_true(self):
        raw = '{"intervene":true,"tier":"moderate","type":"re_engagement","content":{"headline":"Stay","body":"Breathe.","cta":"Got it"}}'
        r = self._parse(raw)
        assert r.parse_ok and r.intervene and r.tier == "moderate"
        assert r.type == "re_engagement"

    def test_valid_intervene_false(self):
        raw = '{"intervene":false,"tier":"none","type":null,"content":null}'
        r = self._parse(raw)
        assert r.parse_ok and not r.intervene and r.tier == "none"

    def test_tier_none_forces_no_action(self):
        raw = '{"intervene":true,"tier":"none","type":"chime","content":{}}'
        r = self._parse(raw)
        assert not r.intervene and r.tier == "none" and r.type is None

    def test_unknown_type_normalised(self):
        raw = '{"intervene":true,"tier":"subtle","type":"UNKNOWN_XYZ","content":{}}'
        r   = self._parse(raw)
        assert r.parse_ok and r.type is None

    def test_unknown_tier_normalised(self):
        raw = '{"intervene":true,"tier":"critical","type":"chime","content":{}}'
        assert self._parse(raw).tier == "none"

    def test_no_json_fallback(self):
        r = self._parse("Hello, I cannot produce JSON right now.")
        assert not r.parse_ok and not r.intervene

    def test_leading_whitespace_stray_tokens(self):
        raw = '   \n{"intervene":true,"tier":"subtle","type":"chime","content":{"sound":"chime"}}\n extra'
        r   = self._parse(raw)
        assert r.parse_ok and r.intervene

    def test_latency_propagated(self):
        raw = '{"intervene":false,"tier":"none","type":null,"content":null}'
        assert self._parse(raw, latency=42).latency_ms == 42

    def test_is_actionable(self):
        raw = '{"intervene":true,"tier":"moderate","type":"focus_point","content":{"prompt":"find it"}}'
        assert self._parse(raw).is_actionable()

    def test_not_actionable_null_content(self):
        raw = '{"intervene":true,"tier":"moderate","type":"focus_point","content":null}'
        assert not self._parse(raw).is_actionable()


class TestActiveInterventionTracker:
    """Gate logic — all branches of ActiveInterventionTracker.check()."""

    def _tracker(self, min_gap=0, break_cd=300, auto_dismiss=60):
        """Fresh tracker with 0-second min_gap by default for gate logic tests."""
        from app.services.intervention.engine import ActiveInterventionTracker
        return ActiveInterventionTracker(
            min_gap_seconds        = min_gap,
            break_cooldown_seconds = break_cd,
            auto_dismiss_seconds   = auto_dismiss,
        )

    # ── Initially clear ───────────────────────────────────────────────────────

    def test_initially_clear_text_prompt(self):
        t = self._tracker()
        d = t.check(1, "re_engagement")
        assert d.allowed and d.reason == "clear"

    def test_initially_clear_passive(self):
        t = self._tracker()
        assert t.check(1, "gamification").allowed

    def test_initially_clear_instant(self):
        t = self._tracker()
        assert t.check(1, "chime").allowed

    # ── No duplicate types ────────────────────────────────────────────────────

    def test_duplicate_type_blocked(self):
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point", "subtle")
        d = t.check(1, "focus_point")
        assert not d.allowed and d.reason == "type_already_active"

    def test_different_type_allowed_after_first_fire(self):
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point", "subtle")
        assert t.check(1, "re_engagement").allowed

    # ── Text-prompt cap ───────────────────────────────────────────────────────

    def test_two_text_prompts_allowed(self):
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point",   "subtle")
        t.mark_fired(1, 11, "re_engagement", "moderate")
        # third text prompt blocked
        d = t.check(1, "section_summary")
        assert not d.allowed and d.reason == "text_slots_full"

    def test_non_text_allowed_when_text_slots_full(self):
        """text_reformat is non-text; it should still fire when text slots are full."""
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point",   "subtle")
        t.mark_fired(1, 11, "re_engagement", "moderate")
        assert t.check(1, "text_reformat").allowed

    # ── Total foreground cap ──────────────────────────────────────────────────

    def test_three_foreground_slots_fill(self):
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point",   "subtle")
        t.mark_fired(1, 11, "re_engagement", "moderate")
        t.mark_fired(1, 12, "text_reformat", "moderate")
        # Any further foreground type is blocked
        d = t.check(1, "break_suggestion")
        assert not d.allowed and d.reason == "all_slots_full"

    def test_passive_always_fires_past_cap(self):
        """Passive types bypass slot checks entirely."""
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point",   "subtle")
        t.mark_fired(1, 11, "re_engagement", "moderate")
        t.mark_fired(1, 12, "text_reformat", "moderate")
        assert t.check(1, "gamification").allowed
        assert t.check(1, "ambient_sound").allowed

    def test_instant_bypasses_slot_check(self):
        """Chime never occupies a slot so it fires even when 3 slots are full."""
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point",   "subtle")
        t.mark_fired(1, 11, "re_engagement", "moderate")
        t.mark_fired(1, 12, "text_reformat", "moderate")
        assert t.check(1, "chime").allowed

    # ── Break suggestion rules ────────────────────────────────────────────────

    def test_break_active_blocks_everything(self):
        t = self._tracker()
        t.mark_fired(1, 99, "break_suggestion", "strong")
        for itype in ["re_engagement", "chime", "gamification", "text_reformat"]:
            d = t.check(1, itype)
            assert not d.allowed and d.reason == "break_active", itype

    def test_post_break_cooldown_blocks_new_break(self):
        t = self._tracker(break_cd=300)
        t.mark_fired(1, 99, "break_suggestion", "strong")
        t.acknowledge(1, "break_suggestion")   # starts the 5-min clock
        d = t.check(1, "break_suggestion")
        assert not d.allowed and d.reason == "post_break_cooldown"

    def test_post_break_non_nuclear_allowed_immediately(self):
        """After a break is acknowledged, non-break types fire immediately."""
        t = self._tracker(break_cd=300)
        t.mark_fired(1, 99, "break_suggestion", "strong")
        t.acknowledge(1, "break_suggestion")
        assert t.check(1, "re_engagement").allowed

    # ── Minimum gap floor ─────────────────────────────────────────────────────

    def test_min_gap_blocks_within_window(self):
        t = self._tracker(min_gap=60)
        t.mark_fired(1, 10, "focus_point", "subtle")
        t.acknowledge(1, "focus_point")  # free the slot, but gap remains
        d = t.check(1, "re_engagement")
        assert not d.allowed and d.reason == "min_gap"

    def test_min_gap_clears_after_elapsed(self):
        t = self._tracker(min_gap=1)   # 1-second gap for testing
        t.mark_fired(1, 10, "focus_point", "subtle")
        t.acknowledge(1, "focus_point")
        import time; time.sleep(1.1)
        assert t.check(1, "re_engagement").allowed

    # ── Acknowledge ────────────────────────────────────────────────────────────

    def test_acknowledge_frees_slot(self):
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point", "subtle")
        assert not t.check(1, "focus_point").allowed
        t.acknowledge(1, "focus_point")
        assert t.check(1, "focus_point").allowed

    def test_acknowledge_different_sessions_independent(self):
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point", "subtle")
        t.mark_fired(2, 20, "re_engagement", "moderate")
        t.acknowledge(1, "focus_point")
        assert t.check(1, "focus_point").allowed
        assert not t.check(2, "re_engagement").allowed

    # ── Auto-dismiss ─────────────────────────────────────────────────────────

    def test_stale_text_prompt_auto_dismissed(self):
        t = self._tracker(auto_dismiss=1)   # 1-second TTL
        t.mark_fired(1, 10, "focus_point", "subtle")
        # Backdate the fired_at
        t._active[1]["focus_point"].fired_at -= timedelta(seconds=2)
        # Now the slot should be freed automatically
        assert t.check(1, "focus_point").allowed

    def test_passive_not_auto_dismissed(self):
        """Gamification/ambient stay forever unless explicitly acknowledged."""
        t = self._tracker(auto_dismiss=1)
        t.mark_fired(1, 10, "gamification", "subtle")
        t._active[1]["gamification"].fired_at -= timedelta(seconds=10)
        # Still present after auto-dismiss sweep
        active = t.active_for_session(1)
        assert any(ai.itype == "gamification" for ai in active)

    # ── Status / helpers ──────────────────────────────────────────────────────

    def test_status_clear_initially(self):
        t = self._tracker()
        assert t.status(1) == "clear"

    def test_status_cooling_after_fire(self):
        t = self._tracker(min_gap=60)
        t.mark_fired(1, 10, "focus_point", "subtle")
        assert t.status(1) == "cooling"   # min_gap blocks re_engagement check

    def test_seconds_since_last_none_initially(self):
        t = self._tracker()
        assert t.seconds_since_last(1) is None

    def test_seconds_since_last_after_fire(self):
        t = self._tracker()
        t.mark_fired(1, 10, "chime", "subtle")  # chime updates last_fired
        secs = t.seconds_since_last(1)
        assert secs is not None and secs < 5

    def test_reset_clears_all_state(self):
        t = self._tracker()
        t.mark_fired(1, 10, "focus_point", "subtle")
        t.reset(1)
        assert t.check(1, "focus_point").allowed
        assert t.seconds_since_last(1) is None
        assert t.active_for_session(1) == []

    def test_cooldown_status_property(self):
        from app.services.intervention.engine import FireDecision
        assert FireDecision(True,  "clear").cooldown_status   == "clear"
        assert FireDecision(False, "min_gap").cooldown_status == "cooling"


class TestManualTemplates:
    EXPECTED = {
        "re_engagement", "focus_point", "section_summary", "comprehension_check",
        "break_suggestion", "gamification", "chime", "ambient_sound", "text_reformat",
    }

    def test_all_types_present(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        assert self.EXPECTED == set(MANUAL_TEMPLATES.keys())

    def test_re_engagement_keys(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        t = MANUAL_TEMPLATES["re_engagement"]
        assert "headline" in t and "body" in t and "cta" in t

    def test_comprehension_check_keys(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        t = MANUAL_TEMPLATES["comprehension_check"]
        assert "type" in t and "question" in t and "answer" in t

    def test_break_suggestion_auto_pause(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        assert MANUAL_TEMPLATES["break_suggestion"].get("auto_pause") is True

    def test_gamification_xp(self):
        from app.services.intervention.templates import MANUAL_TEMPLATES
        assert isinstance(MANUAL_TEMPLATES["gamification"].get("xp_awarded"), int)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INTEGRATION TESTS — test DB, HTTPX, no real Ollama
# ═══════════════════════════════════════════════════════════════════════════════

async def _create_session(api_client: AsyncClient, auth_headers: dict) -> int:
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


def _reset_tracker_for(session_id: int) -> None:
    """Between integration tests, clear in-process tracker state for the session."""
    from app.services.intervention.engine import get_active_tracker
    get_active_tracker().reset(session_id)


class TestEngineHealth:
    async def test_shape(self, api_client: AsyncClient):
        resp = await api_client.get("/intervention-engine/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "available" in body and "model" in body
        assert body["model"] == "lockin-intervention"

    async def test_no_auth_required(self, api_client: AsyncClient):
        assert (await api_client.get("/intervention-engine/health")).status_code == 200


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
    async def test_all_types_fire(
        self, api_client: AsyncClient, auth_headers: dict, itype: str, tier: str,
    ):
        session_id = await _create_session(api_client, auth_headers)
        _reset_tracker_for(session_id)

        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": itype, "tier": tier},
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"{itype}: {resp.text}"
        body = resp.json()
        assert body["intervene"]       is True
        assert body["type"]            == itype
        assert body["tier"]            == tier
        assert body["session_id"]      == session_id
        assert body["intervention_id"] is not None
        assert body["content"]         is not None
        assert body["fired_at"]        is not None

    async def test_invalid_type_422(self, api_client: AsyncClient, auth_headers: dict):
        session_id = await _create_session(api_client, auth_headers)
        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "FAKE_TYPE", "tier": "subtle"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_invalid_tier_422(self, api_client: AsyncClient, auth_headers: dict):
        session_id = await _create_session(api_client, auth_headers)
        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "chime", "tier": "extreme"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_custom_content_override(self, api_client: AsyncClient, auth_headers: dict):
        session_id = await _create_session(api_client, auth_headers)
        custom = {"headline": "Custom", "body": "Body", "cta": "OK"}
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

    async def test_wrong_session_404(self, api_client: AsyncClient, auth_headers: dict):
        resp = await api_client.post(
            "/sessions/999999/interventions/manual",
            json={"type": "chime", "tier": "subtle"},
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestActiveEndpoint:
    async def test_empty_when_nothing_fired(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        session_id = await _create_session(api_client, auth_headers)
        _reset_tracker_for(session_id)
        resp = await api_client.get(
            f"/sessions/{session_id}/interventions/active",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_shows_active_after_manual_fire(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        session_id = await _create_session(api_client, auth_headers)
        _reset_tracker_for(session_id)

        await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "focus_point", "tier": "subtle"},
            headers=auth_headers,
        )

        resp = await api_client.get(
            f"/sessions/{session_id}/interventions/active",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["type"] == "focus_point"

    async def test_two_simultaneous_types_both_listed(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        session_id = await _create_session(api_client, auth_headers)
        _reset_tracker_for(session_id)

        # Fire focus_point then re_engagement (both text slots, both allowed)
        await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "focus_point", "tier": "subtle"},
            headers=auth_headers,
        )
        await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "re_engagement", "tier": "moderate"},
            headers=auth_headers,
        )

        resp = await api_client.get(
            f"/sessions/{session_id}/interventions/active",
            headers=auth_headers,
        )
        types = {item["type"] for item in resp.json()}
        assert "focus_point" in types
        assert "re_engagement" in types

    async def test_passive_included_in_active(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        """Gamification/ambient_sound appear in /active because they occupy a tracker slot."""
        session_id = await _create_session(api_client, auth_headers)
        _reset_tracker_for(session_id)

        await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "gamification", "tier": "subtle"},
            headers=auth_headers,
        )

        resp = await api_client.get(
            f"/sessions/{session_id}/interventions/active",
            headers=auth_headers,
        )
        types = {item["type"] for item in resp.json()}
        assert "gamification" in types


class TestPendingLegacy:
    async def test_null_when_nothing_fired(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        session_id = await _create_session(api_client, auth_headers)
        _reset_tracker_for(session_id)
        resp = await api_client.get(
            f"/sessions/{session_id}/interventions/pending",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json() is None

    async def test_returns_latest_after_fire(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        session_id = await _create_session(api_client, auth_headers)
        _reset_tracker_for(session_id)

        await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "focus_point", "tier": "subtle"},
            headers=auth_headers,
        )
        resp = await api_client.get(
            f"/sessions/{session_id}/interventions/pending",
            headers=auth_headers,
        )
        body = resp.json()
        assert body is not None
        assert body["type"] == "focus_point"


class TestAcknowledge:
    async def test_acknowledge_success(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        session_id = await _create_session(api_client, auth_headers)
        _reset_tracker_for(session_id)

        fire_resp       = await api_client.post(
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

    async def test_acknowledge_frees_slot(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        """
        After acknowledging a text prompt, /active should not include it.
        """
        session_id = await _create_session(api_client, auth_headers)
        _reset_tracker_for(session_id)

        fire_resp       = await api_client.post(
            f"/sessions/{session_id}/interventions/manual",
            json={"type": "focus_point", "tier": "subtle"},
            headers=auth_headers,
        )
        intervention_id = fire_resp.json()["intervention_id"]

        # Active before acknowledge
        active_before = await api_client.get(
            f"/sessions/{session_id}/interventions/active",
            headers=auth_headers,
        )
        assert any(i["type"] == "focus_point" for i in active_before.json())

        # Acknowledge
        await api_client.post(
            f"/sessions/{session_id}/interventions/{intervention_id}/acknowledge",
            headers=auth_headers,
        )

        # Should no longer appear in /active
        active_after = await api_client.get(
            f"/sessions/{session_id}/interventions/active",
            headers=auth_headers,
        )
        assert not any(i["type"] == "focus_point" for i in active_after.json())

    async def test_acknowledge_wrong_id_404(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        session_id = await _create_session(api_client, auth_headers)
        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/999999/acknowledge",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestTriggerNoData:
    async def test_trigger_before_rf_data_422(
        self, api_client: AsyncClient, auth_headers: dict,
    ):
        session_id = await _create_session(api_client, auth_headers)
        resp = await api_client.post(
            f"/sessions/{session_id}/interventions/trigger",
            headers=auth_headers,
        )
        assert resp.status_code == 422
        assert "attentional-state" in resp.json()["detail"].lower()
