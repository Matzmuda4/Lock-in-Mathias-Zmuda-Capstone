"""
Tests for Phase 7 — Personalized Drift Modelling.

Structure
---------
TestDriftModel         — pure maths (model.py)
TestDriftFeatures      — pure feature extraction (features.py)
TestDriftIntegration   — API: batch ingestion updates state, /drift endpoint
"""

import io
import math
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select

_MINIMAL_PDF = (
    b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)

_BATCH = {
    "scroll_delta_sum": 200.0,
    "scroll_delta_abs_sum": 200.0,
    "scroll_delta_pos_sum": 200.0,
    "scroll_delta_neg_sum": 0.0,
    "scroll_event_count": 4,
    "scroll_direction_changes": 0,
    "scroll_pause_seconds": 0.3,
    "idle_seconds": 0.2,
    "mouse_path_px": 150.0,
    "mouse_net_px": 120.0,
    "window_focus_state": "focused",
    "current_paragraph_id": "chunk-1",
    "current_chunk_index": 0,
    "viewport_progress_ratio": 0.3,
    "viewport_height_px": 800.0,
    "viewport_width_px": 1280.0,
    "reader_container_height_px": 760.0,
    "client_timestamp": "2026-02-25T12:00:00Z",
}


# ─── Pure model unit tests ─────────────────────────────────────────────────────


class TestDriftModel:
    def test_pace_dev_symmetric(self) -> None:
        """pace_ratio 0.5 and 2.0 should produce the same pace_dev (|log|)."""
        from app.services.drift.types import WindowFeatures

        f_slow = WindowFeatures(n_batches=10, pace_ratio=0.5, pace_dev=abs(math.log(0.5)))
        f_fast = WindowFeatures(n_batches=10, pace_ratio=2.0, pace_dev=abs(math.log(2.0)))
        assert abs(f_slow.pace_dev - f_fast.pace_dev) < 1e-9

    def test_disruption_increases_with_high_idle(self) -> None:
        """Higher idle_ratio → higher disruption_score."""
        from app.services.drift.model import compute_disruption_score, compute_z_scores
        from app.services.drift.types import WindowFeatures

        baseline = {"idle_ratio_mean": 0.05, "idle_ratio_std": 0.05}
        f_low = WindowFeatures(n_batches=10, idle_ratio_mean=0.05)
        f_high = WindowFeatures(n_batches=10, idle_ratio_mean=0.80)
        d_low, _ = compute_disruption_score(compute_z_scores(f_low, baseline), baseline)
        d_high, _ = compute_disruption_score(compute_z_scores(f_high, baseline), baseline)
        assert d_high > d_low

    def test_disruption_increases_with_high_jitter(self) -> None:
        from app.services.drift.model import compute_disruption_score, compute_z_scores
        from app.services.drift.types import WindowFeatures

        baseline = {"scroll_jitter_mean": 0.10, "scroll_jitter_std": 0.05}
        d_low, _ = compute_disruption_score(
            compute_z_scores(WindowFeatures(n_batches=5, scroll_jitter_mean=0.10), baseline),
            baseline,
        )
        d_high, _ = compute_disruption_score(
            compute_z_scores(WindowFeatures(n_batches=5, scroll_jitter_mean=0.80), baseline),
            baseline,
        )
        assert d_high > d_low

    def test_disruption_increases_with_focus_loss(self) -> None:
        from app.services.drift.model import compute_disruption_score, compute_z_scores
        from app.services.drift.types import WindowFeatures

        d_low, _ = compute_disruption_score(
            compute_z_scores(WindowFeatures(n_batches=5, focus_loss_rate=0.0), {}), {}
        )
        d_high, _ = compute_disruption_score(
            compute_z_scores(WindowFeatures(n_batches=5, focus_loss_rate=1.0), {}), {}
        )
        assert d_high > d_low

    def test_disruption_increases_with_high_regress(self) -> None:
        from app.services.drift.model import compute_disruption_score, compute_z_scores
        from app.services.drift.types import WindowFeatures

        baseline = {"regress_rate_mean": 0.05, "regress_rate_std": 0.05}
        d_low, _ = compute_disruption_score(
            compute_z_scores(WindowFeatures(n_batches=5, regress_rate_mean=0.05), baseline),
            baseline,
        )
        d_high, _ = compute_disruption_score(
            compute_z_scores(WindowFeatures(n_batches=5, regress_rate_mean=0.50), baseline),
            baseline,
        )
        assert d_high > d_low

    def test_disruption_bounded_zero_to_one(self) -> None:
        from app.services.drift.model import compute_disruption_score
        from app.services.drift.types import ZScores

        z_zero = ZScores()
        z_max = ZScores(z_idle=3, z_focus_loss=3, z_jitter=3, z_regress=3,
                        z_pause=3, z_stagnation=3, z_mouse=3, z_pace=3, z_skim=3)
        d_zero, _ = compute_disruption_score(z_zero, {})
        d_max, _ = compute_disruption_score(z_max, {})
        assert 0.0 <= d_zero <= 1.0
        assert 0.0 <= d_max <= 1.0

    def test_drift_level_increases_with_distraction(self) -> None:
        """
        Distracted features should drive drift_level up over time.

        With the exponential model (drift = 1-exp(-beta_ema * t)), passing
        increasing elapsed_minutes with a high-disruption feature set should
        produce clearly elevated drift.
        """
        from app.services.drift.model import BETA0, compute_drift_result
        from app.services.drift.types import WindowFeatures

        distracted = WindowFeatures(n_batches=15, idle_ratio_mean=0.9, focus_loss_rate=1.0)
        prev_beta_ema = BETA0
        prev_ema = 0.0
        result = None
        for i in range(30):
            elapsed = (i + 1) * 2.0 / 60.0
            result = compute_drift_result(distracted, {}, elapsed, prev_ema, prev_beta_ema)
            prev_beta_ema = result.beta_ema
            prev_ema = result.drift_ema
        assert result is not None
        # v4: BETA_MAX = 0.30 → max drift at 1 min = 1-exp(-0.30*1) ≈ 26%.
        # Heavy distraction (blur+idle) should reach that ceiling.
        assert result.drift_level > 0.20

    def test_drift_level_decreases_with_engagement(self) -> None:
        """
        Engaged reading should lower beta_effective vs distracted, leading
        to lower drift when elapsed time is the same.
        """
        from app.services.drift.model import BETA0, compute_drift_result
        from app.services.drift.types import WindowFeatures

        focused = WindowFeatures(n_batches=15, idle_ratio_mean=0.02, focus_loss_rate=0.0)
        distracted = WindowFeatures(n_batches=15, idle_ratio_mean=0.90, focus_loss_rate=1.0)

        r_f = compute_drift_result(focused, {}, 5.0, 0.0, BETA0)
        r_d = compute_drift_result(distracted, {}, 5.0, 0.0, BETA0)
        # Focused should produce lower beta and thus lower drift
        assert r_f.beta_effective < r_d.beta_effective
        assert r_f.drift_level < r_d.drift_level

    def test_ema_update_correct(self) -> None:
        from app.services.drift.model import update_ema, EMA_ALPHA

        ema = update_ema(0.8, 0.0)
        assert abs(ema - EMA_ALPHA * 0.8) < 1e-9

    def test_ema_converges_to_drift_score(self) -> None:
        from app.services.drift.model import update_ema

        ema = 0.0
        for _ in range(100):
            ema = update_ema(0.5, ema)
        assert abs(ema - 0.5) < 0.02

    def test_confidence_scales_with_batch_count(self) -> None:
        from app.services.drift.model import compute_confidence

        assert compute_confidence(0) == 0.0
        assert compute_confidence(15) == 1.0
        assert 0 < compute_confidence(7) < 1.0

    def test_full_result_produces_valid_ranges(self) -> None:
        from app.services.drift.model import compute_drift_result
        from app.services.drift.types import WindowFeatures

        f = WindowFeatures(n_batches=10, idle_ratio_mean=0.3, focus_loss_rate=0.2,
                           scroll_jitter_mean=0.2, pace_dev=0.3)
        r = compute_drift_result(f, {}, elapsed_minutes=5.0, prev_ema=0.0)
        assert 0.0 <= r.disruption_score <= 1.0
        assert 0.0 <= r.engagement_score <= 1.0
        assert 0.0 <= r.attention_score <= 1.0
        assert 0.0 <= r.drift_score <= 1.0
        assert 0.0 <= r.drift_ema <= 1.0
        assert 0.0 <= r.confidence <= 1.0


# ─── Feature extraction tests ─────────────────────────────────────────────────


class TestDriftFeatures:
    def test_scroll_velocity_norm_uses_viewport(self) -> None:
        from app.services.drift.features import scroll_velocity_norm

        small = scroll_velocity_norm(100.0, 500.0)
        large = scroll_velocity_norm(100.0, 1000.0)
        assert small > large
        assert abs(small - 100.0 / (500.0 * 2.0)) < 1e-9

    def test_jitter_ratio_correct(self) -> None:
        from app.services.drift.features import jitter_ratio

        assert abs(jitter_ratio(2, 10) - 0.2) < 1e-9
        assert jitter_ratio(0, 0) == 0.0

    def test_regress_rate_correct(self) -> None:
        from app.services.drift.features import regress_rate

        r = regress_rate(300.0, 100.0)
        assert abs(r - 100.0 / 400.0) < 1e-6
        assert regress_rate(0.0, 0.0) < 1e-3

    def test_mouse_efficiency_clamped(self) -> None:
        from app.services.drift.features import mouse_efficiency

        assert mouse_efficiency(0.0, 0.0) >= 0.0
        assert mouse_efficiency(100.0, 100.0) == 1.0
        assert mouse_efficiency(100.0, 200.0) == 1.0   # net > path → clamp to 1

    def test_paragraph_stagnation_high_when_same_id(self) -> None:
        from app.services.drift.features import paragraph_stagnation

        batches = [{"current_paragraph_id": "chunk-5"}] * 10
        assert paragraph_stagnation(batches) >= 0.5

    def test_paragraph_stagnation_low_when_varied(self) -> None:
        from app.services.drift.features import paragraph_stagnation

        batches = [{"current_paragraph_id": f"chunk-{i}"} for i in range(10)]
        assert paragraph_stagnation(batches) < 0.5

    def test_extract_features_returns_correct_n_batches(self) -> None:
        from app.services.drift.features import extract_features

        batches = [{**_BATCH}] * 8
        f = extract_features(batches, {"chunk-1": 100}, baseline_wpm_effective=150.0)
        assert f.n_batches == 8

    def test_extract_features_idle_ratio(self) -> None:
        from app.services.drift.features import extract_features

        high_idle = [{**_BATCH, "idle_seconds": 2.0}] * 5   # ratio = 1.0
        f = extract_features(high_idle, {}, 0.0)
        assert f.idle_ratio_mean == 1.0

    def test_extract_features_focus_loss_rate(self) -> None:
        from app.services.drift.features import extract_features

        blurred = [{**_BATCH, "window_focus_state": "blurred"}] * 4
        focused = [{**_BATCH, "window_focus_state": "focused"}] * 4
        f = extract_features(blurred + focused, {}, 0.0)
        assert abs(f.focus_loss_rate - 0.5) < 1e-9

    def test_extract_features_normalised_scroll(self) -> None:
        from app.services.drift.features import extract_features

        batch_big_vp = [{**_BATCH, "viewport_height_px": 1600.0}] * 5
        batch_sml_vp = [{**_BATCH, "viewport_height_px": 400.0}]  * 5
        f_big = extract_features(batch_big_vp, {}, 0.0)
        f_sml = extract_features(batch_sml_vp, {}, 0.0)
        assert f_sml.scroll_velocity_norm_mean > f_big.scroll_velocity_norm_mean


# ─── Integration tests ─────────────────────────────────────────────────────────


@pytest.fixture()
async def normal_session(api_client: AsyncClient, auth_headers: dict) -> dict:
    """Upload a doc and start a normal session. Returns {session_id, doc_id}."""
    doc_resp = await api_client.post(
        "/documents/upload",
        data={"title": "Drift Test Doc"},
        files={"file": ("d.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
        headers=auth_headers,
    )
    doc_id = doc_resp.json()["id"]

    sess_resp = await api_client.post(
        "/sessions/start",
        json={"document_id": doc_id, "name": "Drift Test", "mode": "baseline"},
        headers=auth_headers,
    )
    return {"session_id": sess_resp.json()["id"], "doc_id": doc_id}


class TestDriftIntegration:
    async def test_batch_creates_drift_state(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        normal_session: dict,
    ) -> None:
        """Posting a telemetry batch should create a session_drift_states row."""
        sid = normal_session["session_id"]

        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": sid, **_BATCH},
            headers=auth_headers,
        )
        assert resp.status_code == 201

        from app.db.models import SessionDriftState
        from app.db.session import async_session_factory
        async with async_session_factory() as db:
            row = (await db.execute(
                select(SessionDriftState).where(SessionDriftState.session_id == sid)
            )).scalar_one_or_none()
        assert row is not None
        assert 0.0 <= row.disruption_score <= 1.0
        assert 0.0 <= row.drift_score <= 1.0

    async def test_drift_endpoint_returns_200(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        normal_session: dict,
    ) -> None:
        sid = normal_session["session_id"]

        # Send a few batches first
        for _ in range(3):
            await api_client.post(
                "/activity/batch",
                json={"session_id": sid, **_BATCH},
                headers=auth_headers,
            )

        resp = await api_client.get(f"/sessions/{sid}/drift", headers=auth_headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "drift_ema" in body
        assert "beta_effective" in body
        assert 0.0 <= body["drift_score"] < 1.0

    async def test_drift_endpoint_enforces_ownership(
        self,
        api_client: AsyncClient,
        normal_session: dict,
    ) -> None:
        sid = normal_session["session_id"]
        resp_b = await api_client.post(
            "/auth/register",
            json={"username": "driftintruder", "email": "di@x.com", "password": "pw123456"},
        )
        headers_b = {"Authorization": f"Bearer {resp_b.json()['access_token']}"}
        resp = await api_client.get(f"/sessions/{sid}/drift", headers=headers_b)
        assert resp.status_code == 404

    async def test_drift_endpoint_requires_auth(
        self,
        api_client: AsyncClient,
        normal_session: dict,
    ) -> None:
        sid = normal_session["session_id"]
        resp = await api_client.get(f"/sessions/{sid}/drift")
        assert resp.status_code == 401

    async def test_multiple_batches_increase_drift(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        normal_session: dict,
    ) -> None:
        """High-idle batches should produce higher drift than normal batches."""
        sid = normal_session["session_id"]
        idle_batch = {**_BATCH, "idle_seconds": 60.0, "window_focus_state": "blurred"}

        for _ in range(10):
            await api_client.post(
                "/activity/batch",
                json={"session_id": sid, **idle_batch},
                headers=auth_headers,
            )

        resp = await api_client.get(f"/sessions/{sid}/drift", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        # With continuous blur + idle, disruption should be clearly positive
        assert body["disruption_score"] > 0.1

    async def test_drift_debug_blocked_without_debug_flag(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        normal_session: dict,
    ) -> None:
        sid = normal_session["session_id"]
        resp = await api_client.get(f"/sessions/{sid}/drift/debug", headers=auth_headers)
        # debug=False by default → 404
        assert resp.status_code == 404

    async def test_drift_debug_returns_features_when_enabled(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        normal_session: dict,
    ) -> None:
        sid = normal_session["session_id"]
        await api_client.post(
            "/activity/batch",
            json={"session_id": sid, **_BATCH},
            headers=auth_headers,
        )
        with patch("app.routers.drift.settings") as mock_settings:
            mock_settings.debug = True
            resp = await api_client.get(f"/sessions/{sid}/drift/debug", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "features" in body
        assert "z_scores" in body
        assert "pace_ratio" in body

    async def test_calibration_session_also_gets_drift(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Calibration sessions should also have drift state computed."""
        _CALIB_TEXT = "Alpha beta gamma.\n\nDelta epsilon zeta.\n\nEta theta iota.\n"
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write(_CALIB_TEXT)
            txt_path = Path(f.name)

        with patch("app.routers.calibration._CALIB_TXT", txt_path):
            start = await api_client.post("/calibration/start", headers=auth_headers)
        assert start.status_code == 201
        sid = start.json()["session_id"]

        await api_client.post(
            "/activity/batch",
            json={"session_id": sid, **_BATCH},
            headers=auth_headers,
        )

        from app.db.models import SessionDriftState
        from app.db.session import async_session_factory
        async with async_session_factory() as db:
            row = (await db.execute(
                select(SessionDriftState).where(SessionDriftState.session_id == sid)
            )).scalar_one_or_none()
        assert row is not None
        txt_path.unlink(missing_ok=True)
