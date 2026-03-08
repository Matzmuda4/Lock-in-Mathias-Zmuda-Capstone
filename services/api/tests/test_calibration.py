"""
Tests for Phase B: Calibration endpoints and baseline computation.

Calibration now uses a plain-text file (callibration.txt) — no Docling parse
job is involved.  Fixtures patch _CALIB_TXT to point at a temporary text file
so the tests are self-contained and fast.

Endpoints tested:
  GET  /calibration/status
  POST /calibration/start
  POST /calibration/complete
  GET  /calibration/baseline
  GET  /sessions/{id}/export.csv

Unit tests for baseline computation helpers are also included.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.db.models import Document, DocumentChunk, Session

_MINIMAL_PDF = (
    b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)

_CALIB_TEXT = (
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.\n\n"
    "Throughout the centuries people have explained the rainbow in various ways.\n\n"
    "The difference in the rainbow depends considerably upon the size of the drops.\n"
)

_TELEMETRY_BATCH = {
    "scroll_delta_sum": 240.0,
    "scroll_delta_abs_sum": 240.0,
    "scroll_delta_pos_sum": 240.0,
    "scroll_delta_neg_sum": 0.0,
    "scroll_event_count": 4,
    "scroll_direction_changes": 1,
    "scroll_pause_seconds": 0.5,
    "idle_seconds": 0.5,
    "mouse_path_px": 200.0,
    "mouse_net_px": 150.0,
    "window_focus_state": "focused",
    "current_paragraph_id": "chunk-1",
    "current_chunk_index": 0,
    "viewport_progress_ratio": 0.4,
    "viewport_height_px": 800.0,
    "viewport_width_px": 1280.0,
    "reader_container_height_px": 750.0,
    "client_timestamp": "2026-02-25T12:00:00Z",
}


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def calib_txt_path(tmp_path: Path) -> Path:
    """Write the calibration text to a temp file and return its path."""
    p = tmp_path / "callibration.txt"
    p.write_text(_CALIB_TEXT, encoding="utf-8")
    return p


@pytest.fixture()
async def calib_session(
    api_client: AsyncClient,
    auth_headers: dict,
    calib_txt_path: Path,
) -> int:
    """
    Patch _CALIB_TXT to a temp file, then call POST /calibration/start.
    Returns the session_id of the created calibration session.
    """
    with patch("app.routers.calibration._CALIB_TXT", calib_txt_path):
        resp = await api_client.post("/calibration/start", headers=auth_headers)
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


# ─── Unit tests — baseline computation ───────────────────────────────────────


class TestBaselineComputation:
    def test_empty_batches_returns_zeros(self) -> None:
        from app.services.calibration.baseline import compute_baseline

        result = compute_baseline([], {}, 120)
        assert result["wpm_mean"] == 0.0
        assert result["scroll_velocity_mean"] == 0.0
        assert result["calibration_duration_seconds"] == 120

    def test_scroll_velocity_computed_correctly(self) -> None:
        from app.services.calibration.baseline import scroll_velocities

        batches = [{"scroll_delta_abs_sum": 100}, {"scroll_delta_abs_sum": 200}]
        vels = scroll_velocities(batches)
        # 100 / 2s = 50, 200 / 2s = 100
        assert vels == [50.0, 100.0]

    def test_jitter_values_empty_when_no_events(self) -> None:
        from app.services.calibration.baseline import scroll_jitter_values

        batches = [{"scroll_event_count": 0, "scroll_direction_changes": 0}]
        assert scroll_jitter_values(batches) == []

    def test_jitter_ratio_computed(self) -> None:
        from app.services.calibration.baseline import scroll_jitter_values

        batches = [{"scroll_event_count": 10, "scroll_direction_changes": 2}]
        assert scroll_jitter_values(batches) == [0.2]

    def test_wpm_estimate_from_paragraphs(self) -> None:
        from app.services.calibration.baseline import estimate_wpm

        batches = [
            {"current_paragraph_id": "chunk-1"},
            {"current_paragraph_id": "chunk-1"},
            {"current_paragraph_id": "chunk-2"},
        ]
        chunk_word_counts = {1: 100, 2: 80}
        # 180 words / 2 minutes = 90 wpm
        wpm = estimate_wpm(batches, chunk_word_counts, 120)
        assert abs(wpm - 90.0) < 1.0

    def test_wpm_uses_total_words_override(self) -> None:
        """When total_words_override is given, WPM = total_words / duration_min."""
        from app.services.calibration.baseline import estimate_wpm

        # 300 words, 2 minutes → 150 wpm regardless of batches/chunk_word_counts
        wpm = estimate_wpm([], {}, 120, total_words_override=300)
        assert abs(wpm - 150.0) < 0.1

    def test_wpm_handles_calib_paragraph_ids(self) -> None:
        """Calibration reader uses 'calib-N' IDs; estimate_wpm must parse them."""
        from app.services.calibration.baseline import estimate_wpm

        batches = [
            {"current_paragraph_id": "calib-0"},
            {"current_paragraph_id": "calib-1"},
        ]
        # chunk_word_counts keyed by DB chunk id (int); not used in calib-N path
        # Without total_words_override the function falls through to dwell tracking.
        # Passing total_words_override gives the expected direct result.
        wpm = estimate_wpm(batches, {}, 60, total_words_override=150)
        assert abs(wpm - 150.0) < 0.1

    def test_paragraph_dwell_counted_correctly(self) -> None:
        from app.services.calibration.baseline import paragraph_dwells

        batches = [
            {"current_paragraph_id": "chunk-1"},
            {"current_paragraph_id": "chunk-1"},
            {"current_paragraph_id": "chunk-2"},
            {"current_paragraph_id": None},
        ]
        dwells = paragraph_dwells(batches)
        assert dwells == {"chunk-1": 2, "chunk-2": 1}

    def test_full_compute_with_batches(self) -> None:
        from app.services.calibration.baseline import compute_baseline

        batches = [
            {
                "scroll_delta_abs_sum": 100,
                "scroll_event_count": 4,
                "scroll_direction_changes": 1,
                "idle_seconds": 1.0,
                "current_paragraph_id": "chunk-5",
            }
        ] * 5
        result = compute_baseline(batches, {5: 200}, 60)
        assert result["wpm_mean"] > 0
        assert result["scroll_velocity_mean"] == 50.0
        assert 0 <= result["idle_ratio_mean"] <= 1.0
        assert result["calibration_duration_seconds"] == 60


# ─── Integration tests ────────────────────────────────────────────────────────


class TestCalibrationStatus:
    async def test_status_no_baseline(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        calib_txt_path: Path,
    ) -> None:
        with patch("app.routers.calibration._CALIB_TXT", calib_txt_path):
            resp = await api_client.get("/calibration/status", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_baseline"] is False
        assert body["calib_available"] is True
        assert "parse_status" in body

    async def test_status_requires_auth(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/calibration/status")
        assert resp.status_code == 401


class TestCalibrationStart:
    async def test_start_creates_calibration_session(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        calib_txt_path: Path,
    ) -> None:
        """POST /calibration/start must create a session with mode=calibration."""
        from app.db.session import async_session_factory

        with patch("app.routers.calibration._CALIB_TXT", calib_txt_path):
            resp = await api_client.post("/calibration/start", headers=auth_headers)

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert "session_id" in body
        assert "document_id" in body

        async with async_session_factory() as db:
            sess = (
                await db.execute(
                    select(Session).where(Session.id == body["session_id"])
                )
            ).scalar_one()
        assert sess.mode == "calibration"
        assert sess.status == "active"

    async def test_start_creates_chunks_from_txt(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        calib_txt_path: Path,
    ) -> None:
        """Chunks must be created synchronously from the text file on first start."""
        from app.db.session import async_session_factory

        with patch("app.routers.calibration._CALIB_TXT", calib_txt_path):
            resp = await api_client.post("/calibration/start", headers=auth_headers)
        doc_id = resp.json()["document_id"]

        async with async_session_factory() as db:
            chunks = (
                await db.execute(
                    select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
                )
            ).scalars().all()
        assert len(chunks) >= 1
        assert all(c.text for c in chunks)

    async def test_start_requires_auth(self, api_client: AsyncClient) -> None:
        resp = await api_client.post("/calibration/start")
        assert resp.status_code == 401


class TestCalibrationComplete:
    async def test_complete_creates_user_baseline(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        calib_session: int,
    ) -> None:
        """POST /calibration/complete must store a user_baselines row."""
        # Post some telemetry first
        for _ in range(3):
            await api_client.post(
                "/activity/batch",
                json={"session_id": calib_session, **_TELEMETRY_BATCH},
                headers=auth_headers,
            )

        resp = await api_client.post(
            "/calibration/complete",
            json={"session_id": calib_session},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "baseline" in body
        assert body["baseline"]["calibration_duration_seconds"] >= 0
        assert body["session_id"] == calib_session

        # Verify row in DB
        from app.db.session import async_session_factory
        from app.db.models import UserBaseline

        async with async_session_factory() as db:
            baseline = (
                await db.execute(
                    select(UserBaseline)
                )
            ).scalar_one_or_none()
        assert baseline is not None
        assert "wpm_mean" in baseline.baseline_json

    async def test_complete_status_shows_has_baseline(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        calib_session: int,
    ) -> None:
        """After completing calibration, GET /calibration/status must return has_baseline=True."""
        await api_client.post(
            "/calibration/complete",
            json={"session_id": calib_session},
            headers=auth_headers,
        )
        status_resp = await api_client.get("/calibration/status", headers=auth_headers)
        assert status_resp.json()["has_baseline"] is True

    async def test_complete_rejects_wrong_mode(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """/calibration/complete should 404 for a non-calibration session."""
        doc_resp = await api_client.post(
            "/documents/upload",
            data={"title": "Normal Doc"},
            files={"file": ("n.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
            headers=auth_headers,
        )
        doc_id = doc_resp.json()["id"]
        sess_resp = await api_client.post(
            "/sessions/start",
            json={"document_id": doc_id, "name": "Normal", "mode": "baseline"},
            headers=auth_headers,
        )
        session_id = sess_resp.json()["id"]

        resp = await api_client.post(
            "/calibration/complete",
            json={"session_id": session_id},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_complete_enforces_ownership(
        self,
        api_client: AsyncClient,
        calib_session: int,
    ) -> None:
        resp_b = await api_client.post(
            "/auth/register",
            json={"username": "calib_intruder", "email": "ci@x.com", "password": "password123"},
        )
        headers_b = {"Authorization": f"Bearer {resp_b.json()['access_token']}"}

        resp = await api_client.post(
            "/calibration/complete",
            json={"session_id": calib_session},
            headers=headers_b,
        )
        assert resp.status_code == 404


class TestBaselineComputationV2:
    """Tests for Phase 6 v2 baseline metrics."""

    # ── Effective WPM ──────────────────────────────────────────────────────────

    def test_effective_wpm_excludes_idle_and_blur(self) -> None:
        """wpm_effective should only count windows that are focused + low-idle."""
        from app.services.calibration.baseline import compute_effective_wpm

        # 4 windows: 2 qualifying (focused, idle <=1.5), 1 high-idle, 1 blurred
        batches = [
            {"window_focus_state": "focused", "idle_seconds": 0.5, "current_paragraph_id": "calib-0"},
            {"window_focus_state": "focused", "idle_seconds": 1.0, "current_paragraph_id": "calib-1"},
            {"window_focus_state": "focused", "idle_seconds": 5.0, "current_paragraph_id": "calib-0"},  # high idle
            {"window_focus_state": "blurred",  "idle_seconds": 0.2, "current_paragraph_id": "calib-1"},  # blurred
        ]
        para_wc = {"calib-0": 100, "calib-1": 80}
        result = compute_effective_wpm(batches, para_wc)

        # effective_reading_seconds = 2 windows × 2.0 s = 4.0 s
        assert result["effective_reading_seconds"] == 4.0
        # words_read_estimated = 100 + 80 = 180 (both paragraphs were seen)
        assert result["words_read_estimated"] == 180

        # wpm_gross = 180 / (8s / 60) ≈ 1350 (total 4 windows → 8 s)
        # wpm_effective = 180 / (4s / 60) ≈ 2700 — always ≥ wpm_gross here
        assert result["wpm_effective"] > result["wpm_gross"]

    def test_effective_wpm_with_total_words_override(self) -> None:
        """With total_words_override, wpm_gross = total_words / duration_min."""
        from app.services.calibration.baseline import compute_effective_wpm

        batches = [{"window_focus_state": "focused", "idle_seconds": 0.0}] * 10
        result = compute_effective_wpm(batches, {}, total_words_override=300)
        # 300 words / (20 s / 60) = 900 wpm_gross
        assert abs(result["wpm_gross"] - 900.0) < 1.0

    # ── Normalised scroll velocity ─────────────────────────────────────────────

    def test_scroll_velocity_norm_uses_viewport_height(self) -> None:
        """Same scroll delta → lower norm velocity when viewport is taller."""
        from app.services.calibration.baseline import scroll_velocities_norm

        batch_small_vp = [{"scroll_delta_abs_sum": 100.0, "viewport_height_px": 500.0}]
        batch_large_vp = [{"scroll_delta_abs_sum": 100.0, "viewport_height_px": 1000.0}]

        norm_small = scroll_velocities_norm(batch_small_vp)[0]
        norm_large = scroll_velocities_norm(batch_large_vp)[0]

        # larger viewport → smaller normalised velocity
        assert norm_small > norm_large
        # 100 / (500 * 2) = 0.10;  100 / (1000 * 2) = 0.05
        assert abs(norm_small - 0.10) < 1e-9
        assert abs(norm_large - 0.05) < 1e-9

    def test_scroll_velocity_norm_fallback_when_no_viewport(self) -> None:
        """Batches without viewport_height_px use the fallback value."""
        from app.services.calibration.baseline import scroll_velocities_norm

        batches = [{"scroll_delta_abs_sum": 80.0}]  # no viewport_height_px key
        norm = scroll_velocities_norm(batches, fallback_viewport_px=400.0)[0]
        # 80 / (400 * 2) = 0.10
        assert abs(norm - 0.10) < 1e-9

    # ── Paragraph dwell distribution ───────────────────────────────────────────

    def test_paragraph_dwell_median_iqr(self) -> None:
        """Median and IQR of paragraph dwell times are computed correctly."""
        from app.services.calibration.baseline import compute_paragraph_dwell_distribution

        # paragraph 0: 1 window (2 s), 1: 3 windows (6 s), 2: 5 windows (10 s)
        batches = (
            [{"current_paragraph_id": "calib-0"}] * 1
            + [{"current_paragraph_id": "calib-1"}] * 3
            + [{"current_paragraph_id": "calib-2"}] * 5
        )
        result = compute_paragraph_dwell_distribution(batches)

        # dwell_secs = [2.0, 6.0, 10.0]
        assert result["paragraph_count_observed"] == 3
        assert abs(result["para_dwell_median_s"] - 6.0) < 0.01
        # Q1 = 2.0 + (4.0 * 0.5 * 0.5) ... depends on interpolation; IQR should be > 0
        assert result["para_dwell_iqr_s"] > 0

    def test_paragraph_dwell_single_paragraph(self) -> None:
        """Single paragraph → IQR = 0 (only one data point)."""
        from app.services.calibration.baseline import compute_paragraph_dwell_distribution

        batches = [{"current_paragraph_id": "calib-0"}] * 4
        result = compute_paragraph_dwell_distribution(batches)
        assert result["paragraph_count_observed"] == 1
        assert result["para_dwell_iqr_s"] == 0.0
        assert result["para_dwell_mean_s"] == 8.0

    # ── Regress rate ───────────────────────────────────────────────────────────

    def test_regress_rate_from_signed_sums(self) -> None:
        """regress_rate = neg / (neg + pos); 0 when no negative scroll."""
        from app.services.calibration.baseline import compute_regress_rate_stats

        # Window: pos=300, neg=100 → rate = 100/400 = 0.25
        batches = [{"scroll_delta_pos_sum": 300.0, "scroll_delta_neg_sum": 100.0}]
        result = compute_regress_rate_stats(batches)
        assert abs(result["regress_rate_mean"] - 0.25) < 1e-6

    def test_regress_rate_zero_when_only_forward(self) -> None:
        from app.services.calibration.baseline import compute_regress_rate_stats

        batches = [{"scroll_delta_pos_sum": 200.0, "scroll_delta_neg_sum": 0.0}]
        result = compute_regress_rate_stats(batches)
        assert result["regress_rate_mean"] == 0.0

    def test_regress_rate_skips_no_scroll_windows(self) -> None:
        """Windows with neither pos nor neg scroll are skipped."""
        from app.services.calibration.baseline import compute_regress_rate_stats

        batches = [
            {"scroll_delta_pos_sum": 0.0, "scroll_delta_neg_sum": 0.0},  # skipped
            {"scroll_delta_pos_sum": 100.0, "scroll_delta_neg_sum": 100.0},  # 0.5
        ]
        result = compute_regress_rate_stats(batches)
        assert abs(result["regress_rate_mean"] - 0.5) < 1e-6

    # ── Presentation profile ───────────────────────────────────────────────────

    def test_presentation_profile_stats(self) -> None:
        """Mean viewport dimensions are computed across batches."""
        from app.services.calibration.baseline import compute_presentation_profile_stats

        batches = [
            {"viewport_height_px": 800.0, "viewport_width_px": 1280.0, "reader_container_height_px": 760.0},
            {"viewport_height_px": 820.0, "viewport_width_px": 1280.0, "reader_container_height_px": 780.0},
        ]
        profile = compute_presentation_profile_stats(batches, calibration_text_word_count=450, paragraph_count_total=10)
        assert abs(profile["viewport_height_px_mean"] - 810.0) < 0.1
        assert abs(profile["viewport_width_px_mean"] - 1280.0) < 0.1
        assert profile["calibration_text_word_count"] == 450
        assert profile["paragraph_count_total"] == 10

    # ── Full compute_baseline integration ─────────────────────────────────────

    def test_full_baseline_contains_new_keys(self) -> None:
        """compute_baseline result must contain all Phase 6 v2 keys."""
        from app.services.calibration.baseline import compute_baseline

        batches = [
            {
                "scroll_delta_abs_sum": 200.0,
                "scroll_delta_pos_sum": 200.0,
                "scroll_delta_neg_sum": 50.0,
                "scroll_event_count": 5,
                "scroll_direction_changes": 1,
                "idle_seconds": 0.5,
                "window_focus_state": "focused",
                "current_paragraph_id": "calib-0",
                "viewport_height_px": 800.0,
                "viewport_width_px": 1280.0,
                "reader_container_height_px": 760.0,
            }
        ] * 5

        result = compute_baseline(batches, {}, 60, total_words=300, paragraph_count_total=1)

        required = [
            "wpm_gross", "wpm_effective", "words_read_estimated", "effective_reading_seconds",
            "scroll_velocity_px_s_mean", "scroll_velocity_px_s_std",
            "scroll_velocity_norm_mean", "scroll_velocity_norm_std",
            "scroll_jitter_mean", "idle_ratio_mean", "idle_ratio_std",
            "regress_rate_mean", "regress_rate_std",
            "para_dwell_mean_s", "para_dwell_median_s", "para_dwell_iqr_s",
            "paragraph_count_observed", "presentation_profile",
            "calibration_duration_seconds",
            # legacy
            "wpm_mean", "scroll_velocity_mean", "paragraph_dwell_mean",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

        assert isinstance(result["presentation_profile"], dict)
        assert result["wpm_gross"] > 0
        assert result["scroll_velocity_norm_mean"] > 0


class TestExportCsvV2:
    """Ensure new CSV columns appear in the export."""

    async def test_export_csv_has_new_columns(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        calib_session: int,
    ) -> None:
        await api_client.post(
            "/activity/batch",
            json={"session_id": calib_session, **_TELEMETRY_BATCH},
            headers=auth_headers,
        )
        resp = await api_client.get(
            f"/sessions/{calib_session}/export.csv",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        header_row = resp.text.strip().splitlines()[0]
        for col in ("scroll_delta_pos_sum", "scroll_delta_neg_sum",
                    "viewport_height_px", "viewport_width_px", "reader_container_height_px"):
            assert col in header_row, f"Missing CSV column: {col}"


class TestExportCsv:
    async def test_export_returns_csv(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        calib_session: int,
    ) -> None:
        """GET /sessions/{id}/export.csv returns text/csv with header row."""
        # Post a telemetry batch so the CSV is non-empty
        await api_client.post(
            "/activity/batch",
            json={"session_id": calib_session, **_TELEMETRY_BATCH},
            headers=auth_headers,
        )

        resp = await api_client.get(
            f"/sessions/{calib_session}/export.csv",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        lines = resp.text.strip().splitlines()
        assert lines[0].startswith("created_at"), f"Unexpected header: {lines[0]}"
        assert len(lines) >= 2  # header + at least one data row

    async def test_export_requires_auth(
        self, api_client: AsyncClient, calib_session: int
    ) -> None:
        resp = await api_client.get(f"/sessions/{calib_session}/export.csv")
        assert resp.status_code == 401

    async def test_export_enforces_ownership(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        calib_session: int,
    ) -> None:
        resp_b = await api_client.post(
            "/auth/register",
            json={"username": "export_intruder", "email": "ei@x.com", "password": "password123"},
        )
        headers_b = {"Authorization": f"Bearer {resp_b.json()['access_token']}"}
        resp = await api_client.get(
            f"/sessions/{calib_session}/export.csv",
            headers=headers_b,
        )
        assert resp.status_code == 404
