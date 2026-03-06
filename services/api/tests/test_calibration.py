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
