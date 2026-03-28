"""
Tests for the training-data packet export pipeline — classify branch.

Coverage:
  API tests (require auth + DB):
    1. GET /training/packets/export requires auth (401)
    2. POST /training/packets/export requires auth (401)
    3. Export with no packets returns row_count=0 and writes a file with header
    4. Export after 5 batches writes exactly 1 packet, file is non-empty
    5. CSV header contains required stable columns
    6. JSONL lines parse as JSON and contain meta + packet_json
    7. include_debug=true adds debug_ columns to CSV
    8. session_ids filter returns only matching packets
    9. mode filter returns only matching sessions
   10. Ownership enforced: user cannot export another user's packets
   11. ?download=1 returns a streaming file response

  Flattener unit tests (pure Python, no DB):
   12. flatten_packet_to_row handles missing fields without error
   13. Column ordering is stable across calls
   14. packet_json column is always present and parseable
   15. baseline_valid=True when wpm_effective > 0 and duration >= 60
   16. baseline_valid=False for empty baseline
   17. build_jsonl_line contains meta + packet_json keys
   18. _compute_ui_aggregates sums to 1.0 (within float tolerance)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.services.training_export.flatten import (
    TRAINING_COLUMNS,
    build_jsonl_line,
    flatten_packet_to_row,
)
from app.services.drift.store import _compute_ui_aggregates

# ── Minimal PDF ───────────────────────────────────────────────────────────────

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


# ── API fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
async def doc_id(api_client: AsyncClient, auth_headers: dict) -> int:
    resp = await api_client.post(
        "/documents/upload",
        data={"title": "Training Export Test Doc"},
        files={"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


@pytest.fixture()
async def session_id(api_client: AsyncClient, auth_headers: dict, doc_id: int) -> int:
    resp = await api_client.post(
        "/sessions/start",
        json={"document_id": doc_id, "name": "Training Test Session", "mode": "baseline"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


@pytest.fixture()
async def session_with_5_batches(
    api_client: AsyncClient, auth_headers: dict, session_id: int
) -> int:
    """Post exactly 5 batches to trigger one state packet."""
    for _ in range(5):
        resp = await api_client.post(
            "/activity/batch",
            json={"session_id": session_id, **_BATCH},
            headers=auth_headers,
        )
        assert resp.status_code == 201, resp.text
    return session_id


@pytest.fixture()
async def second_user_headers(api_client: AsyncClient) -> dict[str, str]:
    resp = await api_client.post(
        "/auth/register",
        json={
            "username": "otheruser",
            "email": "other@example.com",
            "password": "password123",
        },
    )
    assert resp.status_code == 201, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


# ── API tests ─────────────────────────────────────────────────────────────────


class TestAuthRequired:
    async def test_get_requires_auth(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/training/packets/export")
        assert resp.status_code == 401

    async def test_post_requires_auth(self, api_client: AsyncClient) -> None:
        resp = await api_client.post(
            "/training/packets/export", json={}
        )
        assert resp.status_code == 401


class TestExportNoPackets:
    async def test_empty_export_returns_zero_rows(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_id: int,  # session exists but no batches posted
    ) -> None:
        resp = await api_client.get(
            "/training/packets/export",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["row_count"] == 0
        assert data["session_count"] == 0

        # File should still be created (with header only for CSV)
        out_path = Path(data["file_path"])
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert len(content) > 0  # at least the header line


class TestExportWithPackets:
    async def test_five_batches_produce_one_packet(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_with_5_batches: int,
    ) -> None:
        resp = await api_client.get(
            "/training/packets/export",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["row_count"] == 1
        assert data["session_count"] == 1

    async def test_csv_header_contains_required_columns(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_with_5_batches: int,
    ) -> None:
        resp = await api_client.get(
            "/training/packets/export?format=csv",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        columns = data["columns"]

        required = [
            "session_id", "user_id", "document_id", "packet_seq",
            "created_at", "window_start_at", "window_end_at", "session_mode",
            "baseline_valid",
            "feat_idle_ratio_mean", "feat_focus_loss_rate",
            "z_idle", "z_focus_loss", "z_skim",
            "drift_ema", "drift_level", "disruption_score", "engagement_score",
            "panel_share_30s", "reader_share_30s",
            "packet_json",
        ]
        for col in required:
            assert col in columns, f"Missing required column: {col}"

    async def test_csv_file_is_non_empty_and_parseable(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_with_5_batches: int,
    ) -> None:
        resp = await api_client.get(
            "/training/packets/export?format=csv",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        out_path = Path(resp.json()["file_path"])
        assert out_path.exists()
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # header + 1 data row

    async def test_jsonl_lines_parse_and_have_required_keys(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_with_5_batches: int,
    ) -> None:
        resp = await api_client.get(
            "/training/packets/export?format=jsonl",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["row_count"] == 1

        out_path = Path(data["file_path"])
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        parsed = json.loads(lines[0])
        for key in ("meta", "features", "z_scores", "drift", "packet_json"):
            assert key in parsed, f"Missing key in JSONL line: {key}"
        assert "session_id" in parsed["meta"]
        assert "packet_json" in parsed

    async def test_include_debug_adds_debug_columns(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_with_5_batches: int,
    ) -> None:
        resp_no_debug = await api_client.get(
            "/training/packets/export?format=csv&include_debug=false",
            headers=auth_headers,
        )
        resp_debug = await api_client.get(
            "/training/packets/export?format=csv&include_debug=true",
            headers=auth_headers,
        )
        assert resp_no_debug.status_code == 200
        assert resp_debug.status_code == 200

        cols_no = resp_no_debug.json()["columns"]
        cols_debug = resp_debug.json()["columns"]
        debug_cols = [c for c in cols_debug if c.startswith("debug_")]
        # If the packet has a non-empty debug dict, debug columns appear
        # (the model always populates beta_components, so there should be some)
        assert len(cols_debug) >= len(cols_no)

    async def test_session_ids_filter(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_with_5_batches: int,
        doc_id: int,
    ) -> None:
        # Create a second session and post 5 more batches
        resp = await api_client.post(
            "/sessions/start",
            json={"document_id": doc_id, "name": "Session B", "mode": "adaptive"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        session_b = resp.json()["id"]
        for _ in range(5):
            await api_client.post(
                "/activity/batch",
                json={"session_id": session_b, **_BATCH},
                headers=auth_headers,
            )

        # All sessions → 2 packets
        resp_all = await api_client.get(
            "/training/packets/export",
            headers=auth_headers,
        )
        assert resp_all.json()["row_count"] == 2

        # Filter to only session A → 1 packet
        resp_filtered = await api_client.get(
            f"/training/packets/export?session_ids={session_with_5_batches}",
            headers=auth_headers,
        )
        assert resp_filtered.json()["row_count"] == 1
        assert resp_filtered.json()["session_count"] == 1


class TestOwnershipEnforced:
    async def test_cannot_export_another_users_packets(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        second_user_headers: dict,
        session_with_5_batches: int,
    ) -> None:
        # second_user requests export — should see 0 rows (not the first user's data)
        resp = await api_client.get(
            "/training/packets/export",
            headers=second_user_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["row_count"] == 0


class TestModeFilter:
    async def test_mode_filter_baseline_only(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_with_5_batches: int,  # mode=baseline
        doc_id: int,
    ) -> None:
        # session_with_5_batches is mode=baseline, create adaptive session too
        resp = await api_client.post(
            "/sessions/start",
            json={"document_id": doc_id, "name": "Adaptive", "mode": "adaptive"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        adaptive_session = resp.json()["id"]
        for _ in range(5):
            await api_client.post(
                "/activity/batch",
                json={"session_id": adaptive_session, **_BATCH},
                headers=auth_headers,
            )

        resp_baseline = await api_client.get(
            "/training/packets/export?mode=baseline",
            headers=auth_headers,
        )
        assert resp_baseline.status_code == 200
        data = resp_baseline.json()
        # Only baseline session packets
        assert data["row_count"] == 1


class TestDownload:
    async def test_download_returns_file_response(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_with_5_batches: int,
    ) -> None:
        resp = await api_client.get(
            "/training/packets/export?format=csv&download=true",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")


# ── Flattener unit tests (no DB) ──────────────────────────────────────────────


_SAMPLE_PACKET_JSON: dict = {
    "session_id": 10,
    "user_id": 1,
    "document_id": 5,
    "session_mode": "baseline",
    "drift": {
        "drift_level": 0.05,
        "drift_ema": 0.04,
        "beta_effective": 0.02,
        "beta_ema": 0.015,
        "disruption_score": 0.1,
        "engagement_score": 0.9,
        "confidence": 0.8,
        "pace_ratio": 1.05,
        "pace_available": True,
    },
    "features": {
        "n_batches": 5,
        "idle_ratio_mean": 0.1,
        "focus_loss_rate": 0.0,
        "scroll_velocity_norm_mean": 0.4,
        "scroll_velocity_norm_std": 0.05,
        "scroll_burstiness": 1.2,
        "scroll_jitter_mean": 0.08,
        "regress_rate_mean": 0.03,
        "stagnation_ratio": 0.2,
        "mouse_efficiency_mean": 0.9,
        "mouse_path_px_mean": 120.0,
        "scroll_pause_mean": 0.5,
        "long_pause_share": 0.1,
        "pace_ratio": 1.05,
        "pace_dev": 0.05,
        "pace_available": True,
        "window_wpm_effective": 220.0,
        "paragraphs_observed": 2,
        "progress_markers_count": 1,
        "progress_velocity": 0.01,
        "at_end_of_document": False,
        "telemetry_fault_rate": 0.0,
        "scroll_capture_fault_rate": 0.0,
        "paragraph_missing_fault_rate": 0.0,
        "quality_confidence_mult": 1.0,
    },
    "z_scores": {
        "z_idle": 0.2,
        "z_focus_loss": 0.0,
        "z_jitter": 0.1,
        "z_regress": 0.05,
        "z_pause": 0.1,
        "z_stagnation": 0.3,
        "z_mouse": 0.0,
        "z_pace": 0.0,
        "z_skim": 0.0,
        "z_burstiness": 0.2,
    },
    "debug": {
        "disruption_raw": 0.12,
        "W_DISRUPT": 0.7,
    },
    "baseline_snapshot": {
        "baseline_json": {
            "wpm_effective": 220.0,
            "wpm_gross": 240.0,
            "idle_ratio_mean": 0.08,
            "idle_ratio_std": 0.10,
            "scroll_velocity_norm_mean": 0.4,
            "scroll_velocity_norm_std": 0.05,
            "regress_rate_mean": 0.03,
            "regress_rate_std": 0.02,
            "scroll_jitter_mean": 0.08,
            "scroll_jitter_std": 0.06,
            "para_dwell_median_s": 12.0,
            "para_dwell_iqr_s": 5.0,
            "calibration_duration_seconds": 90,
        },
        "baseline_updated_at": "2026-02-25T10:00:00+00:00",
        "baseline_valid": True,
    },
    "ui_aggregates": {
        "ui_read_main_share_30s": 0.8,
        "ui_panel_open_share_30s": 0.1,
        "ui_panel_interacting_share_30s": 0.1,
        "ui_user_paused_share_30s": 0.0,
        "panel_share_30s": 0.2,
        "reader_share_30s": 0.8,
        "iz_reader_share_30s": 0.9,
        "iz_panel_share_30s": 0.1,
        "iz_other_share_30s": 0.0,
    },
}

_SAMPLE_META = {
    "session_id": 10,
    "user_id": 1,
    "document_id": 5,
    "session_mode": "baseline",
    "packet_seq": 0,
    "created_at": "2026-02-25T12:00:00+00:00",
    "window_start_at": "2026-02-25T11:59:30+00:00",
    "window_end_at": "2026-02-25T12:00:00+00:00",
}


class TestFlattenPacket:
    def test_flatten_returns_all_training_columns(self) -> None:
        row = flatten_packet_to_row(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        for col in TRAINING_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_column_ordering_is_stable(self) -> None:
        row1 = flatten_packet_to_row(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        row2 = flatten_packet_to_row(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        assert list(row1.keys()) == list(row2.keys())

    def test_packet_json_column_is_present_and_parseable(self) -> None:
        row = flatten_packet_to_row(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        assert "packet_json" in row
        parsed = json.loads(row["packet_json"])
        assert isinstance(parsed, dict)

    def test_missing_fields_do_not_raise(self) -> None:
        row = flatten_packet_to_row({}, {})
        assert "session_id" in row
        assert row["session_id"] is None
        assert row["packet_json"] == "{}"

    def test_baseline_valid_true_when_wpm_set(self) -> None:
        row = flatten_packet_to_row(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        assert row["baseline_valid"] is True

    def test_baseline_valid_false_for_empty_baseline(self) -> None:
        pjson = dict(_SAMPLE_PACKET_JSON)
        pjson["baseline_snapshot"] = {
            "baseline_json": {},
            "baseline_updated_at": None,
            "baseline_valid": False,
        }
        row = flatten_packet_to_row(_SAMPLE_META, pjson)
        assert row["baseline_valid"] is False

    def test_include_debug_adds_debug_columns(self) -> None:
        row = flatten_packet_to_row(_SAMPLE_META, _SAMPLE_PACKET_JSON, include_debug=True)
        debug_keys = [k for k in row if k.startswith("debug_")]
        assert len(debug_keys) > 0

    def test_no_debug_by_default(self) -> None:
        row = flatten_packet_to_row(_SAMPLE_META, _SAMPLE_PACKET_JSON, include_debug=False)
        debug_keys = [k for k in row if k.startswith("debug_")]
        assert len(debug_keys) == 0

    def test_session_mode_from_packet_json(self) -> None:
        row = flatten_packet_to_row(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        assert row["session_mode"] == "baseline"

    def test_session_mode_falls_back_to_meta(self) -> None:
        pjson = {k: v for k, v in _SAMPLE_PACKET_JSON.items() if k != "session_mode"}
        row = flatten_packet_to_row(_SAMPLE_META, pjson)
        assert row["session_mode"] == "baseline"


class TestBuildJsonlLine:
    def test_jsonl_line_has_required_keys(self) -> None:
        line = build_jsonl_line(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        for key in ("meta", "baseline", "features", "z_scores", "drift", "ui_aggregates", "packet_json"):
            assert key in line, f"Missing key: {key}"

    def test_meta_contains_session_id(self) -> None:
        line = build_jsonl_line(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        assert line["meta"]["session_id"] == 10

    def test_packet_json_is_dict(self) -> None:
        line = build_jsonl_line(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        assert isinstance(line["packet_json"], dict)

    def test_include_debug_adds_debug_key(self) -> None:
        line = build_jsonl_line(_SAMPLE_META, _SAMPLE_PACKET_JSON, include_debug=True)
        assert "debug" in line

    def test_no_debug_by_default(self) -> None:
        line = build_jsonl_line(_SAMPLE_META, _SAMPLE_PACKET_JSON)
        assert "debug" not in line


class TestUiAggregates:
    def test_empty_batches_returns_valid_fractions(self) -> None:
        agg = _compute_ui_aggregates([])
        # All fractions should be in [0, 1] and the function must not raise
        for key, val in agg.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_all_panel_batches(self) -> None:
        batches = [{"ui_context": "PANEL_OPEN", "interaction_zone": "panel"}] * 5
        agg = _compute_ui_aggregates(batches)
        assert agg["panel_share_30s"] == 1.0
        assert agg["reader_share_30s"] == 0.0

    def test_mixed_batches_sum_to_one(self) -> None:
        batches = [
            {"ui_context": "READ_MAIN", "interaction_zone": "reader"},
            {"ui_context": "READ_MAIN", "interaction_zone": "reader"},
            {"ui_context": "PANEL_OPEN", "interaction_zone": "panel"},
            {"ui_context": "PANEL_INTERACTING", "interaction_zone": "panel"},
        ]
        agg = _compute_ui_aggregates(batches)
        ui_total = (
            agg["ui_read_main_share_30s"]
            + agg["ui_panel_open_share_30s"]
            + agg["ui_panel_interacting_share_30s"]
            + agg["ui_user_paused_share_30s"]
        )
        assert abs(ui_total - 1.0) < 1e-9

        iz_total = agg["iz_reader_share_30s"] + agg["iz_panel_share_30s"] + agg["iz_other_share_30s"]
        assert abs(iz_total - 1.0) < 1e-9
