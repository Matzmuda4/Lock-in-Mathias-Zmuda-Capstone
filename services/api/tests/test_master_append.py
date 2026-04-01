"""
Tests for the master training JSONL append pipeline.

Covers:
- Unit tests for master_append.py (pure file I/O, no DB)
  - append produces SESSION_START / SESSION_END sentinels
  - data lines are valid JSON with correct key format
  - CHUNK_BREAK sentinels appear at every 20 packets
  - baseline file is written / overwritten
  - deduplication index prevents re-appending packets
  - append_to_master=False leaves master file untouched
- Integration tests via the API endpoint
  - GET /sessions/{id}/export/bundle?append_to_master=1
    - response includes master_append object
    - master_jsonl_path exists on disk
    - baseline file written
    - appended_packet_count matches state_packet_count
  - Re-exporting same session with append_to_master=1 skips all packets
  - Without append_to_master param, master_append field is null
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.services.training_export.master_append import (
    _CHUNK_SIZE,
    MasterAppendResult,
    append_session_to_master,
)

# ── Shared minimal PDF fixture ────────────────────────────────────────────────

_MINIMAL_PDF = (
    b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_export_folder(
    tmp_path: Path,
    *,
    user_id: int = 1,
    session_id: int = 10,
    packet_count: int = 5,
    baseline_valid: bool = True,
) -> Path:
    """Build a fake per-session export folder in tmp_path."""
    folder = tmp_path / f"session_{session_id}"
    folder.mkdir()

    # session_meta.json
    meta = {
        "session_id": session_id,
        "user_id": user_id,
        "document_id": 42,
        "mode": "adaptive",
        "started_at": "2026-02-01T10:00:00+00:00",
        "ended_at": "2026-02-01T10:30:00+00:00",
        "protocol_tag": "test_run",
    }
    (folder / "session_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # baseline.json
    baseline = {
        "user_id": user_id,
        "baseline_valid": baseline_valid,
        "baseline_updated_at": "2026-01-31T09:00:00+00:00",
        "baseline": {"wpm_effective": 220.0, "idle_ratio_mean": 0.30},
    }
    (folder / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")

    # state_packets.jsonl
    lines = []
    for seq in range(packet_count):
        entry = {
            "session_id": session_id,
            "created_at": f"2026-02-01T10:0{seq}:00+00:00",
            "packet_seq": seq,
            "window_start_at": f"2026-02-01T10:0{seq}:00+00:00",
            "window_end_at": f"2026-02-01T10:0{seq}:30+00:00",
            "packet": {
                "drift": {"ema": 0.1, "level": "focused"},
                "features": {"idle_ratio_mean": 0.25},
                "z_scores": {"z_idle": 0.0},
                "ui_aggregates": {"ui_read_main_share_30s": 1.0},
                "baseline_snapshot": {"wpm_effective": 220.0} if baseline_valid else None,
            },
        }
        lines.append(json.dumps(entry))
    (folder / "state_packets.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return folder


def _read_master_lines(master_dir: Path) -> list[str]:
    path = master_dir / "unlabelled.jsonl"
    if not path.exists():
        return []
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _is_sentinel(line: str) -> bool:
    return line.startswith("# ")


def _data_lines(lines: list[str]) -> list[str]:
    return [l for l in lines if not _is_sentinel(l)]


def _sentinel_lines(lines: list[str]) -> list[str]:
    return [l for l in lines if _is_sentinel(l)]


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — pure Python, no DB, no HTTP
# ═══════════════════════════════════════════════════════════════════════════════


class TestMasterAppendUnit:
    def test_basic_append_creates_master_file(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=1, session_id=10, packet_count=3)
        master_dir = tmp_path / "TrainingData"

        result = append_session_to_master(
            export_folder=folder,
            master_dir=master_dir,
            user_id=1,
            session_id=10,
        )

        assert result.master_jsonl_path.exists()
        assert result.appended_packet_count == 3
        assert result.skipped_packet_count == 0

    def test_sentinel_structure(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=1, session_id=10, packet_count=3)
        master_dir = tmp_path / "TrainingData"
        append_session_to_master(folder, master_dir, user_id=1, session_id=10)

        lines = _read_master_lines(master_dir)
        sentinels = _sentinel_lines(lines)

        assert any(s.startswith("# SESSION_START") for s in sentinels), "missing SESSION_START"
        assert any(s.startswith("# SESSION_END") for s in sentinels), "missing SESSION_END"

    def test_data_lines_are_valid_json(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=2, session_id=20, packet_count=5)
        master_dir = tmp_path / "TrainingData"
        append_session_to_master(folder, master_dir, user_id=2, session_id=20)

        data = _data_lines(_read_master_lines(master_dir))
        assert len(data) == 5
        for line in data:
            obj = json.loads(line)  # must not raise
            assert "key" in obj
            assert "drift" in obj
            assert "features" in obj
            assert "z_scores" in obj
            assert "packet_raw" in obj

    def test_key_format(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=3, session_id=30, packet_count=2)
        master_dir = tmp_path / "TrainingData"
        append_session_to_master(folder, master_dir, user_id=3, session_id=30)

        data = _data_lines(_read_master_lines(master_dir))
        keys = [json.loads(l)["key"] for l in data]
        assert keys == ["u3_s30_p0", "u3_s30_p1"]

    def test_chunk_break_sentinels(self, tmp_path: Path) -> None:
        # With 25 packets: CHUNK_BREAK should appear once (before packet at index 20)
        folder = _make_export_folder(
            tmp_path, user_id=1, session_id=10, packet_count=25
        )
        master_dir = tmp_path / "TrainingData"
        append_session_to_master(folder, master_dir, user_id=1, session_id=10)

        sentinels = _sentinel_lines(_read_master_lines(master_dir))
        chunk_breaks = [s for s in sentinels if s.startswith("# CHUNK_BREAK")]
        assert len(chunk_breaks) == 1

        # With 41 packets (indices 0-40): CHUNK_BREAKs at i=20 and i=40 → 2 total
        folder2 = _make_export_folder(
            tmp_path, user_id=4, session_id=40, packet_count=41
        )
        master_dir2 = tmp_path / "TrainingData2"
        append_session_to_master(folder2, master_dir2, user_id=4, session_id=40)
        sentinels2 = _sentinel_lines(_read_master_lines(master_dir2))
        chunk_breaks2 = [s for s in sentinels2 if s.startswith("# CHUNK_BREAK")]
        assert len(chunk_breaks2) == 2

    def test_no_chunk_break_below_chunk_size(self, tmp_path: Path) -> None:
        folder = _make_export_folder(
            tmp_path, user_id=1, session_id=10, packet_count=_CHUNK_SIZE
        )
        master_dir = tmp_path / "TrainingData"
        append_session_to_master(folder, master_dir, user_id=1, session_id=10)

        sentinels = _sentinel_lines(_read_master_lines(master_dir))
        chunk_breaks = [s for s in sentinels if s.startswith("# CHUNK_BREAK")]
        assert len(chunk_breaks) == 0

    def test_baseline_file_written(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=5, session_id=50, packet_count=2)
        master_dir = tmp_path / "TrainingData"
        result = append_session_to_master(folder, master_dir, user_id=5, session_id=50)

        assert result.baseline_path.exists()
        baseline = json.loads(result.baseline_path.read_text(encoding="utf-8"))
        assert baseline["user_id"] == 5
        assert baseline["baseline_valid"] is True

    def test_baseline_file_overwritten_on_re_export(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=5, session_id=50, packet_count=1)
        master_dir = tmp_path / "TrainingData"
        append_session_to_master(folder, master_dir, user_id=5, session_id=50)

        # Mutate baseline in folder and re-export
        baseline_file = folder / "baseline.json"
        data = json.loads(baseline_file.read_text(encoding="utf-8"))
        data["baseline"]["wpm_effective"] = 999.0
        baseline_file.write_text(json.dumps(data), encoding="utf-8")

        # Need a new session to trigger fresh append (different session_id, same user)
        folder2 = _make_export_folder(tmp_path, user_id=5, session_id=51, packet_count=1)
        baseline_file2 = folder2 / "baseline.json"
        data2 = json.loads(baseline_file2.read_text(encoding="utf-8"))
        data2["baseline"]["wpm_effective"] = 999.0
        baseline_file2.write_text(json.dumps(data2), encoding="utf-8")

        append_session_to_master(folder2, master_dir, user_id=5, session_id=51)
        written = json.loads(
            (master_dir / "baselines" / "user_5_baseline.json").read_text(encoding="utf-8")
        )
        assert written["baseline"]["wpm_effective"] == 999.0

    def test_deduplication_no_double_append(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=1, session_id=10, packet_count=5)
        master_dir = tmp_path / "TrainingData"

        r1 = append_session_to_master(folder, master_dir, user_id=1, session_id=10)
        r2 = append_session_to_master(folder, master_dir, user_id=1, session_id=10)

        assert r1.appended_packet_count == 5
        assert r2.appended_packet_count == 0
        assert r2.skipped_packet_count == 5

        # File must contain exactly 5 data lines (from first append only)
        data = _data_lines(_read_master_lines(master_dir))
        assert len(data) == 5

    def test_deduplication_appends_only_new_packets(self, tmp_path: Path) -> None:
        # First append: 3 packets (seq 0, 1, 2)
        folder = _make_export_folder(tmp_path, user_id=1, session_id=10, packet_count=3)
        master_dir = tmp_path / "TrainingData"
        r1 = append_session_to_master(folder, master_dir, user_id=1, session_id=10)
        assert r1.appended_packet_count == 3

        # Add 2 more packets to the jsonl (seq 3, 4)
        packets_file = folder / "state_packets.jsonl"
        existing = packets_file.read_text(encoding="utf-8")
        for seq in [3, 4]:
            entry = {
                "session_id": 10,
                "created_at": f"2026-02-01T10:0{seq}:00+00:00",
                "packet_seq": seq,
                "window_start_at": None,
                "window_end_at": None,
                "packet": {"drift": {}, "features": {}, "z_scores": {}, "ui_aggregates": {}},
            }
            existing += json.dumps(entry) + "\n"
        packets_file.write_text(existing, encoding="utf-8")

        r2 = append_session_to_master(folder, master_dir, user_id=1, session_id=10)
        assert r2.appended_packet_count == 2
        assert r2.skipped_packet_count == 3

        # Total data lines = 5
        data = _data_lines(_read_master_lines(master_dir))
        assert len(data) == 5

    def test_empty_packets_skips_master_write(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=1, session_id=10, packet_count=0)
        master_dir = tmp_path / "TrainingData"
        result = append_session_to_master(folder, master_dir, user_id=1, session_id=10)

        assert result.appended_packet_count == 0
        # Master file should not exist (or be empty) — no data to write
        master_path = master_dir / "unlabelled.jsonl"
        if master_path.exists():
            lines = _data_lines(_read_master_lines(master_dir))
            assert len(lines) == 0

    def test_baseline_embedded_in_packet_flag(self, tmp_path: Path) -> None:
        # With embedded baseline_snapshot in packets
        folder = _make_export_folder(
            tmp_path, user_id=1, session_id=10, packet_count=2, baseline_valid=True
        )
        master_dir = tmp_path / "TrainingData"
        result = append_session_to_master(folder, master_dir, user_id=1, session_id=10)
        assert result.baseline_embedded_in_packet is True

        # Without embedded baseline_snapshot (baseline_valid=False causes None snapshot)
        folder2 = _make_export_folder(
            tmp_path, user_id=2, session_id=20, packet_count=2, baseline_valid=False
        )
        master_dir2 = tmp_path / "TrainingData2"
        result2 = append_session_to_master(folder2, master_dir2, user_id=2, session_id=20)
        assert result2.baseline_embedded_in_packet is False

    def test_session_start_header_fields(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=7, session_id=70, packet_count=1)
        master_dir = tmp_path / "TrainingData"
        append_session_to_master(folder, master_dir, user_id=7, session_id=70)

        sentinels = _sentinel_lines(_read_master_lines(master_dir))
        start = next(s for s in sentinels if s.startswith("# SESSION_START"))
        header = json.loads(start[len("# SESSION_START "):])

        assert header["user_id"] == 7
        assert header["session_id"] == 70
        assert header["session_mode"] == "adaptive"
        assert header["protocol_tag"] == "test_run"
        assert header["baseline_valid"] is True
        assert "baseline_ref" in header

    def test_baseline_ref_format(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=9, session_id=90, packet_count=1)
        master_dir = tmp_path / "TrainingData"
        result = append_session_to_master(folder, master_dir, user_id=9, session_id=90)
        assert result.baseline_ref == "TrainingData/baselines/user_9_baseline.json"

    def test_multiple_sessions_append_independently(self, tmp_path: Path) -> None:
        master_dir = tmp_path / "TrainingData"
        for sid in [100, 101, 102]:
            folder = _make_export_folder(
                tmp_path, user_id=1, session_id=sid, packet_count=3
            )
            append_session_to_master(folder, master_dir, user_id=1, session_id=sid)

        data = _data_lines(_read_master_lines(master_dir))
        assert len(data) == 9  # 3 sessions × 3 packets

    def test_append_index_file_written(self, tmp_path: Path) -> None:
        folder = _make_export_folder(tmp_path, user_id=1, session_id=10, packet_count=4)
        master_dir = tmp_path / "TrainingData"
        append_session_to_master(folder, master_dir, user_id=1, session_id=10)

        index_path = master_dir / "_append_index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text(encoding="utf-8"))
        assert index["10"] == 3  # max packet_seq for session 10 (0-indexed, 4 packets → max 3)

    def test_missing_export_files_handled_gracefully(self, tmp_path: Path) -> None:
        # Empty folder — no session_meta, no baseline, no packets
        empty_folder = tmp_path / "empty_session"
        empty_folder.mkdir()
        master_dir = tmp_path / "TrainingData"

        result = append_session_to_master(
            empty_folder, master_dir, user_id=1, session_id=999
        )
        assert result.appended_packet_count == 0
        assert result.baseline_path.exists()  # baseline file is always written


# ═══════════════════════════════════════════════════════════════════════════════
# API integration tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
async def doc_id_ma(api_client: AsyncClient, auth_headers: dict) -> int:
    resp = await api_client.post(
        "/documents/upload",
        data={"title": "Master Append Test Doc"},
        files={"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


@pytest.fixture()
async def session_id_ma(
    api_client: AsyncClient, auth_headers: dict, doc_id_ma: int
) -> int:
    resp = await api_client.post(
        "/sessions/start",
        json={"document_id": doc_id_ma, "name": "Master Append Session", "mode": "adaptive"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


class TestMasterAppendEndpoint:
    async def test_append_to_master_false_returns_null_field(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_id_ma: int,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr(
            "app.services.exports.service.settings",
            type("S", (), {"training_exports_dir": tmp_path, "app_version": "0.1.0"})(),
        )
        resp = await api_client.get(
            f"/sessions/{session_id_ma}/export/bundle",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["master_append"] is None

    async def test_append_to_master_true_returns_info(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_id_ma: int,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        master_dir = tmp_path / "TrainingData"
        monkeypatch.setattr(
            "app.services.exports.service.settings",
            type("S", (), {"training_exports_dir": tmp_path, "app_version": "0.1.0"})(),
        )
        monkeypatch.setattr(
            "app.routers.exports.settings",
            type("S", (), {
                "training_exports_dir": tmp_path,
                "training_master_dir": master_dir,
                "app_version": "0.1.0",
            })(),
        )

        resp = await api_client.get(
            f"/sessions/{session_id_ma}/export/bundle?append_to_master=1",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["master_append"] is not None
        ma = body["master_append"]
        assert "master_jsonl_path" in ma
        assert "appended_packet_count" in ma
        assert "skipped_packet_count" in ma
        assert "baseline_path" in ma
        assert "baseline_ref" in ma
        assert "baseline_embedded_in_packet" in ma

    async def test_append_counts_match_state_packet_count(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_id_ma: int,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        master_dir = tmp_path / "TrainingData"
        monkeypatch.setattr(
            "app.services.exports.service.settings",
            type("S", (), {"training_exports_dir": tmp_path, "app_version": "0.1.0"})(),
        )
        monkeypatch.setattr(
            "app.routers.exports.settings",
            type("S", (), {
                "training_exports_dir": tmp_path,
                "training_master_dir": master_dir,
                "app_version": "0.1.0",
            })(),
        )

        resp = await api_client.get(
            f"/sessions/{session_id_ma}/export/bundle?append_to_master=1",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        state_count = body["state_packet_count"]
        appended = body["master_append"]["appended_packet_count"]
        # appended_packet_count must equal state_packet_count
        assert appended == state_count

    async def test_re_export_skips_already_appended(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_id_ma: int,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        master_dir = tmp_path / "TrainingData"
        fake_settings = type("S", (), {
            "training_exports_dir": tmp_path,
            "training_master_dir": master_dir,
            "app_version": "0.1.0",
        })()
        monkeypatch.setattr("app.services.exports.service.settings", fake_settings)
        monkeypatch.setattr("app.routers.exports.settings", fake_settings)

        url = f"/sessions/{session_id_ma}/export/bundle?append_to_master=1"

        r1 = await api_client.get(url, headers=auth_headers)
        r2 = await api_client.get(url, headers=auth_headers)
        assert r1.status_code == 200
        assert r2.status_code == 200

        first_appended = r1.json()["master_append"]["appended_packet_count"]
        second_appended = r2.json()["master_append"]["appended_packet_count"]
        second_skipped = r2.json()["master_append"]["skipped_packet_count"]

        assert second_appended == 0
        assert second_skipped == first_appended

    async def test_baseline_file_exists_after_append(
        self,
        api_client: AsyncClient,
        auth_headers: dict,
        session_id_ma: int,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        master_dir = tmp_path / "TrainingData"
        fake_settings = type("S", (), {
            "training_exports_dir": tmp_path,
            "training_master_dir": master_dir,
            "app_version": "0.1.0",
        })()
        monkeypatch.setattr("app.services.exports.service.settings", fake_settings)
        monkeypatch.setattr("app.routers.exports.settings", fake_settings)

        resp = await api_client.get(
            f"/sessions/{session_id_ma}/export/bundle?append_to_master=1",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        baseline_path = Path(resp.json()["master_append"]["baseline_path"])
        assert baseline_path.exists()
        content = json.loads(baseline_path.read_text(encoding="utf-8"))
        # baseline.json must contain a user_id field
        assert "user_id" in content
