"""
Master training JSONL append service — classify branch.

Reads the already-written per-session export bundle (state_packets.jsonl,
session_meta.json, baseline.json) and appends formatted lines into:

    TrainingData/unlabelled.jsonl

Also writes/overwrites the per-user baseline file:

    TrainingData/baselines/user_{user_id}_baseline.json

File format
-----------
Comment sentinel lines (start with '# ') carry structural metadata.
All data lines are valid JSON objects, one per line.

Sentinel syntax::

    # SESSION_START {...}   — once per session, before first packet
    # CHUNK_BREAK {...}     — before every 20th packet within a session
    # SESSION_END {...}     — once per session, after last packet

Deduplication
-------------
``TrainingData/_append_index.json`` maps session_id (string) →
max_packet_seq_appended.  Re-exporting a session only appends packets
whose ``packet_seq`` is strictly greater than the last recorded value,
so the master file is always append-only and never has duplicate rows.

Thread safety
-------------
All writes are performed under an exclusive ``fcntl.flock`` on
``TrainingData/.append.lock``, preventing interleaved writes from
concurrent export requests.
"""

from __future__ import annotations

import fcntl
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Insert a CHUNK_BREAK sentinel before every Nth packet (at position i > 0,
# where i % _CHUNK_SIZE == 0).
_CHUNK_SIZE = 20


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class MasterAppendResult:
    master_jsonl_path: Path
    baseline_path: Path
    appended_packet_count: int
    skipped_packet_count: int
    baseline_embedded_in_packet: bool
    baseline_ref: str


# ── Index helpers ─────────────────────────────────────────────────────────────


def _load_index(index_path: Path) -> dict[str, int]:
    """Load the deduplication index, returning an empty dict on any failure."""
    if index_path.exists():
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_index(index_path: Path, index: dict[str, int]) -> None:
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


# ── Per-line builder ──────────────────────────────────────────────────────────


def _build_packet_line(
    raw_entry: dict[str, Any],
    user_id: int,
    session_id: int,
    baseline_ref: str,
) -> dict[str, Any]:
    """
    Build one master JSONL data line from a per-session state_packets.jsonl entry.

    The per-session file uses the key ``packet`` for the raw packet dict.
    The master file exposes the structured sub-fields at top level for easy
    LLM parsing and keeps the full original packet as ``packet_raw``.

    Stable key format: ``u{user_id}_s{session_id}_p{packet_seq}``
    """
    pkt: dict[str, Any] = raw_entry.get("packet") or {}
    seq = raw_entry.get("packet_seq", 0)

    return {
        "key": f"u{user_id}_s{session_id}_p{seq}",
        "user_id": user_id,
        "session_id": session_id,
        "packet_seq": seq,
        "created_at": raw_entry.get("created_at"),
        "window_start_at": raw_entry.get("window_start_at"),
        "window_end_at": raw_entry.get("window_end_at"),
        # ── Structured summary fields (human-readable for LLM) ────────────────
        "drift": pkt.get("drift", {}),
        "features": pkt.get("features", {}),
        "z_scores": pkt.get("z_scores", {}),
        "ui_aggregates": pkt.get("ui_aggregates", {}),
        "baseline_snapshot": pkt.get("baseline_snapshot"),
        # ── Provenance ────────────────────────────────────────────────────────
        "baseline_ref": baseline_ref,
        # ── Lossless fallback ─────────────────────────────────────────────────
        "packet_raw": pkt,
    }


# ── Public entry point ────────────────────────────────────────────────────────


def append_session_to_master(
    export_folder: Path,
    master_dir: Path,
    user_id: int,
    session_id: int,
) -> MasterAppendResult:
    """
    Append new state packets from *export_folder* into the master unlabelled.jsonl.

    *export_folder* is the per-session bundle directory written by
    :func:`export_session_bundle` and must contain:

    - ``state_packets.jsonl``
    - ``session_meta.json``
    - ``baseline.json``

    All file I/O is performed under an exclusive ``fcntl.flock`` so concurrent
    export requests never interleave their writes.

    Returns a :class:`MasterAppendResult` describing what was written.
    """
    master_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir = master_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    master_path = master_dir / "unlabelled.jsonl"
    index_path = master_dir / "_append_index.json"
    baseline_path = baselines_dir / f"user_{user_id}_baseline.json"
    baseline_ref = f"TrainingData/baselines/user_{user_id}_baseline.json"

    # ── Read source files ─────────────────────────────────────────────────────

    packet_rows: list[dict[str, Any]] = []
    packets_file = export_folder / "state_packets.jsonl"
    if packets_file.exists():
        for raw_line in packets_file.read_text(encoding="utf-8").splitlines():
            raw_line = raw_line.strip()
            if raw_line:
                try:
                    packet_rows.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    pass

    session_meta_raw: dict[str, Any] = {}
    meta_file = export_folder / "session_meta.json"
    if meta_file.exists():
        try:
            session_meta_raw = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    baseline_raw: dict[str, Any] = {}
    baseline_file = export_folder / "baseline.json"
    if baseline_file.exists():
        try:
            baseline_raw = json.loads(baseline_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    # ── Acquire exclusive file lock ───────────────────────────────────────────

    lock_path = master_dir / ".append.lock"
    lock_fd = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # ── 1. Deduplication ──────────────────────────────────────────────────
        index = _load_index(index_path)
        last_seq = index.get(str(session_id), -1)
        new_packets = [p for p in packet_rows if p.get("packet_seq", 0) > last_seq]
        skipped = len(packet_rows) - len(new_packets)

        # ── 2. Overwrite per-user baseline file ───────────────────────────────
        baseline_path.write_text(
            json.dumps(baseline_raw, indent=2, default=str),
            encoding="utf-8",
        )

        # ── 3. Check for embedded baseline in packets ─────────────────────────
        has_embedded_baseline = any(
            (p.get("packet") or {}).get("baseline_snapshot") is not None
            for p in new_packets
        )

        # ── 4. Append to master JSONL ─────────────────────────────────────────
        if new_packets:
            session_header: dict[str, Any] = {
                "user_id": user_id,
                "session_id": session_id,
                "document_id": session_meta_raw.get("document_id"),
                "session_mode": session_meta_raw.get("mode"),
                "started_at": session_meta_raw.get("started_at"),
                "ended_at": session_meta_raw.get("ended_at"),
                "protocol_tag": session_meta_raw.get("protocol_tag"),
                "baseline_valid": baseline_raw.get("baseline_valid"),
                "baseline_updated_at": baseline_raw.get("baseline_updated_at"),
                "baseline_ref": baseline_ref,
                "append_started_at": datetime.now(timezone.utc).isoformat(),
            }

            with master_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"# SESSION_START {json.dumps(session_header, default=str)}\n"
                )

                for i, p in enumerate(new_packets):
                    # Insert CHUNK_BREAK sentinel before every 20th packet
                    if i > 0 and i % _CHUNK_SIZE == 0:
                        chunk_meta = {
                            "session_id": session_id,
                            "after_packet_seq": new_packets[i - 1].get("packet_seq"),
                            "chunk_index": i // _CHUNK_SIZE,
                        }
                        f.write(f"# CHUNK_BREAK {json.dumps(chunk_meta)}\n")

                    line_obj = _build_packet_line(p, user_id, session_id, baseline_ref)
                    f.write(json.dumps(line_obj, default=str) + "\n")

                session_end: dict[str, Any] = {
                    "session_id": session_id,
                    "packet_count": len(new_packets),
                    "first_packet_seq": new_packets[0].get("packet_seq"),
                    "last_packet_seq": new_packets[-1].get("packet_seq"),
                }
                f.write(f"# SESSION_END {json.dumps(session_end)}\n")

            # Persist updated deduplication index
            max_seq = max(p.get("packet_seq", 0) for p in new_packets)
            index[str(session_id)] = max_seq
            _save_index(index_path, index)

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    return MasterAppendResult(
        master_jsonl_path=master_path,
        baseline_path=baseline_path,
        appended_packet_count=len(new_packets),
        skipped_packet_count=skipped,
        baseline_embedded_in_packet=has_embedded_baseline,
        baseline_ref=baseline_ref,
    )
