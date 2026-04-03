"""
Training-data flattener — classify branch.

Converts one session_state_packets row into a stable, ordered, model-friendly
flat dict suitable for CSV or JSONL export.

The schema is versioned (SCHEMA_VERSION) so Colab notebooks can detect breaking
changes.  Missing fields default to None / 0.0 — they never raise KeyError.

Column groups (in output order):
  1. identifiers       — session / user / document / packet metadata
  2. baseline scalars  — key calibration metrics (not the full baseline_json)
  3. window features   — 30-second rolling feature vector
  4. z-scores          — normalised deviations vs baseline
  5. drift summary     — model output (NOT to be used as training labels)
  6. ui aggregates     — UI context / interaction zone distributions
  7. packet_json       — full JSON string (lossless fallback)
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA_VERSION = "2.0.0"

# ── Ordered stable column list ────────────────────────────────────────────────
# Any additions must be appended at the END of each group to preserve ordering.

TRAINING_COLUMNS: list[str] = [
    # ── 1. Identifiers ────────────────────────────────────────────────────────
    "schema_version",
    "session_id",
    "user_id",
    "document_id",
    "packet_seq",
    "created_at",
    "window_start_at",
    "window_end_at",
    "session_mode",

    # ── 2. Baseline scalars ───────────────────────────────────────────────────
    "baseline_valid",
    "baseline_updated_at",
    "bl_wpm_effective",
    "bl_wpm_gross",
    "bl_idle_ratio_mean",
    "bl_idle_ratio_std",
    "bl_scroll_velocity_norm_mean",
    "bl_scroll_velocity_norm_std",
    "bl_regress_rate_mean",
    "bl_regress_rate_std",
    "bl_scroll_jitter_mean",
    "bl_scroll_jitter_std",
    "bl_para_dwell_median_s",
    "bl_para_dwell_iqr_s",
    "bl_calibration_duration_s",

    # ── 3. Window features (30-second rolling window) ─────────────────────────
    "feat_n_batches",
    "feat_idle_ratio_mean",
    "feat_focus_loss_rate",
    "feat_scroll_velocity_norm_mean",
    "feat_scroll_velocity_norm_std",
    "feat_scroll_burstiness",
    "feat_scroll_jitter_mean",
    "feat_regress_rate_mean",
    "feat_stagnation_ratio",
    "feat_mouse_efficiency_mean",
    "feat_mouse_path_px_mean",
    "feat_scroll_pause_mean",
    "feat_long_pause_share",
    "feat_pace_ratio",
    "feat_pace_dev",
    "feat_pace_available",
    "feat_window_wpm_effective",
    "feat_paragraphs_observed",
    "feat_progress_markers_count",
    "feat_progress_velocity",
    "feat_at_end_of_document",
    "feat_telemetry_fault_rate",
    "feat_scroll_capture_fault_rate",
    "feat_paragraph_missing_fault_rate",
    "feat_quality_confidence_mult",

    # ── 4. Z-scores ───────────────────────────────────────────────────────────
    "z_idle",
    "z_focus_loss",
    "z_jitter",
    "z_regress",
    "z_pause",
    "z_stagnation",
    "z_mouse",
    "z_pace",
    "z_skim",
    "z_burstiness",

    # ── 5. Drift summary (model output — NOT training labels) ─────────────────
    "drift_ema",
    "drift_level",
    "beta_ema",
    "beta_effective",
    "disruption_score",
    "engagement_score",
    "confidence",
    "pace_ratio",
    "pace_available",

    # ── 6. UI aggregates (30-second window distributions) ─────────────────────
    "ui_read_main_share_30s",
    "ui_panel_open_share_30s",
    "ui_panel_interacting_share_30s",
    "ui_user_paused_share_30s",
    "panel_share_30s",
    "reader_share_30s",
    "iz_reader_share_30s",
    "iz_panel_share_30s",
    "iz_other_share_30s",

    # ── 7. Lossless fallback ──────────────────────────────────────────────────
    "packet_json",
]


def flatten_packet_to_row(
    packet_row_meta: dict[str, Any],
    packet_json: dict[str, Any],
    include_debug: bool = False,
) -> dict[str, Any]:
    """
    Flatten one session_state_packets row into a stable ordered dict.

    Parameters
    ----------
    packet_row_meta : dict with ORM-level fields:
        session_id, user_id (from sessions join), document_id, packet_seq,
        created_at, window_start_at, window_end_at, session_mode.
    packet_json : the JSONB payload stored in session_state_packets.packet_json.
    include_debug : if True, append beta_components fields.

    Returns
    -------
    Ordered dict with TRAINING_COLUMNS keys (plus optional debug columns).
    Missing values are None.
    """
    bsn = packet_json.get("baseline_snapshot") or {}
    bl = bsn.get("baseline_json") or {}
    feat = packet_json.get("features") or {}
    zsc = packet_json.get("z_scores") or {}
    drift = packet_json.get("drift") or {}
    ui = packet_json.get("ui_aggregates") or {}

    # Session identity — prefer packet_json (new packets embed these), fall back
    # to the explicit meta passed by the caller (covers older packets).
    meta = packet_row_meta

    row: dict[str, Any] = {
        # Identifiers
        "schema_version": SCHEMA_VERSION,
        "session_id": packet_json.get("session_id") or meta.get("session_id"),
        "user_id": packet_json.get("user_id") or meta.get("user_id"),
        "document_id": packet_json.get("document_id") or meta.get("document_id"),
        "packet_seq": meta.get("packet_seq"),
        "created_at": _iso(meta.get("created_at")),
        "window_start_at": _iso(meta.get("window_start_at")),
        "window_end_at": _iso(meta.get("window_end_at")),
        "session_mode": packet_json.get("session_mode") or meta.get("session_mode"),

        # Baseline scalars
        "baseline_valid": bsn.get("baseline_valid"),
        "baseline_updated_at": bsn.get("baseline_updated_at"),
        "bl_wpm_effective": bl.get("wpm_effective"),
        "bl_wpm_gross": bl.get("wpm_gross"),
        "bl_idle_ratio_mean": bl.get("idle_ratio_mean"),
        "bl_idle_ratio_std": bl.get("idle_ratio_std"),
        "bl_scroll_velocity_norm_mean": bl.get("scroll_velocity_norm_mean"),
        "bl_scroll_velocity_norm_std": bl.get("scroll_velocity_norm_std"),
        "bl_regress_rate_mean": bl.get("regress_rate_mean"),
        "bl_regress_rate_std": bl.get("regress_rate_std"),
        "bl_scroll_jitter_mean": bl.get("scroll_jitter_mean"),
        "bl_scroll_jitter_std": bl.get("scroll_jitter_std"),
        "bl_para_dwell_median_s": bl.get("para_dwell_median_s"),
        "bl_para_dwell_iqr_s": bl.get("para_dwell_iqr_s"),
        "bl_calibration_duration_s": bl.get("calibration_duration_seconds"),

        # Window features
        "feat_n_batches": feat.get("n_batches"),
        "feat_idle_ratio_mean": feat.get("idle_ratio_mean"),
        "feat_focus_loss_rate": feat.get("focus_loss_rate"),
        "feat_scroll_velocity_norm_mean": feat.get("scroll_velocity_norm_mean"),
        "feat_scroll_velocity_norm_std": feat.get("scroll_velocity_norm_std"),
        "feat_scroll_burstiness": feat.get("scroll_burstiness"),
        "feat_scroll_jitter_mean": feat.get("scroll_jitter_mean"),
        "feat_regress_rate_mean": feat.get("regress_rate_mean"),
        "feat_stagnation_ratio": feat.get("stagnation_ratio"),
        "feat_mouse_efficiency_mean": feat.get("mouse_efficiency_mean"),
        "feat_mouse_path_px_mean": feat.get("mouse_path_px_mean"),
        "feat_scroll_pause_mean": feat.get("scroll_pause_mean"),
        "feat_long_pause_share": feat.get("long_pause_share"),
        "feat_pace_ratio": feat.get("pace_ratio"),
        "feat_pace_dev": feat.get("pace_dev"),
        "feat_pace_available": feat.get("pace_available"),
        "feat_window_wpm_effective": feat.get("window_wpm_effective"),
        "feat_paragraphs_observed": feat.get("paragraphs_observed"),
        "feat_progress_markers_count": feat.get("progress_markers_count"),
        "feat_progress_velocity": feat.get("progress_velocity"),
        "feat_at_end_of_document": feat.get("at_end_of_document"),
        "feat_telemetry_fault_rate": feat.get("telemetry_fault_rate"),
        "feat_scroll_capture_fault_rate": feat.get("scroll_capture_fault_rate"),
        "feat_paragraph_missing_fault_rate": feat.get("paragraph_missing_fault_rate"),
        "feat_quality_confidence_mult": feat.get("quality_confidence_mult"),

        # Z-scores
        "z_idle": zsc.get("z_idle"),
        "z_focus_loss": zsc.get("z_focus_loss"),
        "z_jitter": zsc.get("z_jitter"),
        "z_regress": zsc.get("z_regress"),
        "z_pause": zsc.get("z_pause"),
        "z_stagnation": zsc.get("z_stagnation"),
        "z_mouse": zsc.get("z_mouse"),
        "z_pace": zsc.get("z_pace"),
        "z_skim": zsc.get("z_skim"),
        "z_burstiness": zsc.get("z_burstiness"),

        # Drift summary
        "drift_ema": drift.get("drift_ema"),
        "drift_level": drift.get("drift_level"),
        "beta_ema": drift.get("beta_ema"),
        "beta_effective": drift.get("beta_effective"),
        "disruption_score": drift.get("disruption_score"),
        "engagement_score": drift.get("engagement_score"),
        "confidence": drift.get("confidence"),
        "pace_ratio": drift.get("pace_ratio"),
        "pace_available": drift.get("pace_available"),

        # UI aggregates
        "ui_read_main_share_30s": ui.get("ui_read_main_share_30s"),
        "ui_panel_open_share_30s": ui.get("ui_panel_open_share_30s"),
        "ui_panel_interacting_share_30s": ui.get("ui_panel_interacting_share_30s"),
        "ui_user_paused_share_30s": ui.get("ui_user_paused_share_30s"),
        "panel_share_30s": ui.get("panel_share_30s"),
        "reader_share_30s": ui.get("reader_share_30s"),
        "iz_reader_share_30s": ui.get("iz_reader_share_30s"),
        "iz_panel_share_30s": ui.get("iz_panel_share_30s"),
        "iz_other_share_30s": ui.get("iz_other_share_30s"),

        # Lossless full JSON
        "packet_json": json.dumps(packet_json, default=str),
    }

    if include_debug:
        dbg = packet_json.get("debug") or {}
        for k, v in dbg.items():
            row[f"debug_{k}"] = v

    # Return in declared column order (plus any debug extras at the end)
    base_keys = list(TRAINING_COLUMNS)
    ordered: dict[str, Any] = {k: row.get(k) for k in base_keys}
    if include_debug:
        for k, v in row.items():
            if k.startswith("debug_"):
                ordered[k] = v

    return ordered


def build_jsonl_line(
    packet_row_meta: dict[str, Any],
    packet_json: dict[str, Any],
    include_debug: bool = False,
) -> dict[str, Any]:
    """
    Build a structured JSONL object (one dict per line).

    Schema:
    { "meta": {...}, "baseline": {...}, "features": {...},
      "z_scores": {...}, "drift": {...}, "ui_aggregates": {...},
      "packet_json": {...} }
    """
    bsn = packet_json.get("baseline_snapshot") or {}
    bl = bsn.get("baseline_json") or {}

    line: dict[str, Any] = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "session_id": packet_json.get("session_id") or packet_row_meta.get("session_id"),
            "user_id": packet_json.get("user_id") or packet_row_meta.get("user_id"),
            "document_id": packet_json.get("document_id") or packet_row_meta.get("document_id"),
            "session_mode": packet_json.get("session_mode") or packet_row_meta.get("session_mode"),
            "packet_seq": packet_row_meta.get("packet_seq"),
            "created_at": _iso(packet_row_meta.get("created_at")),
            "window_start_at": _iso(packet_row_meta.get("window_start_at")),
            "window_end_at": _iso(packet_row_meta.get("window_end_at")),
        },
        "baseline": {
            "baseline_valid": bsn.get("baseline_valid"),
            "baseline_updated_at": bsn.get("baseline_updated_at"),
            **bl,
        },
        "features": packet_json.get("features") or {},
        "z_scores": packet_json.get("z_scores") or {},
        "drift": packet_json.get("drift") or {},
        "ui_aggregates": packet_json.get("ui_aggregates") or {},
        "packet_json": packet_json,
    }
    if include_debug:
        line["debug"] = packet_json.get("debug") or {}
    return line


# ── Internal helpers ──────────────────────────────────────────────────────────


def _iso(value: Any) -> str | None:
    """Return ISO-8601 string for a datetime or passthrough a string."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
