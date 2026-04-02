"""
RF Feature Extractor — single source of truth for the 19-element feature vector.

CRITICAL: this function MUST remain byte-for-byte identical to the
build_feature_vector() function used in the Colab training notebook
(rf_classifier_v2.pkl training Cell 8).  Any change here must be
mirrored in the notebook and the model must be retrained.

Feature layout (19 columns):
  [0..9]   z-scores   — personalised deviation signals
  [10..14] raw feats  — absolute behavioural rates
  [15..18] context    — window completeness, velocity, pace gate, raw focus-loss

Input: packet_json dict as produced by store._build_packet_json().
  - packet_json["z_scores"]  → ZScores dataclass fields
  - packet_json["features"]  → WindowFeatures dataclass fields
  - packet_json["drift"]     → drift block (pace_ratio, pace_available)
"""

from __future__ import annotations

from typing import Any

# Ordered list matches the column order the RF was trained on.
FEATURE_COLS: tuple[str, ...] = (
    # ── Z-scores (personalised, floored 0, capped 3) ─────────────────────────
    "z_idle",
    "z_skim",
    "z_regress",
    "z_pace",
    "z_pause",
    "z_jitter",
    "z_burstiness",
    "z_focus_loss",
    "z_stagnation",
    "z_mouse",
    # ── Raw behavioural features ──────────────────────────────────────────────
    "pace_ratio",
    "idle_ratio_mean",
    "stagnation_ratio",
    "regress_rate_mean",
    "panel_interaction_share",
    # ── Context features ──────────────────────────────────────────────────────
    "n_batches_norm",      # window completeness proxy: min(n,16)/16 → [0,1]
    "progress_velocity",   # rate of forward viewport progress (can be negative)
    "pace_available",      # z_skim validity gate: 1.0 or 0.0
    "focus_loss_rate",     # raw focus-loss proportion (pre-z-scoring)
)

# Cap used for n_batches_norm — MUST match the training notebook value (min(n,16)/16).
# Do NOT change this without retraining the model.
_NORM_CAP: int = 16

# Minimum batches required before the classifier fires.
# The live 30-second window yields at most 15 batches (30 s / 2 s per batch).
# Using 14 as the gate gives one batch of tolerance for timing jitter while
# still ensuring a near-complete window (≥ 28 s of signal).
FULL_WINDOW_BATCHES: int = 14


def build_feature_vector(packet_json: dict[str, Any]) -> list[float]:
    """
    Convert a live packet_json dict to a 19-element float feature vector.

    Mirrors the Colab training function exactly:
      - z-scores from packet_json["z_scores"]
      - raw features from packet_json["features"]
      - pace_ratio from drift block (with fallback to features block)
      - pace_available from features block (with fallback to drift block)

    All missing keys default to neutral values so the vector is always
    19 elements and safe to pass directly to the RF's predict_proba().
    """
    zs: dict[str, Any] = packet_json.get("z_scores") or {}
    ft: dict[str, Any] = packet_json.get("features") or {}
    dr: dict[str, Any] = packet_json.get("drift") or {}

    n_batches = float(ft.get("n_batches", FULL_WINDOW_BATCHES))

    # pace_ratio lives in drift block in live packets; fall back to features
    pace_ratio_raw = dr.get("pace_ratio")
    if pace_ratio_raw is None:
        pace_ratio_raw = ft.get("pace_ratio", 1.0)
    pace_ratio = float(pace_ratio_raw) if pace_ratio_raw is not None else 1.0

    # pace_available: bool → 0.0 / 1.0
    pace_available_raw = ft.get("pace_available")
    if pace_available_raw is None:
        pace_available_raw = dr.get("pace_available", False)
    pace_available = 1.0 if pace_available_raw else 0.0

    return [
        # Z-scores
        float(zs.get("z_idle",            0.0)),
        float(zs.get("z_skim",            0.0)),
        float(zs.get("z_regress",         0.0)),
        float(zs.get("z_pace",            0.0)),
        float(zs.get("z_pause",           0.0)),
        float(zs.get("z_jitter",          0.0)),
        float(zs.get("z_burstiness",      0.0)),
        float(zs.get("z_focus_loss",      0.0)),
        float(zs.get("z_stagnation",      0.0)),
        float(zs.get("z_mouse",           0.0)),
        # Raw behavioural
        pace_ratio,
        float(ft.get("idle_ratio_mean",        0.0)),
        float(ft.get("stagnation_ratio",       0.0)),
        float(ft.get("regress_rate_mean",      0.0)),
        float(ft.get("panel_interaction_share", 0.0)),
        # Context
        min(n_batches, float(_NORM_CAP)) / float(_NORM_CAP),
        float(ft.get("progress_velocity",      0.001)),
        pace_available,
        float(ft.get("focus_loss_rate",        0.0)),
    ]


def is_full_window(packet_json: dict[str, Any]) -> bool:
    """
    Return True when the packet covers a sufficiently complete 30-second window.

    The live 30-second window yields at most 15 batches (2 s cadence).
    FULL_WINDOW_BATCHES = 14 gives one batch of jitter tolerance while still
    ensuring ≥ 28 s of signal before the classifier fires.

    Classification on very short windows would constitute distribution shift
    relative to the training data, which was labelled on mature windows.
    """
    ft: dict[str, Any] = packet_json.get("features") or {}
    return int(ft.get("n_batches", 0)) >= FULL_WINDOW_BATCHES
