"""
Pure drift-score mathematics — Phase 7.

All functions are side-effect-free and unit-testable.

Model summary
-------------
1. Z-score normalise window features against the user's calibration baseline.
   Each z_pos is capped at 2.0 to prevent any single signal from dominating.
2. Compute beta_raw as a weighted sum of positive deviations.
3. Apply beta EMA: beta_ema = 0.20 * beta_raw + 0.80 * prev_beta_ema
   (smooths transient spikes).
4. Apply confidence gating: beta_gated = lerp(beta0, beta_ema, confidence)
   (near beta0 during warm-up; full beta once window is saturated).
5. A(t) = exp(-beta_gated * t_minutes)
6. drift_score = 1 - A(t)
7. drift_ema = 0.25 * drift_score + 0.75 * prev_ema

Calibrated expected drift curves (beta0 = 0.03):
  Perfect focus:          beta ≈ 0.03–0.07  → drift_ema < 0.25 at 5 min
  Mild distraction:       beta ≈ 0.10–0.20  → drift_ema ≈ 0.40 at 5 min
  Heavy distraction:      beta ≈ 0.30–0.60  → drift_ema ≈ 0.70+ at 5 min
"""

from __future__ import annotations

import math
from typing import Any

from app.services.drift.types import DriftResult, WindowFeatures, ZScores

# ── Tunable model constants ───────────────────────────────────────────────────

# Natural decay rate when the reader is perfectly focused.
# At beta0=0.03 and t=10 min: A = exp(-0.30) = 0.74 → drift ≈ 26 %.
BETA0: float = 0.03

# Weights — each term contributes at most W * 2.0 to beta
# (z_pos is capped at 2.0).  Max possible addition per term shown in comment.
W_IDLE: float = 0.10        # max +0.20
W_FOCUS_LOSS: float = 0.15  # max +0.30
W_JITTER: float = 0.05      # max +0.10
W_REGRESS: float = 0.04     # max +0.08
W_PAUSE: float = 0.04       # max +0.08
W_STAGNATION: float = 0.05  # max +0.10
W_MOUSE: float = 0.03       # max +0.06 (only when mouse is active)
W_PACE: float = 0.10        # max +0.20 (only when pace_available)

# Max raw beta before clamping (sum of beta0 + all max contributions):
# 0.03 + 0.20 + 0.30 + 0.10 + 0.08 + 0.08 + 0.10 + 0.06 + 0.20 = 1.15
BETA_MIN: float = 0.02
BETA_MAX: float = 0.60      # conservative cap; prevents runaway at high t

# Beta EMA — smooths transient spikes (5-window time constant)
BETA_EMA_ALPHA: float = 0.20

# Drift score EMA
EMA_ALPHA: float = 0.25

# Pace z-score scale: 0.35 nats ≈ 1.4× speed deviation → z = 1.0
# A 2× deviation (log(2)≈0.693) → z ≈ 2.0 (capped)
PACE_SCALE: float = 0.35

# Cap for all z_pos terms — prevents any single signal from overwhelming beta
Z_POS_CAP: float = 2.0

_EPS: float = 1e-5


# ── Normalisation helpers ─────────────────────────────────────────────────────


def z_score(value: float, mean: float, std: float, clip: float = Z_POS_CAP) -> float:
    """Standard z-score clamped to [-clip, +clip]."""
    z = (value - mean) / (std + _EPS)
    return max(-clip, min(clip, z))


def z_pos(value: float, mean: float, std: float) -> float:
    """Z-score floored at 0, capped at Z_POS_CAP."""
    return max(0.0, z_score(value, mean, std))


# ── Z-score bundle ────────────────────────────────────────────────────────────


def compute_z_scores(
    features: WindowFeatures,
    baseline: dict[str, Any],
) -> ZScores:
    """
    Map window features → normalized deviations using the user baseline.

    Signal-specific notes:
    - z_mouse is 0 when mouse_path_px_mean < threshold (stationary = not applicable).
    - z_pace is 0 when pace_available is False (insufficient evidence).
    - z_stagnation fires only when a single paragraph dominates >80 % of the window.
    """
    b = baseline

    z_idle = z_pos(
        features.idle_ratio_mean,
        b.get("idle_ratio_mean", 0.05),
        max(b.get("idle_ratio_std", 0.08), 0.04),
    )

    z_focus = z_pos(
        features.focus_loss_rate,
        0.0,
        0.08,  # small std: any blur is above mean
    )

    z_jitter = z_pos(
        features.scroll_jitter_mean,
        b.get("scroll_jitter_mean", 0.10),
        max(b.get("scroll_jitter_std", 0.10), 0.05),
    )

    z_regress = z_pos(
        features.regress_rate_mean,
        b.get("regress_rate_mean", 0.05),
        max(b.get("regress_rate_std", 0.06), 0.03),
    )

    z_pause = z_pos(
        features.scroll_pause_mean,
        b.get("idle_seconds_mean", 1.0),
        max(b.get("idle_seconds_std", 1.0), 0.5),
    )

    z_stagnation = z_pos(
        features.paragraph_stagnation,
        0.0,
        0.15,
    )

    # Mouse efficiency: skip entirely when mouse is not moving.
    # An inactive mouse during reading is normal and carries no drift signal.
    if features.mouse_path_px_mean < 10.0:
        z_mouse_val = 0.0
    else:
        baseline_mouse_eff = 0.70
        z_mouse_val = z_pos(
            baseline_mouse_eff - features.mouse_efficiency_mean,
            0.0,
            0.15,
        )

    # Pace: only when pace_available (>= 2 paragraphs, >= 10 s effective).
    if not features.pace_available:
        z_pace_val = 0.0
    else:
        z_pace_val = min(Z_POS_CAP, features.pace_dev / (PACE_SCALE + _EPS))

    return ZScores(
        z_idle=z_idle,
        z_focus_loss=z_focus,
        z_jitter=z_jitter,
        z_regress=z_regress,
        z_pause=z_pause,
        z_stagnation=z_stagnation,
        z_mouse=z_mouse_val,
        z_pace=z_pace_val,
    )


# ── Beta effective ────────────────────────────────────────────────────────────


def compute_beta_raw(z: ZScores) -> tuple[float, dict[str, float]]:
    """
    Weighted sum of normalised deviations → raw decay rate.

    Returns (beta_raw, components) where components documents each term's
    contribution for the debug endpoint.
    """
    components = {
        "beta0": BETA0,
        "idle":       W_IDLE       * z.z_idle,
        "focus_loss": W_FOCUS_LOSS * z.z_focus_loss,
        "jitter":     W_JITTER     * z.z_jitter,
        "regress":    W_REGRESS    * z.z_regress,
        "pause":      W_PAUSE      * z.z_pause,
        "stagnation": W_STAGNATION * z.z_stagnation,
        "mouse":      W_MOUSE      * z.z_mouse,
        "pace":       W_PACE       * z.z_pace,
    }
    beta = sum(components.values())
    return max(BETA_MIN, min(BETA_MAX, beta)), components


def apply_beta_ema(beta_raw: float, prev_beta_ema: float) -> float:
    """
    Smooth beta over time: beta_ema = alpha * raw + (1 - alpha) * prev.
    This prevents large single-window spikes from immediately inflating drift.
    """
    return BETA_EMA_ALPHA * beta_raw + (1.0 - BETA_EMA_ALPHA) * prev_beta_ema


def apply_confidence_gating(beta_ema: float, confidence: float) -> float:
    """
    Linearly interpolate between beta0 (low confidence) and beta_ema (full confidence).

    During warm-up (first ~30 s of data) beta stays near beta0 regardless of signals.
    """
    return BETA0 * (1.0 - confidence) + beta_ema * confidence


# ── Attention and drift ───────────────────────────────────────────────────────


def compute_attention(beta: float, elapsed_minutes: float) -> float:
    """A(t) = exp(-beta * t).  Always in (0, 1]."""
    return math.exp(-beta * max(0.0, elapsed_minutes))


def compute_drift(attention: float) -> float:
    """drift_score = 1 - A(t), always in [0, 1)."""
    return 1.0 - attention


def update_ema(drift_score: float, prev_ema: float, alpha: float = EMA_ALPHA) -> float:
    """drift_ema = alpha * drift + (1 - alpha) * prev_ema."""
    return alpha * drift_score + (1.0 - alpha) * prev_ema


# ── Confidence ────────────────────────────────────────────────────────────────


def compute_confidence(n_batches: int, full_window: int = 15) -> float:
    """min(1.0, n_batches / full_window).  30 s window at 2 s/batch = 15 batches."""
    return min(1.0, n_batches / max(full_window, 1))


# ── Elapsed time helper ───────────────────────────────────────────────────────


def elapsed_minutes_from_seconds(elapsed_seconds: float) -> float:
    """Deterministic helper: seconds → minutes, asserted unit conversion."""
    return elapsed_seconds / 60.0


# ── Entry point ───────────────────────────────────────────────────────────────


def compute_drift_result(
    features: WindowFeatures,
    baseline: dict[str, Any],
    elapsed_minutes: float,
    prev_ema: float,
    prev_beta_ema: float = BETA0,
) -> DriftResult:
    """
    Full pipeline: features → z-scores → beta → EMA → gating → attention → drift.

    Parameters
    ----------
    features:
        Extracted from the current rolling window.
    baseline:
        User's baseline_json (may be {} if calibration not done yet).
    elapsed_minutes:
        Session elapsed time in MINUTES (wall clock from started_at).
    prev_ema:
        Previous drift_ema value (0.0 at session start).
    prev_beta_ema:
        Previous smoothed beta (defaults to BETA0 at session start).
    """
    z = compute_z_scores(features, baseline)
    beta_raw, components = compute_beta_raw(z)
    beta_ema = apply_beta_ema(beta_raw, prev_beta_ema)
    confidence = compute_confidence(features.n_batches)
    beta_gated = apply_confidence_gating(beta_ema, confidence)

    attention = compute_attention(beta_gated, elapsed_minutes)
    drift = compute_drift(attention)
    ema = update_ema(drift, prev_ema)

    # Add summary fields to components for debug readability
    components["beta_raw"] = beta_raw
    components["beta_ema"] = beta_ema
    components["confidence"] = confidence
    components["beta_gated"] = beta_gated

    return DriftResult(
        beta_effective=beta_gated,
        beta_ema=beta_ema,
        attention_score=attention,
        drift_score=drift,
        drift_ema=ema,
        confidence=confidence,
        beta_components=components,
        features=features,
        z_scores=z,
    )
