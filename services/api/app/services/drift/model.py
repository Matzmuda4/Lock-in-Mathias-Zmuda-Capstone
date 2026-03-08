"""
Hybrid drift model — Phase 7 (stabilized, bidirectional beta).

Mathematical foundation
-----------------------
Attention follows an exponential decay:

    A(t) = exp(-beta_effective * t)
    drift_score = 1 - A(t)

This guarantees:
- drift is NEVER permanently at 0 (it grows naturally with time-on-task)
- drift can DECREASE if beta drops below the current value (re-engagement)
- drift grows faster when beta_effective is higher (distraction/instability)

Beta modulation
---------------
beta_effective is computed every window cycle:

    beta_raw = beta0
             + confidence * W_DISRUPT * disruption_score
             - confidence * W_ENGAGE  * engagement_score

    beta_effective = clamp(beta_raw, BETA_MIN, BETA_MAX)
    beta_ema       = BETA_EMA_ALPHA * beta_effective + (1 - BETA_EMA_ALPHA) * prev_beta_ema

Then:
    drift_score = 1 - exp(-beta_ema * elapsed_minutes)
    drift_ema   = EMA_ALPHA * drift_score + (1 - EMA_ALPHA) * prev_ema

Disruption/engagement scoring
------------------------------
disruption_score ∈ [0,1]  — sigmoid of weighted z-score sum
    rises when: idle spikes, focus lost, stagnation, regress, jitter, skimming
engagement_score ∈ [0,1]  — multiplicative calm × pace-alignment
    rises when: calm (low idle + focus), pace near baseline, progress markers

Skimming detection
------------------
Skimming (pace_ratio > SKIM_THRESHOLD) fires two signals simultaneously:
  z_pace  — symmetric too-fast/too-slow deviation
  z_skim  — asymmetric extra boost for pace_ratio > 1.3 (reading too fast)
This ensures fast scrolling through many paragraphs is always detected,
even when other metrics (idle, focus) look normal.

Expected drift curves (typical calibrated user)
-----------------------------------------------
Focused reading:    beta ≈ 0.03 → drift 21% at 3 min, 26% at 10 min
Mild distraction:   beta ≈ 0.12 → drift 30% at 3 min, 70% at 10 min
Heavy distraction:  beta ≈ 0.48 → drift 62% at 2 min, 78% at 3 min
Skimming (1.7×):    beta ≈ 0.19 → drift 43% at 3 min, 61% at 5 min
"""

from __future__ import annotations

import math
from typing import Any

from app.services.drift.types import DriftResult, WindowFeatures, ZScores

# ── Constants ─────────────────────────────────────────────────────────────────

# Natural time-on-task decay rate (drift at baseline is ~3% at 1 min, ~21% at 8 min)
BETA0: float = 0.03
BETA_MIN: float = 0.02    # floor: even perfect engagement can't eliminate decay
BETA_MAX: float = 0.65    # ceiling: extreme distraction
BETA_EMA_ALPHA: float = 0.10   # slow EMA for stable beta; 90% there in ~1 min

# Beta modulation weights
W_DISRUPT: float = 0.50   # disruption_score × this → added to beta (max +0.50)
W_ENGAGE: float = 0.18    # engagement_score × this → subtracted from beta (max -0.18)

# Drift display smoothing
EMA_ALPHA: float = 0.25

# Z-score cap per signal term
Z_POS_CAP: float = 3.0

# Disruption sigmoid parameters: sigmoid((raw - CENTER) / SCALE)
DISRUPT_CENTER: float = 0.35
DISRUPT_SCALE: float = 0.25

# Skimming thresholds
SKIM_THRESHOLD: float = 1.3   # pace_ratio > this triggers asymmetric skimming signal
SKIM_SCALE: float = 0.5       # (pace_ratio - 1) / this → z_skim

# Disruption component weights (W_D_X × z_X → disruption_raw)
W_D_IDLE: float = 0.10
W_D_FOCUS: float = 0.12
W_D_STAGNATION: float = 0.08
W_D_JITTER: float = 0.06
W_D_REGRESS: float = 0.07
W_D_PACE: float = 0.15
W_D_SKIM: float = 0.18    # asymmetric: only fires when pace_ratio > SKIM_THRESHOLD
W_D_BURSTINESS: float = 0.05

_EPS: float = 1e-5


# ── Normalisation helpers ─────────────────────────────────────────────────────


def z_score(value: float, mean: float, std: float, clip: float = Z_POS_CAP) -> float:
    z = (value - mean) / (std + _EPS)
    return max(-clip, min(clip, z))


def z_pos(value: float, mean: float, std: float) -> float:
    """Z-score floored at 0, capped at Z_POS_CAP."""
    return max(0.0, z_score(value, mean, std))


def _sigmoid(raw: float, center: float, scale: float) -> float:
    x = max(-50.0, min(50.0, (raw - center) / (scale + _EPS)))
    return 1.0 / (1.0 + math.exp(-x))


# ── Z-score bundle ────────────────────────────────────────────────────────────


def compute_z_scores(features: WindowFeatures, baseline: dict[str, Any]) -> ZScores:
    """
    Normalize window features against the user's calibration baseline.

    Key behaviour:
    - z_mouse is 0 when mouse is stationary (< 10 px path) — not a distraction signal.
    - z_pace and z_skim are 0 when pace_available is False.
    - z_stagnation uses a personalized expected mean from para_dwell_median_s.
    """
    b = baseline

    z_idle = z_pos(
        features.idle_ratio_mean,
        b.get("idle_ratio_mean", 0.05),
        max(b.get("idle_ratio_std", 0.08), 0.04),
    )

    z_focus = z_pos(features.focus_loss_rate, 0.0, 0.08)

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

    # Stagnation: personalized expected fraction from para_dwell_median_s
    stagnation_mu = min(max(b.get("para_dwell_median_s", 10.0) / 30.0, 0.05), 0.80)
    z_stagnation = z_pos(features.stagnation_ratio, stagnation_mu, 0.15)

    # Mouse: skip when stationary
    z_mouse_val = (
        0.0
        if features.mouse_path_px_mean < 10.0
        else z_pos(0.70 - features.mouse_efficiency_mean, 0.0, 0.15)
    )

    # Pace: gated
    if not features.pace_available:
        z_pace_val = 0.0
    else:
        pace_scale = min(
            max(
                b.get("para_dwell_iqr_s", 5.0)
                / max(b.get("para_dwell_median_s", 10.0), _EPS),
                0.15,
            ),
            0.60,
        )
        z_pace_val = min(Z_POS_CAP, features.pace_dev / (pace_scale + _EPS))

    # Skimming asymmetric signal
    z_skim_val = (
        min(Z_POS_CAP, (features.pace_ratio - 1.0) / SKIM_SCALE)
        if features.pace_available and features.pace_ratio > SKIM_THRESHOLD
        else 0.0
    )

    z_burstiness = z_pos(features.scroll_burstiness, 1.0, 0.5)

    return ZScores(
        z_idle=z_idle,
        z_focus_loss=z_focus,
        z_jitter=z_jitter,
        z_regress=z_regress,
        z_pause=z_pause,
        z_stagnation=z_stagnation,
        z_mouse=z_mouse_val,
        z_pace=z_pace_val,
        z_skim=z_skim_val,
        z_burstiness=z_burstiness,
    )


# ── Disruption score ──────────────────────────────────────────────────────────


def compute_disruption_score(
    z: ZScores,
    baseline: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """
    Compute disruption_score ∈ [0,1] via sigmoid of weighted z-score sum.
    Returns (score, per-term breakdown dict).

    Higher-variance baseline signals get slightly down-weighted (the user is
    naturally variable, so we're less surprised by deviations).
    """

    def _adj(base_w: float, std_key: str, mean_key: str) -> float:
        std = baseline.get(std_key, 0.0)
        mu = max(abs(baseline.get(mean_key, 0.01)), 0.01)
        return base_w / (1.0 + min(std / mu, 2.0))

    w_idle = _adj(W_D_IDLE, "idle_ratio_std", "idle_ratio_mean")
    w_jitter = _adj(W_D_JITTER, "scroll_jitter_std", "scroll_jitter_mean")
    w_regress = _adj(W_D_REGRESS, "regress_rate_std", "regress_rate_mean")

    components: dict[str, float] = {
        "idle":        w_idle         * z.z_idle,
        "focus_loss":  W_D_FOCUS      * z.z_focus_loss,
        "stagnation":  W_D_STAGNATION * z.z_stagnation,
        "jitter":      w_jitter       * z.z_jitter,
        "regress":     w_regress      * z.z_regress,
        "pace":        W_D_PACE       * z.z_pace,
        "skim":        W_D_SKIM       * z.z_skim,
        "burstiness":  W_D_BURSTINESS * z.z_burstiness,
    }
    disruption_raw = sum(components.values())
    score = _sigmoid(disruption_raw, DISRUPT_CENTER, DISRUPT_SCALE)
    components["disruption_raw"] = disruption_raw
    components["disruption_score"] = score
    return score, components


# ── Engagement score ──────────────────────────────────────────────────────────


def compute_engagement_score(z: ZScores, features: WindowFeatures) -> float:
    """
    Compute engagement_score ∈ [0,1] using a multiplicative model.

    engagement = calm × (0.80 × pace_align + marker_boost)

    calm       = (1 - z_idle/CAP) × (1 - z_focus_loss/CAP)
    pace_align = 1 - max(z_pace, z_skim)/CAP   if pace_available else 0.5 (neutral)

    The multiplicative design ensures:
    - Skimming (bad pace_align) → engagement reduced even when calm is high
    - Distracted (calm → 0) → engagement → 0 regardless of pace
    - Focused steady reading → engagement ≈ 0.4 (moderate: no pace available yet)
    - Focused + on-pace → engagement ≈ 0.7
    """
    calm = (
        (1.0 - min(z.z_idle / Z_POS_CAP, 1.0))
        * (1.0 - min(z.z_focus_loss / Z_POS_CAP, 1.0))
    )

    if features.pace_available:
        worst_pace_z = max(z.z_pace, z.z_skim)
        pace_align = 1.0 - min(worst_pace_z / Z_POS_CAP, 1.0)
    else:
        pace_align = 0.5  # neutral when no evidence yet

    progress_boost = min(1.0, features.progress_markers_count / 2.0)
    score = calm * (0.80 * pace_align + 0.20 * progress_boost)
    return min(1.0, max(0.0, score))


# ── Beta computation ──────────────────────────────────────────────────────────


def compute_beta_raw(
    disruption_score: float,
    engagement_score: float,
    confidence: float,
) -> float:
    """
    Compute beta_raw from disruption, engagement, and confidence gating.

    beta_raw = beta0
             + confidence * W_DISRUPT * disruption_score
             - confidence * W_ENGAGE  * engagement_score
    """
    beta = BETA0 + confidence * W_DISRUPT * disruption_score - confidence * W_ENGAGE * engagement_score
    return min(max(beta, BETA_MIN), BETA_MAX)


def apply_beta_ema(beta_effective: float, prev_beta_ema: float) -> float:
    """Exponential moving average for beta to smooth transient spikes."""
    return BETA_EMA_ALPHA * beta_effective + (1.0 - BETA_EMA_ALPHA) * prev_beta_ema


def compute_attention(beta: float, t_minutes: float) -> float:
    return math.exp(-beta * max(t_minutes, 0.0))


def compute_drift(attention: float) -> float:
    return 1.0 - attention


# ── Confidence ────────────────────────────────────────────────────────────────


def compute_confidence(n_batches: int, full_window: int = 15) -> float:
    return min(1.0, n_batches / max(full_window, 1))


# ── EMA helpers ───────────────────────────────────────────────────────────────


def update_ema(value: float, prev_ema: float, alpha: float = EMA_ALPHA) -> float:
    return alpha * value + (1.0 - alpha) * prev_ema


# ── Elapsed time helper ───────────────────────────────────────────────────────


def elapsed_minutes_from_seconds(elapsed_seconds: float) -> float:
    return elapsed_seconds / 60.0


# ── Personalized rate helper (kept for test compat, not used in main model) ───


def personalized_rates(baseline: dict[str, Any]) -> tuple[float, float]:
    """
    Returns (up_rate, down_rate) adjusted for user variability.
    Kept for backward compatibility; the main model uses W_DISRUPT/W_ENGAGE.
    """
    from app.services.drift.model import W_DISRUPT, W_ENGAGE
    var_factor = min(max(
        baseline.get("idle_ratio_std", 0.05)
        + baseline.get("scroll_jitter_std", 0.05)
        + baseline.get("regress_rate_std", 0.03),
        0.01,
    ), 0.50)
    up_rate = W_DISRUPT * max(0.50, 1.0 - var_factor)
    down_rate = W_ENGAGE * max(0.70, 1.0 - 0.5 * var_factor)
    return up_rate, down_rate


# ── Main entry point ─────────────────────────────────────────────────────────


def compute_drift_result(
    features: WindowFeatures,
    baseline: dict[str, Any],
    elapsed_minutes: float,
    prev_ema: float,
    prev_beta_ema: float = BETA0,
) -> DriftResult:
    """
    Full drift pipeline: exponential decay model with disruption/engagement
    modulating the decay rate.

    Parameters
    ----------
    features:       Extracted from the current 30-second rolling window.
    baseline:       User's baseline_json (may be {} if uncalibrated).
    elapsed_minutes: Session elapsed time (from started_at); drives the exp decay.
    prev_ema:       Previous drift_ema (0.0 at session start).
    prev_beta_ema:  Previous smoothed beta (BETA0 at session start).

    Returns
    -------
    DriftResult with:
    - drift_level  = 1 - exp(-beta_ema * t)   [primary, always > 0 for t > 0]
    - drift_ema    = EMA(drift_level, prev_ema) [smoothed for UI]
    - beta_ema     = smoothed beta (pass back as prev_beta_ema next cycle)
    - disruption_score, engagement_score       [informational, debug]
    """
    z = compute_z_scores(features, baseline)
    disruption_score, components = compute_disruption_score(z, baseline)
    engagement_score = compute_engagement_score(z, features)
    confidence = compute_confidence(features.n_batches)

    # Beta: baseline decay + disruption boost - engagement dampening
    beta_effective = compute_beta_raw(disruption_score, engagement_score, confidence)

    # Smooth beta with slow EMA (prevents single-window spikes from over-driving drift)
    beta_ema = apply_beta_ema(beta_effective, prev_beta_ema)

    # Exponential attention decay
    attention = compute_attention(beta_ema, elapsed_minutes)
    drift_level = compute_drift(attention)

    # EMA-smooth drift for stable UI display
    drift_ema = update_ema(drift_level, prev_ema)

    # Build debug component dict
    components.update({
        "beta0": BETA0,
        "beta_effective": beta_effective,
        "beta_ema": beta_ema,
        "W_DISRUPT_x_disruption": W_DISRUPT * confidence * disruption_score,
        "W_ENGAGE_x_engagement": W_ENGAGE * confidence * engagement_score,
        "engagement_score": engagement_score,
        "confidence": confidence,
        "elapsed_minutes": elapsed_minutes,
    })

    return DriftResult(
        drift_level=drift_level,
        drift_ema=drift_ema,
        disruption_score=disruption_score,
        engagement_score=engagement_score,
        confidence=confidence,
        pace_ratio=features.pace_ratio if features.pace_available else None,
        pace_available=features.pace_available,
        # Legacy compat fields
        beta_effective=beta_effective,
        beta_ema=beta_ema,
        attention_score=attention,
        drift_score=drift_level,
        beta_components=components,
        features=features,
        z_scores=z,
    )
