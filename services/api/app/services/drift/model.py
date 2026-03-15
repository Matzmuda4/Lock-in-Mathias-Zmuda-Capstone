"""
Hybrid drift model — Phase 7 v2 (idle-sensitive, research-grounded weights).

═══════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
═══════════════════════════════════════════════════════════════════════════

Attention follows a personalised exponential decay (Yerkes–Dodson / arousal
literature; exponential forgetting curves — Ebbinghaus 1885; applied to
sustained reading by Smallwood & Schooler 2006):

    A(t)  = exp(−beta_ema × t_minutes)    ∈ (0, 1]
    drift = 1 − A(t)                       ∈ [0, 1)

Properties:
  • drift is NEVER stuck at 0 (grows naturally with time-on-task)
  • drift DECREASES when beta_ema drops (re-engagement, recovery)
  • drift grows faster when beta_ema is large (distraction / idleness)
  • at t = 0 drift is exactly 0; at t → ∞ drift → 1

═══════════════════════════════════════════════════════════════════════════
BETA MODULATION (personalised decay rate)
═══════════════════════════════════════════════════════════════════════════

Every 2-second telemetry cycle recomputes beta_effective, then smooth-tracks
it with a slow EMA so single-window noise cannot spike drift:

    beta_raw = BETA0
             + confidence × W_DISRUPT × disruption_score
             − confidence × W_ENGAGE  × engagement_score

    beta_effective = clamp(beta_raw, BETA_MIN, BETA_MAX)
    beta_ema       = BETA_EMA_ALPHA × beta_effective
                   + (1 − BETA_EMA_ALPHA) × prev_beta_ema

Expected beta values for a correctly-calibrated user:
  Focused reading at baseline    →  beta ≈ BETA_MIN = 0.02
  Slightly above-baseline idle   →  beta ≈ 0.05 – 0.12
  Complete idle, no interaction  →  beta ≈ 0.63 – 0.65 (BETA_MAX)
  Tab away / window blur         →  beta ≈ 0.65 (BETA_MAX)
  Skimming (2× baseline WPM)     →  beta ≈ 0.46

Expected drift curves:
  beta = 0.02 (focused at baseline): 2% @ 1 min,  6% @ 3 min, 18% @ 10 min
  beta = 0.20 (mild distraction):   18% @ 1 min, 45% @ 3 min, 86% @ 10 min
  beta = 0.45 (skimming/regress):   36% @ 1 min, 74% @ 3 min, ~99% @ 10 min
  beta = 0.65 (idle/blur at max):   48% @ 1 min, 86% @ 3 min, ~100% @ 10 min

═══════════════════════════════════════════════════════════════════════════
DISRUPTION SCORE (research-grounded signal weights)
═══════════════════════════════════════════════════════════════════════════

disruption_score = sigmoid((disruption_raw − CENTER) / SCALE)

disruption_raw = W_D_IDLE       × z_idle        # PRIMARY — see below
               + W_D_FOCUS      × z_focus_loss   # explicit disengagement
               + W_D_STAGNATION × z_stagnation   # stuck on same paragraph
               + W_D_PACE       × z_pace         # too fast or too slow
               + W_D_SKIM       × z_skim         # asymmetric fast-skim
               + W_D_REGRESS    × z_regress      # comprehension-difficulty proxy
               + W_D_JITTER     × z_jitter       # restlessness/erratic scroll
               + W_D_BURSTINESS × z_burstiness   # burst scroll instability

Research basis for signal ordering:
  1. Idle / lack of interaction  (W_D_IDLE = 0.22)
     Smallwood & Schooler (2006, Psych. Bull.) — sustained mind-wandering is
     the strongest predictor of offline, stimulus-independent thought.
     Unsworth & McMillan (2013) link scroll inactivity to mind-wandering.
  2. Focus loss / tab-away  (W_D_FOCUS = 0.18)
     Direct behavioural evidence: the user has explicitly left the reading
     context. Slightly lower than idle because brief accidental switches occur.
  3. Stagnation on paragraph  (W_D_STAGNATION = 0.12)
     Rayner (1998, Psych. Bull.) — abnormally long dwell on a single region
     indicates comprehension difficulty or zoning-out.
  4. Pace deviation  (W_D_PACE = 0.15) + Skim  (W_D_SKIM = 0.18)
     Just & Carpenter (1980) — reading rate is tightly coupled to cognitive
     processing depth. Skimming (impulsive fast forward) AND crawling (stuck)
     both signal attentional anomaly. Skim receives an asymmetric extra weight.
  5. Regress rate  (W_D_REGRESS = 0.10)
     Rayner & Pollatsek (1989) — high regressive saccade rate correlates with
     reading difficulty / comprehension failure. Weighted below idle because
     some regress is deliberate review.
  6. Jitter / direction changes  (W_D_JITTER = 0.08)
     Novel signal; restless oscillatory scroll is associated with attention
     restlessness (no published norm; weight is conservative).
  7. Scroll burstiness  (W_D_BURSTINESS = 0.05) — secondary instability signal.

IMPORTANT: idle weight is NEVER down-weighted by baseline variability.
Idle is the single most reliable attention signal regardless of user pattern.
Jitter and regress weights use a softened variability adjustment (0.5×
instead of 1.0×) so highly-variable users are only mildly de-sensitised.

═══════════════════════════════════════════════════════════════════════════
ENGAGEMENT SCORE (multiplicative — skimming cannot hide behind calmness)
═══════════════════════════════════════════════════════════════════════════

engagement = calm × (0.80 × pace_align + 0.20 × progress_boost)

calm       = (1 − z_idle/3) × (1 − z_focus_loss/3)   ∈ [0, 1]
pace_align = 1 − max(z_pace, z_skim) / 3              ∈ [0, 1]  if pace_available
           = 0.5 (neutral)                             if pace not yet measurable

The multiplicative design ensures:
  • Skimming (low pace_align) reduces engagement even when the user is calm
  • Blur / idle (low calm)   reduces engagement even when pace looks normal
  • Both bad simultaneously → engagement → 0 → beta approaches BETA_MAX

═══════════════════════════════════════════════════════════════════════════
CALIBRATION BASELINE USAGE
═══════════════════════════════════════════════════════════════════════════

All z-scores are computed against the user's own calibration baseline:

    z_idle      = z_pos(idle_ratio_mean,  baseline["idle_ratio_mean"],
                                          baseline["idle_ratio_std"])
    z_stagnation = z_pos(stagnation_ratio, stagnation_mu, 0.15)
                   where stagnation_mu = para_dwell_median_s / 30.0
    z_pace      = pace_dev / pace_scale
                  where pace_scale from para_dwell_iqr/median ratio
    ... etc.

This means the SAME idle_ratio = 0.8 can produce very different z_idle
depending on the user:
  • User A: baseline idle_ratio_mean=0.35, std=0.20 → z_idle = 2.25
  • User B: baseline idle_ratio_mean=0.70, std=0.15 → z_idle = 0.67
User B naturally pauses more; they are less disrupted by the same raw idle.

Fallback defaults when calibration baseline is absent:
  idle_ratio_mean = 0.35  (population average for reading sessions)
  idle_ratio_std  = 0.20  (realistic window-to-window variability)
Users without calibration get a reasonable population baseline so drift
still responds correctly. Calibration is mandatory in the app, so fallbacks
are a safety net, not the primary path.
"""

from __future__ import annotations

import math
from typing import Any

from app.services.drift.types import DriftResult, WindowFeatures, ZScores

# ── Constants ─────────────────────────────────────────────────────────────────

# Natural time-on-task decay rate
# BETA0 = 0.03 → drift ≈ 3% at 1 min, 9% at 3 min, 26% at 10 min
BETA0: float = 0.03
BETA_MIN: float = 0.02    # floor: even perfect engagement can't eliminate decay
BETA_MAX: float = 0.65    # ceiling: extreme distraction / full idle
BETA_EMA_ALPHA: float = 0.20  # 50% convergence in ~3 steps (6 s) — responsive

# Beta modulation weights (research-grounded; see module docstring)
W_DISRUPT: float = 0.70   # disruption_score contribution to beta (max +0.70)
W_ENGAGE: float = 0.425   # engagement_score reduction of beta  (max -0.425)

# Drift display smoothing (EMA alpha for UI value)
EMA_ALPHA: float = 0.25

# Z-score cap per signal term
Z_POS_CAP: float = 3.0

# Disruption sigmoid parameters: sigmoid((raw - CENTER) / SCALE)
# CENTER = 0.35 means disruption_raw must exceed 0.35 to score > 0.5
DISRUPT_CENTER: float = 0.35
DISRUPT_SCALE: float = 0.25

# Skimming thresholds
SKIM_THRESHOLD: float = 1.3   # pace_ratio > this triggers asymmetric skim signal
SKIM_SCALE: float = 0.5       # (pace_ratio - 1) / this → z_skim

# Disruption component weights (W_D_X × z_X → disruption_raw)
# Ordered by research-backed importance (see module docstring)
W_D_IDLE: float = 0.22        # PRIMARY — never reduced by baseline variability
W_D_FOCUS: float = 0.18       # explicit disengagement
W_D_STAGNATION: float = 0.12  # stuck on paragraph
W_D_PACE: float = 0.15        # pace deviation (too fast or too slow)
W_D_SKIM: float = 0.18        # asymmetric fast-skim signal
W_D_REGRESS: float = 0.10     # high back-scroll rate
W_D_JITTER: float = 0.08      # erratic direction changes
W_D_BURSTINESS: float = 0.05  # burst scroll instability

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

    # Realistic fallbacks for uncalibrated users:
    # readers are idle ~35% of 2-second windows between scrolls
    z_idle = z_pos(
        features.idle_ratio_mean,
        b.get("idle_ratio_mean", 0.35),
        max(b.get("idle_ratio_std", 0.20), 0.08),
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

    Idle weight is FIXED at W_D_IDLE — never reduced by baseline variability.
    Idle is the primary attention signal; all users should be equally sensitive
    to deviations from their own measured idle baseline.

    Jitter and regress use a SOFTENED variability adjustment (0.5× penalty
    vs old 1.0×) so highly-variable users are only mildly de-sensitised.
    """

    def _adj(base_w: float, std_key: str, mean_key: str) -> float:
        """Soft variability adjustment: reduce weight by at most 50%."""
        std = baseline.get(std_key, 0.0)
        mu = max(abs(baseline.get(mean_key, 0.01)), 0.01)
        return base_w / (1.0 + 0.5 * min(std / mu, 2.0))

    # Idle: fixed weight — never adjusted by baseline variability
    w_idle = W_D_IDLE
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
        baseline.get("idle_ratio_std", 0.20)
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

    # ── C1: Skimming fallback via progress_velocity ────────────────────────
    # When pace estimation is unavailable (no paragraph IDs), use the rate of
    # viewport_progress_ratio change as a secondary skimming signal.
    # Only triggers when progress_velocity is strongly positive (fast scroll).
    skim_fallback_z = 0.0
    if not features.pace_available and features.progress_velocity > 0.02:
        # Normalize: > 0.05/s is fast scrolling (half-page per second)
        skim_fallback_z = min(Z_POS_CAP, features.progress_velocity / 0.03)
        z = ZScores(
            z_idle=z.z_idle,
            z_focus_loss=z.z_focus_loss,
            z_jitter=z.z_jitter,
            z_regress=z.z_regress,
            z_pause=z.z_pause,
            z_stagnation=z.z_stagnation,
            z_mouse=z.z_mouse,
            z_pace=z.z_pace,
            z_skim=max(z.z_skim, skim_fallback_z),
            z_burstiness=z.z_burstiness,
        )

    disruption_score, components = compute_disruption_score(z, baseline)
    engagement_score = compute_engagement_score(z, features)

    # ── B2: Apply data-quality confidence penalty ──────────────────────────
    # Multiply base confidence by quality multiplier so broken telemetry
    # reduces the model's aggressiveness.
    base_confidence = compute_confidence(features.n_batches)
    confidence = base_confidence * features.quality_confidence_mult

    # ── C2: Faster re-engagement: increase EMA alpha when engagement is high
    # When the user shows sustained engagement (score > 0.6), allow beta to
    # drop faster by using a higher EMA alpha temporarily.
    effective_ema_alpha = BETA_EMA_ALPHA
    if engagement_score > 0.60 and prev_beta_ema > BETA0 * 2:
        effective_ema_alpha = min(0.40, BETA_EMA_ALPHA * 2)

    # Beta: baseline decay + disruption boost - engagement dampening
    beta_effective = compute_beta_raw(disruption_score, engagement_score, confidence)

    # Smooth beta with EMA (faster when re-engaging)
    beta_ema = effective_ema_alpha * beta_effective + (1.0 - effective_ema_alpha) * prev_beta_ema
    beta_ema = min(max(beta_ema, BETA_MIN), BETA_MAX)

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
        "base_confidence": base_confidence,
        "quality_confidence_mult": features.quality_confidence_mult,
        "skim_fallback_z": skim_fallback_z,
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
