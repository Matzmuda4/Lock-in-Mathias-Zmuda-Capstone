"""
Hybrid drift model — Phase 7 v3 (ADHD-calibrated, accurate for sustained reading).

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
CALIBRATION (v4 — ADHD-appropriate timescales, accurate for real reading)
═══════════════════════════════════════════════════════════════════════════

Research on ADHD sustained attention (Barkley 1997; Tsal et al 2005) shows
that attentional drift typically becomes clinically significant after
10–12 minutes of uninterrupted reading for typical ADHD presentation.

The beta scale is set so that:
  Fully engaged reader at baseline → beta ≈ BETA_MIN = 0.003
    → drift:  ~0.3% @ 1 min,  ~1.5% @ 5 min,  ~3% @ 10 min  (essentially flat)

  Normal reading with minor variation → beta ≈ 0.015–0.025
    → drift:  ~2–3% @ 1 min,  ~8–14% @ 5 min,  ~16–25% @ 10 min

  Moderate distraction (stuck, some blur, pace deviation) → beta ≈ 0.05–0.12
    → drift:  ~5–11% @ 1 min,  ~22–45% @ 5 min,  ~39–70% @ 10 min

  Heavy distraction (tab-away, full idle) → beta ≈ 0.20–0.30 (BETA_MAX)
    → drift:  ~20–26% @ 1 min,  ~63–78% @ 5 min,  ~86–95% @ 10 min

KEY v4 FIXES:
  1. DISRUPT_CENTER raised 0.35 → 0.45: mild signals (small stagnation,
     moderate idle) no longer push disruption_score above 0.5.
  2. W_DISRUPT lowered 0.40 → 0.30 and BETA_MAX lowered 0.40 → 0.30:
     heavy distraction reaches ~0.29 beta (not 0.39), so drift at 1 min
     stays ~25% rather than ~33% for extreme tab-away.
  3. W_D_IDLE lowered 0.22 → 0.15 + wider idle_std floor 0.08 → 0.25:
     natural reading pauses (reader sits still for 10-20s on a long
     paragraph) no longer produce high z_idle.
  4. Baseline sanity checks for broken calibration: if the user's
     calibration was done with broken scroll tracking (jitter_mean = 0.0,
     regress_mean = 0.0), those zero baselines are replaced with realistic
     population-average fallbacks.  This prevents every normal jitter event
     from appearing as a huge deviation from an impossible zero baseline.
  5. Stagnation halved when pace unavailable + no blur: when the reader is
     on a long paragraph (can't distinguish reading from stuck without pace
     data and no explicit focus loss), z_stagnation is halved to avoid
     false positives.

═══════════════════════════════════════════════════════════════════════════
BETA MODULATION (personalised decay rate)
═══════════════════════════════════════════════════════════════════════════

Every 2-second telemetry cycle recomputes beta_effective, then smooth-tracks
it with an EMA so single-window noise cannot spike drift:

    beta_raw = BETA0
             + confidence × W_DISRUPT × disruption_score
             − confidence × W_ENGAGE  × engagement_score

    beta_effective = clamp(beta_raw, BETA_MIN, BETA_MAX)
    beta_ema       = BETA_EMA_ALPHA × beta_effective
                   + (1 − BETA_EMA_ALPHA) × prev_beta_ema

═══════════════════════════════════════════════════════════════════════════
DISRUPTION SCORE (research-grounded signal weights)
═══════════════════════════════════════════════════════════════════════════

disruption_score = sigmoid((disruption_raw − CENTER) / SCALE)

disruption_raw = W_D_IDLE       × z_idle        # PRIMARY — see below
               + W_D_FOCUS      × z_focus_loss   # explicit disengagement
               + W_D_STAGNATION × z_stagnation   # truly stuck (NOT normal dwell)
               + W_D_PACE       × z_pace         # too fast or too slow
               + W_D_SKIM       × z_skim         # asymmetric fast-skim
               + W_D_REGRESS    × z_regress      # comprehension-difficulty proxy
               + W_D_JITTER     × z_jitter       # restlessness/erratic scroll
               + W_D_BURSTINESS × z_burstiness   # erratic velocity pattern

KEY v3 FIX — Stagnation normalisation:
  Staying on one paragraph for the majority of a 30 s window is EXPECTED
  reading behaviour, not a distraction signal.  The stagnation mean is
  floored at 0.45 so that normal dwell (50% of window on one paragraph)
  produces z_stagnation ≈ 0.  Only truly frozen behaviour (80%+ on same
  paragraph AND above expected median) triggers significant z_stagnation.

KEY v3 FIX — Burstiness normalisation:
  Natural reading is bursty: the user scrolls a little, pauses to read,
  scrolls again.  Coefficient of variation ≈ 2–3 is normal.  The baseline
  is now 2.5 (was 1.0) so normal bursting no longer penalises the reader.

═══════════════════════════════════════════════════════════════════════════
ENGAGEMENT SCORE (multiplicative — skimming cannot hide behind calmness)
═══════════════════════════════════════════════════════════════════════════

engagement = calm × (0.80 × pace_align + 0.20 × progress_boost)

calm       = (1 − z_idle/3) × (1 − z_focus_loss/3)   ∈ [0, 1]
pace_align = 1 − max(z_pace, z_skim) / 3              ∈ [0, 1]  if pace_available
           = 0.65 (benefit of the doubt)               if pace not yet measurable

KEY v3 FIX — pace_align default raised from 0.5 → 0.65:
  During the first minute (before 2 paragraphs observed), the absence of
  pace evidence should NOT count against the reader.  0.65 means focused
  reading without pace data still yields engagement ≈ 0.52+, enough to
  hold beta near BETA_MIN.

═══════════════════════════════════════════════════════════════════════════
CALIBRATION BASELINE USAGE
═══════════════════════════════════════════════════════════════════════════

All z-scores are computed against the user's own calibration baseline.
The SAME raw value can produce different z-scores per user:
  User A: baseline idle_ratio_mean=0.35, std=0.20 → z_idle(0.8) = 2.25
  User B: baseline idle_ratio_mean=0.70, std=0.15 → z_idle(0.8) = 0.67

Fallback defaults (when calibration absent or key missing):
  idle_ratio_mean = 0.35, idle_ratio_std = 0.20 (floor 0.25)
  scroll_jitter_mean = 0.10, scroll_jitter_std = 0.10
  regress_rate_mean = 0.05, regress_rate_std = 0.06
  para_dwell_median_s = 10.0, para_dwell_iqr_s = 5.0

Baseline sanity checks (v4 — broken calibration defence):
  If calibration was done with non-functional scroll tracking, the stored
  scroll_jitter_mean and regress_rate_mean will be near 0.  Those values
  produce astronomically high z-scores for any real session activity.
  If jitter_mean < 0.04 or regress_mean < 0.02, override with population
  averages (0.10 / 0.05) so the model behaves sensibly until the user
  re-calibrates with the fixed telemetry.
"""

from __future__ import annotations

import math
from typing import Any

from app.services.drift.types import DriftResult, WindowFeatures, ZScores

# ── Constants ─────────────────────────────────────────────────────────────────

# Natural time-on-task decay rate (very small — represents inevitable fatigue)
# BETA0 = 0.005 → drift ≈ 0.5% at 1 min, 2.5% at 5 min, 4.9% at 10 min
BETA0: float = 0.005
# Floor: fully focused reader experiences almost no drift growth per cycle
BETA_MIN: float = 0.003
# Ceiling: heavy distraction / complete tab-away.
# v4: lowered 0.40 → 0.30 to match W_DISRUPT=0.30 range.
# At 0.30: drift = 26% @ 1 min, 78% @ 5 min, 95% @ 10 min
BETA_MAX: float = 0.30
# EMA alpha for beta smoothing — 0.30 gives ~7 cycles (14 s) to reach 95%
# of a step change.  Fast enough to respond but resistant to single-window spikes.
BETA_EMA_ALPHA: float = 0.30

# Beta modulation weights
# v4: W_DISRUPT lowered 0.40 → 0.30 so BETA0 + W_DISRUPT×1.0 ≈ BETA_MAX
# W_ENGAGE × 1.0 (max engagement) → subtracts up to 0.15 from beta → BETA_MIN floor
W_DISRUPT: float = 0.30
W_ENGAGE: float = 0.15

# Drift display smoothing (EMA alpha for UI value)
EMA_ALPHA: float = 0.30

# Z-score cap per signal term
Z_POS_CAP: float = 3.0

# Disruption sigmoid parameters: sigmoid((raw - CENTER) / SCALE)
# v4: CENTER raised 0.35 → 0.45 so mild/moderate signals don't over-trigger.
# disruption_raw must now exceed 0.45 to produce score > 0.5.
DISRUPT_CENTER: float = 0.45
DISRUPT_SCALE: float = 0.25

# Skimming thresholds
SKIM_THRESHOLD: float = 1.3   # pace_ratio > this triggers asymmetric skim signal
SKIM_SCALE: float = 0.5       # (pace_ratio - 1) / this → z_skim

# Disruption component weights (W_D_X × z_X → disruption_raw)
# Ordered by research-backed importance
# v4: W_D_IDLE lowered 0.22 → 0.15 because natural reading pauses produce
#     high idle_seconds (reader sits still while reading a long paragraph).
#     The wider idle_std floor (0.25) further reduces false positives.
W_D_IDLE: float = 0.15        # idle — less dominant (reading pauses are normal)
W_D_FOCUS: float = 0.18       # explicit disengagement (tab-away, blur)
W_D_STAGNATION: float = 0.12  # truly stuck (beyond normal dwell — see v4 fix)
W_D_PACE: float = 0.15        # pace deviation (too fast or too slow)
W_D_SKIM: float = 0.18        # asymmetric fast-skim signal
W_D_REGRESS: float = 0.10     # high back-scroll rate
W_D_JITTER: float = 0.08      # erratic direction changes
W_D_BURSTINESS: float = 0.05  # erratic velocity bursting (v3: higher baseline)

# Stagnation halving: when pace is unavailable AND focus loss is low, the reader
# is likely still reading a long paragraph.  In this case halve z_stagnation
# to avoid falsely penalising careful, deliberate reading.
_STAGNATION_FOCUS_LOSS_THRESH: float = 0.15  # focus_loss_rate below this → halve

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

    v4 fixes:
    - Baseline sanity checks for broken calibration (jitter/regress stored as 0).
    - Wider idle_ratio_std floor (0.25) so reading pauses don't over-penalise.
    - z_stagnation halved when pace unavailable + focus_loss < threshold.
    """
    b = baseline

    # ── Idle ─────────────────────────────────────────────────────────────────
    # v4: floor idle_std at 0.25 (was 0.08).  A reader sitting still on a
    # long paragraph for 20 s naturally produces idle_ratio ≈ 0.8-1.0.  With
    # std=0.08 that was z=6+ (capped at 3), driving drift up hard even during
    # careful reading.  With std=0.25 the same idle_ratio only produces z≈2.
    z_idle = z_pos(
        features.idle_ratio_mean,
        b.get("idle_ratio_mean", 0.35),
        max(b.get("idle_ratio_std", 0.20), 0.25),
    )

    # Focus loss: slightly less hair-trigger than v2.
    # std=0.12 means one unfocused batch in 8 (12.5%) gives z≈1.0.
    z_focus = z_pos(features.focus_loss_rate, 0.0, 0.12)

    # ── Baseline sanity checks for broken scroll calibration (v4) ─────────
    # If the user calibrated when scroll tracking was broken (bug in
    # useTelemetry.ts), scroll_jitter_mean and regress_rate_mean will be
    # stored as ~0.  Using those as the baseline gives astronomically high
    # z-scores for any real jitter/regress in a session.
    # Replace them with realistic population averages until re-calibration.
    _jitter_mu_raw = b.get("scroll_jitter_mean", 0.10)
    _jitter_sigma_raw = b.get("scroll_jitter_std", 0.10)
    if _jitter_mu_raw < 0.04:   # suspiciously low — broken calibration
        _jitter_mu = 0.10
        _jitter_sigma = max(_jitter_sigma_raw, 0.10)
    else:
        _jitter_mu = _jitter_mu_raw
        _jitter_sigma = _jitter_sigma_raw

    _regress_mu_raw = b.get("regress_rate_mean", 0.05)
    _regress_sigma_raw = b.get("regress_rate_std", 0.06)
    if _regress_mu_raw < 0.02:  # suspiciously low — broken calibration
        _regress_mu = 0.05
        _regress_sigma = max(_regress_sigma_raw, 0.06)
    else:
        _regress_mu = _regress_mu_raw
        _regress_sigma = _regress_sigma_raw

    z_jitter = z_pos(
        features.scroll_jitter_mean,
        _jitter_mu,
        max(_jitter_sigma, 0.05),
    )

    z_regress = z_pos(
        features.regress_rate_mean,
        _regress_mu,
        max(_regress_sigma, 0.03),
    )

    z_pause = z_pos(
        features.scroll_pause_mean,
        b.get("idle_seconds_mean", 1.0),
        max(b.get("idle_seconds_std", 1.0), 0.5),
    )

    # ── Stagnation ───────────────────────────────────────────────────────────
    # v3 fix — floor at 0.45 so normal reading (50% of window on one
    # paragraph) produces z_stagnation ≈ 0.  Only true freezing (70%+ above
    # median dwell) triggers a signal.
    stagnation_mu = min(
        max(b.get("para_dwell_median_s", 10.0) / 20.0, 0.45),
        0.85,
    )
    z_stagnation = z_pos(features.stagnation_ratio, stagnation_mu, 0.18)

    # v4 fix — stagnation halving for long-paragraph reading.
    # When pace tracking is unavailable (e.g., first minute, or document has
    # very long paragraphs) AND focus loss is low, we cannot distinguish
    # "reader is still working through a long paragraph" from "reader is
    # stuck / mind-wandering".  Halve the signal to give benefit of the doubt.
    if (
        not features.pace_available
        and features.focus_loss_rate < _STAGNATION_FOCUS_LOSS_THRESH
    ):
        z_stagnation *= 0.5

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

    # Burstiness: v3 fix — natural reading is bursty (scroll-stop-read-scroll).
    # Coefficient of variation ≈ 2–3 is expected; baseline raised from 1.0 to 2.5.
    # Only truly erratic rapid-fire scrolling (CV > 4) generates a signal.
    z_burstiness = z_pos(features.scroll_burstiness, 2.5, 1.0)

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
        # v3 fix: raised from 0.5 → 0.65.
        # Absence of pace evidence is NOT evidence of bad pace.  Give the reader
        # benefit of the doubt during the first ~60 s before paragraph data
        # accumulates.  This prevents the model from penalising normal reading
        # simply because pace estimation hasn't warmed up yet.
        pace_align = 0.65

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
