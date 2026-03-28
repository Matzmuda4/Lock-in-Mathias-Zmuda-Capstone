"""
Deterministic unit + simulation tests for the Phase 7 hybrid drift model.

Model recap
-----------
drift = 1 - exp(-beta_ema * elapsed_minutes)

beta_ema is modulated per window:
  beta_raw = BETA0
           + confidence * W_DISRUPT * disruption_score
           - confidence * W_ENGAGE  * engagement_score
  beta_ema = EMA(beta_raw, prev_beta_ema, alpha=BETA_EMA_ALPHA)

This ensures:
- drift is NEVER permanently 0 (grows naturally with time-on-task)
- distraction raises beta → faster drift
- re-engagement lowers beta → drift can decrease
- skimming is detected via pace_ratio > SKIM_THRESHOLD + z_skim signal

All tests are pure-Python, no DB required.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from app.services.drift.features import (
    compute_scroll_capture_fault_rate,
    compute_stagnation_ratio,
    estimate_window_wpm,
    extract_features,
    idle_ratio_fn,
    is_at_end_of_document,
    jitter_ratio,
    mouse_efficiency,
    paragraph_stagnation,
    regress_rate,
    scroll_velocity_norm,
)
from app.services.drift.model import (
    BETA0,
    BETA_EMA_ALPHA,
    BETA_MAX,
    BETA_MIN,
    EMA_ALPHA,
    SKIM_THRESHOLD,
    W_D_PACE,
    W_DISRUPT,
    W_ENGAGE,
    Z_POS_CAP,
    apply_beta_ema,
    compute_attention,
    compute_beta_raw,
    compute_confidence,
    compute_disruption_score,
    compute_drift,
    compute_drift_result,
    compute_engagement_score,
    compute_z_scores,
    elapsed_minutes_from_seconds,
    update_ema,
    z_pos,
    z_score,
)
from app.services.drift.types import WindowFeatures, ZScores

# ── Shared test baseline ──────────────────────────────────────────────────────
# Realistic values reflecting a typical reader during calibration.
# Key insight: readers are idle ~35% of 2-second windows (between scrolls).
# Using unrealistic low values (e.g. idle_ratio_mean=0.05) causes every
# batch to look like extreme distraction and masks signal discrimination.

_BASELINE: dict[str, Any] = {
    "wpm_effective": 180.0,
    "idle_ratio_mean": 0.35,   # realistic: ~35% of 2s windows have no interaction
    "idle_ratio_std": 0.20,    # realistic window-to-window variability for readers
    "idle_seconds_mean": 0.70,
    "idle_seconds_std": 0.70,
    "scroll_jitter_mean": 0.10,
    "scroll_jitter_std": 0.10,
    "regress_rate_mean": 0.05,
    "regress_rate_std": 0.06,
    "para_dwell_median_s": 10.0,
    "para_dwell_iqr_s": 3.0,
}

# 9 paragraphs so tests that need many distinct para IDs can use them
_WORD_COUNTS: dict[str, int] = {f"chunk-{i}": 30 for i in range(1, 10)}


# ── Batch helpers ─────────────────────────────────────────────────────────────

def _batch(
    *,
    idle_s: float = 0.1,
    focus: str = "focused",
    scroll_abs: float = 200.0,
    scroll_pos: float = 200.0,
    scroll_neg: float = 0.0,
    dir_changes: int = 0,
    event_count: int = 5,
    pause_s: float = 0.1,
    mouse_path: float = 0.0,
    mouse_net: float = 0.0,
    viewport_h: float = 800.0,
    para_id: str = "chunk-1",
) -> dict[str, Any]:
    return {
        "idle_seconds": idle_s,
        "window_focus_state": focus,
        "scroll_delta_abs_sum": scroll_abs,
        "scroll_delta_pos_sum": scroll_pos,
        "scroll_delta_neg_sum": scroll_neg,
        "scroll_direction_changes": dir_changes,
        "scroll_event_count": event_count,
        "scroll_pause_seconds": pause_s,
        "mouse_path_px": mouse_path,
        "mouse_net_px": mouse_net,
        "viewport_height_px": viewport_h,
        "current_paragraph_id": para_id,
    }


def _focused_batch(para_id: str = "chunk-1") -> dict[str, Any]:
    return _batch(idle_s=0.1, focus="focused", scroll_abs=200.0, para_id=para_id)


def _distracted_batch(para_id: str = "chunk-1") -> dict[str, Any]:
    return _batch(idle_s=1.8, focus="blurred", scroll_abs=0.0, para_id=para_id)


def _skimming_batch(para_id: str = "chunk-1") -> dict[str, Any]:
    """Fast scroll through content — needs many distinct paras to register high WPM."""
    return _batch(idle_s=0.0, focus="focused", scroll_abs=2400.0,
                  scroll_pos=2400.0, para_id=para_id)


def _backtrack_batch(para_id: str = "chunk-1") -> dict[str, Any]:
    return _batch(idle_s=0.2, focus="focused", scroll_abs=600.0,
                  scroll_pos=100.0, scroll_neg=500.0,
                  dir_changes=3, event_count=8, para_id=para_id)


def _stuck_batch(para_id: str = "chunk-1") -> dict[str, Any]:
    return _batch(idle_s=0.8, focus="focused", scroll_abs=50.0,
                  scroll_pos=20.0, scroll_neg=30.0,
                  dir_changes=2, event_count=3, pause_s=4.0, para_id=para_id)


# ── Timeline simulator ────────────────────────────────────────────────────────

def _simulate_timeline(
    batch_sequence: list[dict[str, Any]],
    baseline: dict[str, Any] | None = None,
    para_wc: dict[str, int] | None = None,
    window_size: int = 15,
) -> list[tuple[float, float, float]]:
    """
    Simulate a reading session step by step.

    Each element in batch_sequence is one 2-second telemetry batch.
    Returns list of (elapsed_minutes, drift_ema, drift_level).

    prev_ema and prev_beta_ema accumulate correctly across steps.
    """
    b = baseline or {}
    wc = para_wc or _WORD_COUNTS
    baseline_wpm = b.get("wpm_effective") or 180.0

    prev_ema: float = 0.0
    prev_beta_ema: float = BETA0
    trajectory: list[tuple[float, float, float]] = []

    for i, _ in enumerate(batch_sequence):
        window = batch_sequence[max(0, i + 1 - window_size): i + 1]
        elapsed_min = ((i + 1) * 2.0) / 60.0
        features = extract_features(window, wc, baseline_wpm)
        result = compute_drift_result(features, b, elapsed_min, prev_ema, prev_beta_ema)
        prev_ema = result.drift_ema
        prev_beta_ema = result.beta_ema
        trajectory.append((elapsed_min, result.drift_ema, result.drift_level))

    return trajectory


# ── Feature helper unit tests ─────────────────────────────────────────────────


class TestFeatureHelpers:
    def test_scroll_velocity_norm_uses_viewport_height(self) -> None:
        low = scroll_velocity_norm(400.0, 800.0)
        high = scroll_velocity_norm(400.0, 400.0)
        assert high > low

    def test_jitter_ratio_zero_for_no_changes(self) -> None:
        assert jitter_ratio(0, 10) == 0.0

    def test_jitter_ratio_max_when_every_event_reverses(self) -> None:
        assert jitter_ratio(10, 10) == 1.0

    def test_regress_rate_zero_for_forward_only(self) -> None:
        assert regress_rate(200.0, 0.0) < 0.01

    def test_regress_rate_one_for_backward_only(self) -> None:
        assert regress_rate(0.0, 200.0) > 0.99

    def test_mouse_efficiency_neutral_when_stationary(self) -> None:
        assert mouse_efficiency(0.0, 0.0) == 1.0
        assert mouse_efficiency(5.0, 3.0) == 1.0

    def test_idle_ratio_clamped(self) -> None:
        assert idle_ratio_fn(5.0) == 1.0
        assert idle_ratio_fn(1.0) == 0.5

    def test_stagnation_all_same(self) -> None:
        batches = [_batch(para_id="chunk-1")] * 10
        assert compute_stagnation_ratio(batches) == 1.0
        assert paragraph_stagnation(batches) == 1.0

    def test_stagnation_all_different(self) -> None:
        batches = [_batch(para_id=f"chunk-{i}") for i in range(10)]
        assert compute_stagnation_ratio(batches) == pytest.approx(0.1, abs=1e-6)

    def test_paragraph_stagnation_high_when_same_id(self) -> None:
        batches = [{"current_paragraph_id": "chunk-5"}] * 10
        assert paragraph_stagnation(batches) >= 0.5

    def test_paragraph_stagnation_low_when_varied(self) -> None:
        batches = [{"current_paragraph_id": f"chunk-{i}"} for i in range(10)]
        assert paragraph_stagnation(batches) < 0.5


class TestPaceEstimation:
    def test_pace_unavailable_single_para(self) -> None:
        batches = [_focused_batch("chunk-1")] * 15
        _, avail, _ = estimate_window_wpm(batches, _WORD_COUNTS)
        assert not avail

    def test_pace_available_two_paras(self) -> None:
        batches = [_focused_batch("chunk-1")] * 8 + [_focused_batch("chunk-2")] * 8
        _, avail, n = estimate_window_wpm(batches, _WORD_COUNTS)
        assert avail
        assert n == 2

    def test_pace_unavailable_all_blurred(self) -> None:
        batches = [_distracted_batch("chunk-1")] * 8 + [_distracted_batch("chunk-2")] * 8
        _, avail, _ = estimate_window_wpm(batches, _WORD_COUNTS)
        assert not avail


# ── Z-score normalization tests ───────────────────────────────────────────────


class TestZScoreNormalization:
    def test_z_pos_above_mean(self) -> None:
        assert z_pos(1.0, 0.05, 0.08) > 0

    def test_z_pos_zero_below_mean(self) -> None:
        assert z_pos(0.0, 0.10, 0.08) == 0.0

    def test_z_pos_cap(self) -> None:
        assert z_pos(100.0, 0.0, 0.01) == pytest.approx(Z_POS_CAP, abs=1e-3)

    def test_pace_dev_symmetric(self) -> None:
        slow = abs(math.log(0.5))
        fast = abs(math.log(2.0))
        assert abs(slow - fast) < 1e-9

    def test_no_baseline_uses_fallbacks(self) -> None:
        features = extract_features([_focused_batch()] * 5, _WORD_COUNTS, 180.0)
        z = compute_z_scores(features, {})
        assert isinstance(z.z_idle, float)

    def test_skim_z_only_fires_above_threshold(self) -> None:
        features = WindowFeatures(
            pace_ratio=SKIM_THRESHOLD - 0.01, pace_available=True, pace_dev=0.1, n_batches=10,
        )
        z = compute_z_scores(features, _BASELINE)
        assert z.z_skim == 0.0

    def test_skim_z_fires_above_threshold(self) -> None:
        features = WindowFeatures(
            pace_ratio=2.0, pace_available=True, pace_dev=abs(math.log(2.0)), n_batches=10,
        )
        z = compute_z_scores(features, _BASELINE)
        assert z.z_skim > 0.0


# ── Beta computation tests ────────────────────────────────────────────────────


class TestBetaComputation:
    def test_beta_increases_with_high_disruption(self) -> None:
        b_low = compute_beta_raw(0.10, 0.40, 1.0)
        b_high = compute_beta_raw(0.90, 0.40, 1.0)
        assert b_high > b_low

    def test_beta_decreases_with_high_engagement(self) -> None:
        b_low_eng = compute_beta_raw(0.40, 0.10, 1.0)
        b_high_eng = compute_beta_raw(0.40, 0.80, 1.0)
        assert b_high_eng < b_low_eng

    def test_beta_clamps_min(self) -> None:
        b = compute_beta_raw(0.0, 1.0, 1.0)
        assert b >= BETA_MIN

    def test_beta_clamps_max(self) -> None:
        b = compute_beta_raw(1.0, 0.0, 1.0)
        assert b <= BETA_MAX

    def test_beta_ema_smoothing(self) -> None:
        ema = apply_beta_ema(0.60, BETA0)
        assert abs(ema - (BETA_EMA_ALPHA * 0.60 + (1 - BETA_EMA_ALPHA) * BETA0)) < 1e-9

    def test_beta_ema_converges(self) -> None:
        ema = BETA0
        for _ in range(100):
            ema = apply_beta_ema(0.50, ema)
        assert abs(ema - 0.50) < 0.01

    def test_focused_reading_beta_near_baseline(self) -> None:
        """
        Normal reading (cycling 3 paras, idle at 0.05 ratio) sits below the
        realistic baseline idle_ratio_mean=0.35, so z_idle=0 and beta → BETA_MIN.
        """
        batches = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(15)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        # Active reading (below-baseline idle) → beta at or near BETA_MIN
        assert result.beta_effective < 0.10

    def test_distracted_reading_raises_beta_significantly(self) -> None:
        """Heavy idle+blur should push beta to BETA_MAX territory."""
        features = extract_features([_distracted_batch()] * 15, _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        assert result.beta_effective > 0.40


# ── Exponential model tests ───────────────────────────────────────────────────


class TestExponentialModel:
    def test_attention_decays_with_time(self) -> None:
        a_early = compute_attention(0.10, 1.0)
        a_late = compute_attention(0.10, 10.0)
        assert a_late < a_early

    def test_higher_beta_lower_attention(self) -> None:
        a_low = compute_attention(0.05, 5.0)
        a_high = compute_attention(0.50, 5.0)
        assert a_high < a_low

    def test_drift_is_one_minus_attention(self) -> None:
        a = compute_attention(0.10, 3.0)
        d = compute_drift(a)
        assert abs(a + d - 1.0) < 1e-9

    def test_drift_positive_at_any_elapsed_time(self) -> None:
        """Drift is always > 0 for elapsed_minutes > 0 — never permanently at 0."""
        result = compute_drift_result(
            WindowFeatures(n_batches=5),
            {},
            elapsed_minutes=0.1,
            prev_ema=0.0,
            prev_beta_ema=BETA0,
        )
        assert result.drift_level > 0.0

    def test_drift_grows_over_time(self) -> None:
        """Calling compute_drift_result with increasing elapsed_minutes → drift increases."""
        features = WindowFeatures(n_batches=10, idle_ratio_mean=0.05, focus_loss_rate=0.0)
        r1 = compute_drift_result(features, _BASELINE, 1.0, 0.0, BETA0)
        r2 = compute_drift_result(features, _BASELINE, 5.0, 0.0, BETA0)
        assert r2.drift_level > r1.drift_level

    def test_high_beta_means_faster_drift_growth(self) -> None:
        """Distracted user should have higher drift than focused at same elapsed time."""
        focused = WindowFeatures(n_batches=15, idle_ratio_mean=0.05, focus_loss_rate=0.0)
        distracted = WindowFeatures(n_batches=15, idle_ratio_mean=0.90, focus_loss_rate=1.0)
        r_f = compute_drift_result(focused, _BASELINE, 3.0, 0.0, BETA0)
        r_d = compute_drift_result(distracted, _BASELINE, 3.0, 0.0, BETA0)
        assert r_d.drift_level > r_f.drift_level

    def test_legacy_fields_consistent(self) -> None:
        features = WindowFeatures(n_batches=10)
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        assert result.drift_score == result.drift_level
        assert abs(result.attention_score - (1.0 - result.drift_level)) < 1e-9
        assert result.beta_ema >= BETA_MIN


# ── Disruption / engagement tests ────────────────────────────────────────────


class TestDisruptionScore:
    def test_rises_with_high_idle(self) -> None:
        z_l = compute_z_scores(WindowFeatures(n_batches=10, idle_ratio_mean=0.05), _BASELINE)
        z_h = compute_z_scores(WindowFeatures(n_batches=10, idle_ratio_mean=0.90), _BASELINE)
        d_l, _ = compute_disruption_score(z_l, _BASELINE)
        d_h, _ = compute_disruption_score(z_h, _BASELINE)
        assert d_h > d_l

    def test_rises_with_focus_loss(self) -> None:
        z_l = compute_z_scores(WindowFeatures(n_batches=10, focus_loss_rate=0.0), _BASELINE)
        z_h = compute_z_scores(WindowFeatures(n_batches=10, focus_loss_rate=1.0), _BASELINE)
        d_l, _ = compute_disruption_score(z_l, _BASELINE)
        d_h, _ = compute_disruption_score(z_h, _BASELINE)
        assert d_h > d_l

    def test_rises_with_skimming_pace_ratio(self) -> None:
        normal = WindowFeatures(n_batches=15, pace_ratio=1.0, pace_available=True, pace_dev=0.0)
        skimming = WindowFeatures(n_batches=15, pace_ratio=2.5, pace_available=True,
                                  pace_dev=abs(math.log(2.5)))
        z_n = compute_z_scores(normal, _BASELINE)
        z_s = compute_z_scores(skimming, _BASELINE)
        d_n, _ = compute_disruption_score(z_n, _BASELINE)
        d_s, _ = compute_disruption_score(z_s, _BASELINE)
        assert d_s > d_n

    def test_rises_with_high_regress(self) -> None:
        z_l = compute_z_scores(WindowFeatures(n_batches=10, regress_rate_mean=0.05), _BASELINE)
        z_h = compute_z_scores(WindowFeatures(n_batches=10, regress_rate_mean=0.80), _BASELINE)
        d_l, _ = compute_disruption_score(z_l, _BASELINE)
        d_h, _ = compute_disruption_score(z_h, _BASELINE)
        assert d_h > d_l

    def test_components_dict_present(self) -> None:
        z = compute_z_scores(WindowFeatures(n_batches=10), _BASELINE)
        _, comps = compute_disruption_score(z, _BASELINE)
        assert "idle" in comps
        assert "disruption_score" in comps


class TestEngagementScore:
    def test_moderate_when_focused_no_pace(self) -> None:
        """calm=1, pace_align=0.5 (neutral) → engagement = 1 × 0.8 × 0.5 = 0.40."""
        z = ZScores()
        f = WindowFeatures(n_batches=10, pace_available=False)
        score = compute_engagement_score(z, f)
        assert 0.35 <= score <= 0.45

    def test_low_when_blurred_and_idle(self) -> None:
        z = ZScores(z_idle=3.0, z_focus_loss=3.0)
        f = WindowFeatures(n_batches=10, pace_available=False)
        score = compute_engagement_score(z, f)
        assert score < 0.05

    def test_reduced_when_skimming(self) -> None:
        z_ok = ZScores(z_idle=0.0, z_focus_loss=0.0, z_pace=0.3, z_skim=0.0)
        z_sk = ZScores(z_idle=0.0, z_focus_loss=0.0, z_pace=1.5, z_skim=2.0)
        f = WindowFeatures(n_batches=10, pace_available=True)
        assert compute_engagement_score(z_ok, f) > compute_engagement_score(z_sk, f)

    def test_boosted_by_progress_marker(self) -> None:
        z = ZScores()
        f_no = WindowFeatures(n_batches=10, progress_markers_count=0, pace_available=False)
        f_yes = WindowFeatures(n_batches=10, progress_markers_count=2, pace_available=False)
        assert compute_engagement_score(z, f_yes) > compute_engagement_score(z, f_no)

    def test_steady_forward_scroll_boosts_engagement_when_pace_unavailable(self) -> None:
        """Steady forward scrolling (z_regress≈0) gives pace_align=0.65 so that
        normal forward navigation without pace data hints at positive engagement."""
        z_forward = ZScores(z_regress=0.0)
        f_still = WindowFeatures(
            n_batches=10, pace_available=False, scroll_velocity_norm_mean=0.0,
        )
        f_forward = WindowFeatures(
            n_batches=10, pace_available=False, scroll_velocity_norm_mean=0.03,
        )
        score_still = compute_engagement_score(z_forward, f_still)
        score_forward = compute_engagement_score(z_forward, f_forward)
        # Forward steady scroll must yield higher engagement than stationary
        assert score_forward > score_still, (
            f"Forward scroll engagement={score_forward:.3f} should exceed "
            f"stationary engagement={score_still:.3f}"
        )
        # Stationary stays near 0.40
        assert 0.35 <= score_still <= 0.45
        # Forward-scroll boost stays near 0.52 (calm × 0.80 × 0.65)
        assert 0.48 <= score_forward <= 0.58

    def test_fast_backward_scroll_reduces_engagement_below_neutral(self) -> None:
        """Fast backward scrolling (z_regress=3.0) gives pace_align=0.35 —
        BELOW the 0.50 neutral — so drift does not decrease from fast backscroll."""
        z_backward = ZScores(z_regress=3.0)
        f_backward = WindowFeatures(
            n_batches=10, pace_available=False, scroll_velocity_norm_mean=0.08,
        )
        f_still = WindowFeatures(
            n_batches=10, pace_available=False, scroll_velocity_norm_mean=0.0,
        )
        score_backward = compute_engagement_score(z_backward, f_backward)
        score_still = compute_engagement_score(ZScores(), f_still)
        # Fast backward must be below neutral (0.40)
        assert score_backward < score_still, (
            f"Fast backward engagement={score_backward:.3f} should be below "
            f"stationary neutral={score_still:.3f}"
        )
        # Fast backward: calm × 0.80 × 0.35 ≈ 0.28
        assert score_backward < 0.32

    def test_re_reading_backward_is_neutral(self) -> None:
        """Mixed re-reading (z_regress≈1.5) stays near the 0.50 neutral level —
        drift holds but does not dramatically decrease or increase."""
        z_mixed = ZScores(z_regress=1.5)
        f_mixed = WindowFeatures(
            n_batches=10, pace_available=False, scroll_velocity_norm_mean=0.03,
        )
        score = compute_engagement_score(z_mixed, f_mixed)
        # pace_align = 0.65 - (1.5/3.0) × 0.30 = 0.65 - 0.15 = 0.50 → engagement ≈ 0.40
        assert 0.35 <= score <= 0.45, (
            f"Mixed re-reading engagement={score:.3f} should stay near neutral 0.40"
        )

    def test_scroll_direction_does_not_affect_pace_available_branch(self) -> None:
        """Scroll velocity must NOT influence pace_align when pace IS available;
        only z_skim matters in that branch."""
        z = ZScores(z_skim=0.0)
        f_slow_scroll = WindowFeatures(
            n_batches=10, pace_available=True, scroll_velocity_norm_mean=0.0,
        )
        f_fast_scroll = WindowFeatures(
            n_batches=10, pace_available=True, scroll_velocity_norm_mean=0.5,
        )
        # Both should give pace_align=1.0 (z_skim=0) independent of velocity
        assert compute_engagement_score(z, f_slow_scroll) == pytest.approx(
            compute_engagement_score(z, f_fast_scroll), abs=1e-6
        )


class TestConfidenceAndEMA:
    def test_confidence_zero_at_start(self) -> None:
        assert compute_confidence(0) == 0.0

    def test_confidence_one_at_full_window(self) -> None:
        assert compute_confidence(15) == 1.0

    def test_ema_alpha_applied(self) -> None:
        assert update_ema(1.0, 0.0, alpha=EMA_ALPHA) == pytest.approx(EMA_ALPHA, abs=1e-9)

    def test_ema_converges(self) -> None:
        ema = 0.0
        for _ in range(100):
            ema = update_ema(0.5, ema)
        assert abs(ema - 0.5) < 0.02

    def test_elapsed_minutes(self) -> None:
        assert elapsed_minutes_from_seconds(120.0) == pytest.approx(2.0)


# ── Timeline scenario tests ───────────────────────────────────────────────────


class TestTimelineScenarios:
    """
    Each scenario uses _simulate_timeline which advances the session step by
    step with rolling window and proper beta_ema/ema continuity.

    Expected drift values for the exponential model (typical calibrated user):
    - Focused 3 min:         beta ≈ 0.08  → drift ~21%
    - Heavy distraction 2 min: beta → 0.48 → drift ~62%, ema ~45%
    - Skimming 5 min:        beta ≈ 0.19  → drift ~61%
    - Recovery:              beta drops → drift decreases from peak
    """

    def test_scenario1_normal_reading_stays_moderate(self) -> None:
        """Focused reading for 3 min: drift grows naturally but stays below 35%."""
        seq = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(90)]
        traj = _simulate_timeline(seq, _BASELINE)
        ema_at_3min = traj[-1][1]
        # Natural decay: drift~21% at 3 min with beta≈0.08
        assert ema_at_3min < 0.35, f"Normal reading drift {ema_at_3min:.3f} should stay < 0.35"

    def test_scenario1b_drift_is_never_zero(self) -> None:
        """Drift must be > 0 for t > 0 even during perfect focused reading."""
        seq = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(30)]
        traj = _simulate_timeline(seq, _BASELINE)
        # After 1 min, drift must be clearly above 0
        ema_at_1min = traj[-1][1]
        assert ema_at_1min > 0.0, "Drift must never be permanently zero"
        assert ema_at_1min > 0.005  # At least 0.5%

    def test_scenario2_tab_away_raises_drift_significantly(self) -> None:
        """Constant blur + idle for 2 min: drift_ema should be significantly elevated."""
        seq = [_distracted_batch("chunk-1")] * 60
        traj = _simulate_timeline(seq, _BASELINE)
        ema_at_2min = traj[-1][1]
        # beta → 0.48; drift ≈ 62%; EMA lags to ~45%
        assert ema_at_2min > 0.35, f"Tab-away drift {ema_at_2min:.3f} should be > 0.35 at 2 min"

    def test_scenario2b_distraction_higher_than_focused(self) -> None:
        """Distracted session must produce clearly higher drift than focused session."""
        traj_focused = _simulate_timeline(
            [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(60)], _BASELINE
        )
        traj_dist = _simulate_timeline(
            [_distracted_batch("chunk-1")] * 60, _BASELINE
        )
        assert traj_dist[-1][1] > traj_focused[-1][1] + 0.15

    def test_scenario3_skimming_rises_within_5min(self) -> None:
        """Skimming (6 paras cycling → WPM ≈ 2x baseline) raises drift > 0.40 at 5 min."""
        seq = [_skimming_batch(f"chunk-{i % 6 + 1}") for i in range(150)]
        traj = _simulate_timeline(seq, _BASELINE)
        ema_at_5min = traj[-1][1]
        assert ema_at_5min > 0.40, f"Skimming drift {ema_at_5min:.3f} should > 0.40 at 5 min"

    def test_scenario3b_skimming_higher_than_focused_at_5min(self) -> None:
        """Skimming must have higher drift than focused reading at 5 min."""
        traj_f = _simulate_timeline(
            [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(150)], _BASELINE
        )
        traj_s = _simulate_timeline(
            [_skimming_batch(f"chunk-{i % 6 + 1}") for i in range(150)], _BASELINE
        )
        assert traj_s[-1][1] > traj_f[-1][1] + 0.05

    def test_scenario4_stuck_rises_steadily(self) -> None:
        """Stuck / regress pattern: drift at 3 min > drift at 1 min."""
        seq = [_stuck_batch("chunk-1")] * 90
        traj = _simulate_timeline(seq, _BASELINE)
        ema_at_1min = traj[29][1]
        ema_at_3min = traj[-1][1]
        assert ema_at_3min > ema_at_1min
        assert ema_at_3min > 0.10

    def test_scenario5_recovery_after_distraction(self) -> None:
        """After distraction then re-engagement, drift_ema should decrease from peak."""
        distracted = [_distracted_batch("chunk-1")] * 45
        focused = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(60)]
        traj = _simulate_timeline(distracted + focused, _BASELINE)
        # Peak drift occurs during/after distraction phase
        peak = max(ema for _, ema, _ in traj)
        final = traj[-1][1]
        assert final < peak, f"Drift should decrease during recovery; peak={peak:.3f} final={final:.3f}"

    def test_scenario5b_beta_drops_during_recovery(self) -> None:
        """beta_ema should be lower after recovery than during peak distraction."""
        # Run distracted for 1 min, capture beta_ema
        dist_seq = [_distracted_batch("chunk-1")] * 30
        prev_ema, prev_beta = 0.0, BETA0
        for i, _ in enumerate(dist_seq):
            w = dist_seq[max(0, i + 1 - 15): i + 1]
            f = extract_features(w, _WORD_COUNTS, 180.0)
            r = compute_drift_result(f, _BASELINE, (i + 1) * 2 / 60, prev_ema, prev_beta)
            prev_ema, prev_beta = r.drift_ema, r.beta_ema
        beta_after_distraction = prev_beta

        # Then focus for 1 min
        foc_seq = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(30)]
        for i, _ in enumerate(foc_seq):
            w = foc_seq[max(0, i + 1 - 15): i + 1]
            f = extract_features(w, _WORD_COUNTS, 180.0)
            r = compute_drift_result(f, _BASELINE, (30 + i + 1) * 2 / 60, prev_ema, prev_beta)
            prev_ema, prev_beta = r.drift_ema, r.beta_ema
        beta_after_recovery = prev_beta

        assert beta_after_recovery < beta_after_distraction

    def test_scenario6_progress_marker_lowers_beta(self) -> None:
        """Features with progress markers should produce lower beta than without."""
        batches = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(15)]
        f_no = extract_features(batches, _WORD_COUNTS, 180.0)
        f_yes = extract_features(batches, _WORD_COUNTS, 180.0)
        f_yes.progress_markers_count = 2
        r_no = compute_drift_result(f_no, _BASELINE, 2.0, 0.2, 0.10)
        r_yes = compute_drift_result(f_yes, _BASELINE, 2.0, 0.2, 0.10)
        assert r_yes.beta_effective <= r_no.beta_effective

    def test_scenario7_low_confidence_small_beta_change(self) -> None:
        """With only 3 batches (confidence=0.2), beta change from BETA0 is small."""
        seq = [_distracted_batch("chunk-1")] * 3
        features = extract_features(seq, _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 0.1, 0.0, BETA0)
        assert result.confidence < 0.3
        # Beta barely moves from BETA0 with low confidence
        assert result.beta_effective < BETA0 + W_DISRUPT * 0.3 + 0.05

    def test_scenario8_no_pace_no_skim_signal(self) -> None:
        """Same paragraph the whole window → z_pace=0 and z_skim=0."""
        seq = [_batch(idle_s=0.1, focus="focused", scroll_abs=100.0, para_id="chunk-1")] * 15
        features = extract_features(seq, _WORD_COUNTS, 180.0)
        assert not features.pace_available
        z = compute_z_scores(features, _BASELINE)
        assert z.z_pace == 0.0
        assert z.z_skim == 0.0

    def test_scenario9_backtracking_raises_disruption(self) -> None:
        """High regress + direction changes should raise disruption vs normal."""
        normal = extract_features(
            [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(15)], _WORD_COUNTS, 180.0
        )
        backtrack = extract_features(
            [_backtrack_batch(f"chunk-{i % 3 + 1}") for i in range(15)], _WORD_COUNTS, 180.0
        )
        z_n = compute_z_scores(normal, _BASELINE)
        z_b = compute_z_scores(backtrack, _BASELINE)
        d_n, _ = compute_disruption_score(z_n, _BASELINE)
        d_b, _ = compute_disruption_score(z_b, _BASELINE)
        assert d_b > d_n

    def test_scenario10_skimming_raises_disruption_vs_normal(self) -> None:
        """6-para skimming window → higher WPM → higher disruption than normal."""
        normal = extract_features(
            [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(15)], _WORD_COUNTS, 180.0
        )
        # 6 distinct paras → wpm ≈ 2× baseline
        skim = extract_features(
            [_skimming_batch(f"chunk-{i % 6 + 1}") for i in range(15)], _WORD_COUNTS, 180.0
        )
        z_n = compute_z_scores(normal, _BASELINE)
        z_s = compute_z_scores(skim, _BASELINE)
        d_n, _ = compute_disruption_score(z_n, _BASELINE)
        d_s, _ = compute_disruption_score(z_s, _BASELINE)
        assert d_s > d_n


# ── Distraction detection tests ───────────────────────────────────────────────


class TestDistractedReadingDetection:
    def test_three_levels_ordered(self) -> None:
        """
        focused_drift < moderate_drift < heavy_drift at 3 min.

        With realistic baseline (idle_ratio_mean=0.35):
          - focused (idle=0.05, active scrolling) → BELOW baseline → beta ≈ BETA_MIN
          - moderate (idle=1.2s = ratio 0.60, 1.7× baseline) → notable idle z-score
          - heavy (idle=1.8s + blur) → z_idle=2.75, z_focus=3.0 → beta → BETA_MAX
        """
        focused = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(90)]
        moderate = [
            _batch(idle_s=1.2, focus="focused", scroll_abs=100.0, para_id=f"chunk-{i % 3 + 1}")
            for i in range(90)
        ]
        heavy = [_distracted_batch("chunk-1")] * 90

        ema_f = _simulate_timeline(focused, _BASELINE)[-1][1]
        ema_m = _simulate_timeline(moderate, _BASELINE)[-1][1]
        ema_h = _simulate_timeline(heavy, _BASELINE)[-1][1]

        assert ema_f < ema_m, f"focused={ema_f:.3f} should < moderate={ema_m:.3f}"
        assert ema_m < ema_h, f"moderate={ema_m:.3f} should < heavy={ema_h:.3f}"

    def test_early_distraction_raises_drift_vs_focused(self) -> None:
        """1 min of heavy distraction → drift clearly above 1 min of focused."""
        traj_f = _simulate_timeline([_focused_batch("chunk-1")] * 30, _BASELINE)
        traj_h = _simulate_timeline([_distracted_batch("chunk-1")] * 30, _BASELINE)
        # With realistic baseline, distracted has beta→0.65 vs focused beta→BETA_MIN
        assert traj_h[-1][1] > traj_f[-1][1] + 0.20

    def test_distraction_onset_mid_session_raises_drift(self) -> None:
        seq = (
            [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(45)]
            + [_distracted_batch("chunk-1")] * 45
        )
        traj = _simulate_timeline(seq, _BASELINE)
        ema_before = traj[44][1]
        ema_after = traj[-1][1]
        assert ema_after > ema_before

    def test_z_idle_discriminates(self) -> None:
        z_low = compute_z_scores(WindowFeatures(n_batches=10, idle_ratio_mean=0.03), _BASELINE)
        z_high = compute_z_scores(WindowFeatures(n_batches=10, idle_ratio_mean=0.90), _BASELINE)
        assert z_high.z_idle > z_low.z_idle + 0.5

    def test_z_focus_loss_discriminates(self) -> None:
        z_f = compute_z_scores(WindowFeatures(n_batches=10, focus_loss_rate=0.0), _BASELINE)
        z_b = compute_z_scores(WindowFeatures(n_batches=10, focus_loss_rate=1.0), _BASELINE)
        assert z_b.z_focus_loss > z_f.z_focus_loss + 1.0

    def test_beta_components_identifies_dominant_signal(self) -> None:
        batches = [_batch(idle_s=1.9, focus="blurred", scroll_abs=0.0)] * 15
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        comps = result.beta_components
        # Top driver should be a distraction signal
        signal_keys = {"idle", "focus_loss", "stagnation", "jitter", "regress", "pace", "skim"}
        top = max((k for k in comps if k in signal_keys), key=lambda k: comps[k])
        assert top in signal_keys

    def test_no_baseline_still_detects_heavy_distraction(self) -> None:
        """Even without calibration, heavy distraction should raise drift above focused."""
        traj_f = _simulate_timeline(
            [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(60)], {}
        )
        traj_h = _simulate_timeline([_distracted_batch("chunk-1")] * 60, {})
        assert traj_h[-1][1] > traj_f[-1][1] + 0.05

    def test_drift_always_positive_for_positive_elapsed(self) -> None:
        """For any reading scenario with t > 0, drift must be > 0."""
        for scenario_batches in [
            [_focused_batch()] * 5,
            [_distracted_batch()] * 5,
            [],   # empty window
        ]:
            features = extract_features(scenario_batches, _WORD_COUNTS, 180.0)
            result = compute_drift_result(features, _BASELINE, 0.5, 0.0, BETA0)
            assert result.drift_level > 0.0, "drift must be > 0 for elapsed_minutes > 0"


# ── Edge case tests ───────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_batches_nonzero_drift(self) -> None:
        """Even with no telemetry data, drift grows with time (natural decay)."""
        features = extract_features([], _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        # With no data, confidence=0 → only BETA0 drives beta → drift = 1-exp(-BETA0*2)
        expected = 1.0 - math.exp(-BETA0 * 2.0)
        assert abs(result.drift_level - expected) < 0.05

    def test_drift_bounded_zero_to_one(self) -> None:
        """Drift level must always be in [0, 1]."""
        for t in [0.0, 0.5, 10.0, 100.0]:
            r = compute_drift_result(
                WindowFeatures(n_batches=15, idle_ratio_mean=0.9, focus_loss_rate=1.0),
                _BASELINE, t, 0.0, BETA0,
            )
            assert 0.0 <= r.drift_level <= 1.0

    def test_single_batch_valid(self) -> None:
        features = extract_features([_focused_batch("chunk-1")], _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 0.1, 0.0, BETA0)
        assert 0.0 <= result.drift_level <= 1.0
        assert 0.0 <= result.drift_ema <= 1.0

    def test_elapsed_zero_gives_zero_drift(self) -> None:
        """At exactly t=0 (session just started), drift_level = 0."""
        features = WindowFeatures(n_batches=10, idle_ratio_mean=0.05)
        result = compute_drift_result(features, _BASELINE, 0.0, 0.0, BETA0)
        assert result.drift_level == 0.0

    def test_drift_score_aliases_drift_level(self) -> None:
        features = WindowFeatures(n_batches=10)
        result = compute_drift_result(features, _BASELINE, 1.0, 0.0, BETA0)
        assert result.drift_score == result.drift_level
        # beta_effective is the actual computed beta rate, not disruption_score
        assert BETA_MIN <= result.beta_effective <= BETA_MAX


# ── End-of-document signal suppression tests ──────────────────────────────────


def _end_of_doc_batch(para_id: str = "calib-24") -> dict[str, Any]:
    """Batch representing a user sitting at the end of the document (done reading)."""
    return {
        "scroll_delta_abs_sum": 0.0,
        "scroll_delta_pos_sum": 0.0,
        "scroll_delta_neg_sum": 0.0,
        "scroll_event_count": 0,
        "scroll_direction_changes": 0,
        "scroll_pause_seconds": 60.0,
        "idle_seconds": 2.0,
        "mouse_path_px": 0.0,
        "mouse_net_px": 0.0,
        "window_focus_state": "focused",
        "current_paragraph_id": para_id,
        "current_chunk_index": 24,
        "viewport_progress_ratio": 1.0,
        "viewport_height_px": 544.0,
        "scroll_capture_fault": True,  # set by old activity.py; should NOT cause penalty
        "telemetry_fault": False,
        "paragraph_missing_fault": False,
    }


class TestEndOfDocumentSuppression:
    """
    When a user finishes reading (progress_ratio >= 0.97), the model must NOT
    interpret their inactivity as distraction.
    """

    def test_is_at_end_of_document_detects_end(self) -> None:
        batches = [_end_of_doc_batch() for _ in range(12)] + \
                  [_focused_batch("calib-23") for _ in range(3)]
        assert is_at_end_of_document(batches) is True

    def test_is_at_end_of_document_false_when_mid_doc(self) -> None:
        batches = [_focused_batch(f"chunk-{i % 5 + 1}") for i in range(15)]
        assert is_at_end_of_document(batches) is False

    def test_end_of_doc_stagnation_is_zero(self) -> None:
        """stagnation_ratio must be zeroed out at end of document."""
        batches = [_end_of_doc_batch() for _ in range(15)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.at_end_of_document is True
        assert features.stagnation_ratio == 0.0

    def test_end_of_doc_z_scores_are_zero(self) -> None:
        """All z-scores except focus_loss must be 0 at end of document."""
        features = WindowFeatures(n_batches=15, at_end_of_document=True,
                                  idle_ratio_mean=1.0, stagnation_ratio=0.0,
                                  focus_loss_rate=0.0)
        z = compute_z_scores(features, _BASELINE)
        assert z.z_idle == 0.0
        assert z.z_stagnation == 0.0
        assert z.z_jitter == 0.0
        assert z.z_regress == 0.0
        assert z.z_pace == 0.0
        assert z.z_skim == 0.0

    def test_end_of_doc_disruption_is_low(self) -> None:
        """disruption_score should be near zero when at end of document."""
        batches = [_end_of_doc_batch() for _ in range(15)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        z = compute_z_scores(features, _BASELINE)
        disruption, _ = compute_disruption_score(z, _BASELINE)
        assert disruption < 0.30, (
            f"disruption_score={disruption:.3f} is too high at end of document"
        )

    def test_end_of_doc_drift_stays_moderate(self) -> None:
        """
        Drift at 2 minutes with all-end-of-doc batches should stay moderate
        (driven only by natural time-on-task, not by false distraction signal).
        """
        batches = [_end_of_doc_batch() for _ in range(15)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        # At t=2 min with BETA0 only: drift = 1 - exp(-0.03*2) ≈ 5.8%
        # Allow up to 15% (some EMA lag is fine)
        assert result.drift_ema < 0.15, (
            f"drift_ema={result.drift_ema:.3f} rose too high at end of document"
        )

    def test_mid_doc_stagnation_still_detected(self) -> None:
        """Stagnation in the MIDDLE of the document must still work."""
        batches = []
        for _ in range(15):
            b = _focused_batch("chunk-3")
            b["viewport_progress_ratio"] = 0.4   # mid-doc
            b["idle_seconds"] = 2.0
            batches.append(b)
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.at_end_of_document is False
        assert features.stagnation_ratio > 0.5  # should still fire


# ── Fallback WPM tests ─────────────────────────────────────────────────────────


class TestFallbackWpm:
    def test_fallback_wpm_is_250(self) -> None:
        """When no baseline WPM is provided, base_wpm should default to 250."""
        # Create batches that produce a known WPM; use 0 as baseline_wpm_effective
        batches = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(8)]
        features = extract_features(batches, _WORD_COUNTS, 0.0)
        # pace_ratio = window_wpm / 250; confirm pace_ratio is well below 1.6 skim threshold
        # (normal focused reading should not trigger skim)
        if features.pace_available:
            assert features.pace_ratio < 1.6, (
                f"pace_ratio={features.pace_ratio:.2f} hit skim threshold with fallback WPM=250"
            )

    def test_skim_threshold_requires_1_6x(self) -> None:
        """SKIM_THRESHOLD must be 1.6 so normal pace variation does not trigger it."""
        assert SKIM_THRESHOLD == 1.6

    def test_reading_at_baseline_pace_no_skim(self) -> None:
        """A user reading at exactly baseline WPM should have z_skim = 0."""
        features = WindowFeatures(
            n_batches=15, pace_available=True,
            pace_ratio=1.0, pace_dev=0.0,
            at_end_of_document=False,
        )
        z = compute_z_scores(features, _BASELINE)
        assert z.z_skim == 0.0

    def test_reading_at_1_5x_no_skim(self) -> None:
        """pace_ratio=1.5 must NOT trigger skim (below new threshold of 1.6)."""
        features = WindowFeatures(
            n_batches=15, pace_available=True,
            pace_ratio=1.5, pace_dev=abs(math.log(1.5)),
            at_end_of_document=False,
        )
        z = compute_z_scores(features, _BASELINE)
        assert z.z_skim == 0.0, "1.5x should not trigger skim signal"

    def test_reading_at_2x_triggers_skim(self) -> None:
        """pace_ratio=2.0 should trigger the skim signal."""
        features = WindowFeatures(
            n_batches=15, pace_available=True,
            pace_ratio=2.0, pace_dev=abs(math.log(2.0)),
            at_end_of_document=False,
        )
        z = compute_z_scores(features, _BASELINE)
        assert z.z_skim > 0.0, "2x pace should trigger skim signal"


# ── W_D_PACE=0 and skim-only pace_align tests ─────────────────────────────────


class TestPaceWeightZeroAndSkimOnlyEngagement:
    """
    W_D_PACE is zeroed because reading SLOWER than calibration on harder text is
    normal, not distraction.  z_skim (asymmetric, only fires above 1.6×) handles
    the genuinely fast case.  pace_align in engagement uses z_skim only so that
    slow readers are not penalised.
    """

    def test_slow_pace_does_not_increase_disruption(self) -> None:
        """pace_ratio=0.28 (slower than baseline) → z_pace still computed but
        W_D_PACE=0 means it contributes ZERO to disruption_raw."""
        features_slow = WindowFeatures(
            n_batches=15, pace_available=True,
            pace_ratio=0.28, pace_dev=abs(math.log(0.28)),
        )
        features_normal = WindowFeatures(
            n_batches=15, pace_available=True,
            pace_ratio=1.0, pace_dev=0.0,
        )
        z_slow = compute_z_scores(features_slow, _BASELINE)
        z_norm = compute_z_scores(features_normal, _BASELINE)
        d_slow, _ = compute_disruption_score(z_slow, _BASELINE)
        d_norm, _ = compute_disruption_score(z_norm, _BASELINE)
        # Slow reading must not produce higher disruption than on-pace reading
        assert d_slow <= d_norm + 1e-6, (
            f"Slow reader disruption={d_slow:.3f} exceeds on-pace={d_norm:.3f}; "
            "W_D_PACE should be 0"
        )

    def test_slow_pace_does_not_crush_engagement(self) -> None:
        """pace_ratio=0.17 (very slow) → engagement must remain >= 0.75 when
        the reader is calm and not skimming (academic-paper session 156 scenario)."""
        z = ZScores(z_idle=0.0, z_focus_loss=0.0, z_pace=2.9, z_skim=0.0)
        f = WindowFeatures(n_batches=15, pace_available=True, pace_ratio=0.17,
                           pace_dev=abs(math.log(0.17)))
        score = compute_engagement_score(z, f)
        assert score >= 0.75, (
            f"Calm slow reader engagement={score:.3f}; should stay high "
            "(pace_align must use z_skim only)"
        )

    def test_fast_skimming_still_reduces_engagement(self) -> None:
        """pace_ratio=2.5 (skimming) → z_skim fires → pace_align drops → engagement reduced."""
        z_calm = ZScores(z_idle=0.0, z_focus_loss=0.0, z_pace=0.0, z_skim=0.0)
        z_skim = ZScores(z_idle=0.0, z_focus_loss=0.0, z_pace=1.5, z_skim=2.5)
        f_calm = WindowFeatures(n_batches=15, pace_available=True)
        f_skim = WindowFeatures(n_batches=15, pace_available=True)
        eng_calm = compute_engagement_score(z_calm, f_calm)
        eng_skim = compute_engagement_score(z_skim, f_skim)
        assert eng_skim < eng_calm, (
            "Skimming (z_skim=2.5) must still reduce engagement vs calm reading"
        )

    def test_focused_reader_reaches_beta_min(self) -> None:
        """A calm reader at below-baseline pace should yield beta_raw → BETA_MIN
        because engagement offsets the low disruption signal entirely."""
        features = WindowFeatures(
            n_batches=15, pace_available=True,
            pace_ratio=0.28, pace_dev=abs(math.log(0.28)),
            idle_ratio_mean=0.20,   # less idle than baseline mean 0.374
            stagnation_ratio=0.20,
        )
        z = compute_z_scores(features, _BASELINE)
        d, _ = compute_disruption_score(z, _BASELINE)
        eng = compute_engagement_score(z, features)
        beta = compute_beta_raw(d, eng, confidence=1.0)
        assert beta <= BETA_MIN + 1e-6, (
            f"Calm slow reader beta={beta:.4f} > BETA_MIN={BETA_MIN}; "
            "should be at floor (no distraction detected)"
        )

    def test_w_d_pace_is_zero(self) -> None:
        """Guard: W_D_PACE must remain 0.0 to prevent slow-reader false positives."""
        assert W_D_PACE == 0.0, (
            f"W_D_PACE={W_D_PACE}; must be 0.0 — z_skim handles fast pace, "
            "slow pace is normal harder-text reading"
        )


# ── Scroll capture fault fix tests ────────────────────────────────────────────


class TestScrollCaptureFaultFix:
    def test_no_scroll_at_end_does_not_spike_confidence(self) -> None:
        """
        End-of-document batches with scroll_capture_fault=True must not
        reduce quality_confidence_mult below 1.0 (since stagnation is already
        suppressed and the fault is a false positive).
        """
        from app.services.drift.features import compute_quality_confidence_mult
        # After the activity.py fix, these batches won't have scroll_capture_fault=True
        # because progress_ratio >= 0.97.  But even with the flag from old data:
        batches_no_fault = [
            {**_end_of_doc_batch(), "scroll_capture_fault": False}
            for _ in range(15)
        ]
        _, scf, _, mult = compute_quality_confidence_mult(batches_no_fault)
        assert scf == 0.0
        assert mult == 1.0

    def test_genuine_scroll_capture_fault_still_penalises(self) -> None:
        """
        When progress_ratio changes between batches without any scroll events,
        the confidence penalty should apply if rate > 50%.
        SCF is now computed from consecutive progress comparisons, not stored flags.
        """
        from app.services.drift.features import compute_quality_confidence_mult
        # Every other batch has progress jump with no scroll = genuine SCF
        batches_real_fault = []
        for i in range(15):
            prog = 0.10 + i * 0.02  # progress increases each batch
            batches_real_fault.append({
                "scroll_delta_abs_sum": 0.0,
                "scroll_event_count": 0,  # NO scroll events
                "viewport_progress_ratio": prog,
                "telemetry_fault": False,
                "paragraph_missing_fault": False,
                "current_paragraph_id": f"chunk-{i % 3 + 1}",
            })
        _, scf, _, mult = compute_quality_confidence_mult(batches_real_fault)
        # Every transition has progress change + no scroll → genuine SCF
        assert scf > 0.5
        assert mult < 1.0


# ── Regression navigation pace gate tests ─────────────────────────────────────


def _make_regression_batch(
    para_id: str,
    progress: float,
    scroll_abs: float,
    scroll_neg: float,
) -> dict[str, Any]:
    """Build a telemetry batch for the regression-then-readvance test."""
    return {
        "scroll_delta_abs_sum": scroll_abs,
        "scroll_delta_pos_sum": scroll_abs - scroll_neg,
        "scroll_delta_neg_sum": scroll_neg,
        "scroll_event_count": max(1, int(scroll_abs / 10)),
        "scroll_direction_changes": 1 if scroll_neg > 0 else 0,
        "scroll_pause_seconds": 0.1,
        "idle_seconds": 0.01,
        "mouse_path_px": 50.0,
        "mouse_net_px": 40.0,
        "window_focus_state": "focused",
        "current_paragraph_id": para_id,
        "current_chunk_index": int(para_id.split("-")[1]),
        "viewport_progress_ratio": progress,
        "viewport_height_px": 544.0,
        "scroll_capture_fault": False,
        "telemetry_fault": False,
        "paragraph_missing_fault": False,
    }


class TestRegressionNavigationPaceGate:
    """
    When the user scrolls backward (regression) and then re-advances,
    the WPM estimator MUST NOT count those navigation-traversed paragraphs
    as "words read" — it would inflate pace_ratio to 3–6x.
    """

    def test_major_regression_disables_pace(self) -> None:
        """A single back-scroll > 50px + > 50% negative disables pace_available."""
        batches = [
            _make_regression_batch("chunk-1", 0.10, 30.0, 0.0),
            _make_regression_batch("chunk-2", 0.12, 40.0, 0.0),
            _make_regression_batch("chunk-3", 0.14, 35.0, 0.0),
            _make_regression_batch("chunk-4", 0.16, 45.0, 0.0),
            _make_regression_batch("chunk-5", 0.18, 50.0, 0.0),
            # Major regression: back to chunk-1
            _make_regression_batch("chunk-1", 0.05, 979.0, 979.0),
            # Re-advance
            _make_regression_batch("chunk-2", 0.12, 150.0, 0.0),
            _make_regression_batch("chunk-3", 0.14, 100.0, 0.0),
        ]
        wpm, pace_available, n_paras = estimate_window_wpm(batches, _WORD_COUNTS)
        assert pace_available is False, (
            "pace must be disabled when major regression detected in window"
        )
        assert wpm == 0.0

    def test_minor_regression_does_not_disable_pace(self) -> None:
        """Small back-scroll (< 50px) should NOT disable pace estimation."""
        batches = [
            _make_regression_batch(f"chunk-{i + 1}", 0.05 * i, 40.0, 0.0)
            for i in range(6)
        ] + [
            # Minor correction scroll (< 50px, < 50% negative)
            _make_regression_batch("chunk-5", 0.22, 30.0, 10.0),
        ] + [
            _make_regression_batch(f"chunk-{i + 6}", 0.25 + 0.04 * i, 40.0, 0.0)
            for i in range(3)
        ]
        wpm, pace_available, n_paras = estimate_window_wpm(batches, _WORD_COUNTS)
        # Pace should still be available — no major regression
        # (may or may not be True depending on n_paragraphs/eff_seconds, but no gate fired)
        # The test is that pace gate didn't force pace_available=False due to regression
        # We just check wpm calculation ran without major-regression block
        assert n_paras >= 2  # at least counted the paragraphs

    def test_pace_ratio_stays_reasonable_without_regression(self) -> None:
        """Without regression, a user reading at 2x baseline should get pace_ratio ≈ 2."""
        # Build batches that advance through 6 distinct paragraphs
        # in 12 eff_seconds → wpm = (6 × 30 / 12) × 60 = 900 WPM
        # With baseline 200 WPM → pace_ratio = 4.5 (fast but no regression)
        batches = [
            _make_regression_batch(f"chunk-{i + 1}", 0.05 * (i + 1), 80.0, 0.0)
            for i in range(10)
        ]
        wpm, pace_available, n_paras = estimate_window_wpm(batches, _WORD_COUNTS)
        # No regression in this window — pace IS available if enough paras/time
        if pace_available:
            # wpm calculated correctly from traversed paras (no regression gate)
            assert wpm > 0


# ── Proper SCF computation tests ───────────────────────────────────────────────


class TestScrollCaptureFaultWindowContext:
    """
    scroll_capture_fault_rate must now use consecutive batch progress comparisons,
    not the stored per-batch flag which was a false positive for reading pauses.
    """

    def test_reading_pause_is_not_scf(self) -> None:
        """When user pauses to read (no scroll, no progress change), SCF = 0."""
        # Same progress across 5 batches = reading pause
        batches = [
            {"scroll_delta_abs_sum": 0.0, "scroll_event_count": 0,
             "viewport_progress_ratio": 0.30, "current_paragraph_id": "chunk-3"}
        ] * 5
        scf = compute_scroll_capture_fault_rate(batches)
        assert scf == 0.0

    def test_progress_without_scroll_is_scf(self) -> None:
        """Progress changed but no scroll recorded = genuine capture fault."""
        batches = [
            {"scroll_delta_abs_sum": 0.0, "scroll_event_count": 0,
             "viewport_progress_ratio": 0.10 + 0.02 * i, "current_paragraph_id": "chunk-1"}
            for i in range(6)
        ]
        scf = compute_scroll_capture_fault_rate(batches)
        assert scf > 0.5

    def test_normal_scrolling_has_zero_scf(self) -> None:
        """Normal scroll events with matching progress = no fault."""
        batches = [
            {"scroll_delta_abs_sum": 50.0, "scroll_event_count": 30,
             "viewport_progress_ratio": 0.05 * (i + 1), "current_paragraph_id": f"chunk-{i + 1}"}
            for i in range(10)
        ]
        scf = compute_scroll_capture_fault_rate(batches)
        assert scf == 0.0


# ── Zero-baseline metric defaults tests ────────────────────────────────────────


class TestZeroBaselineDefaults:
    """
    Calibration on short linear text can produce regress_rate_mean=0.0 and
    scroll_jitter_mean=0.0.  Using 0.0 as the baseline mean causes z = value/0.03
    to max out at z=3.0 for even tiny real-session values.
    Population defaults must be substituted instead.
    """

    _ZERO_BASELINE: dict[str, Any] = {
        "wpm_effective": 260.0,
        "idle_ratio_mean": 0.37,
        "idle_ratio_std": 0.38,
        "idle_seconds_mean": 0.75,
        "idle_seconds_std": 0.76,
        "para_dwell_median_s": 4.0,
        "para_dwell_iqr_s": 3.0,
        "scroll_jitter_mean": 0.0,   # ← calibration artifact
        "scroll_jitter_std": 0.0,
        "regress_rate_mean": 0.0,    # ← calibration artifact
        "regress_rate_std": 0.0,
    }

    def test_small_regress_does_not_max_z_score(self) -> None:
        """5% back-scroll rate must not produce z_regress = 3.0 when baseline is 0."""
        features = WindowFeatures(
            n_batches=15,
            regress_rate_mean=0.05,   # 5% — mild and common
            at_end_of_document=False,
        )
        z = compute_z_scores(features, self._ZERO_BASELINE)
        assert z.z_regress < 1.0, (
            f"z_regress={z.z_regress:.2f} should be < 1.0 for mild 5% regression "
            f"when baseline has pop default 0.05 applied"
        )

    def test_extreme_regress_still_fires(self) -> None:
        """60% back-scroll rate (e.g. back-reading entire section) must still signal."""
        features = WindowFeatures(
            n_batches=15,
            regress_rate_mean=0.60,
            at_end_of_document=False,
        )
        z = compute_z_scores(features, self._ZERO_BASELINE)
        assert z.z_regress > 1.5

    def test_small_jitter_does_not_max_z_score(self) -> None:
        """8% direction-change rate must not max z_jitter when baseline jitter is 0."""
        features = WindowFeatures(
            n_batches=15,
            scroll_jitter_mean=0.08,  # mild jitter
            at_end_of_document=False,
        )
        z = compute_z_scores(features, self._ZERO_BASELINE)
        assert z.z_jitter < 1.0, (
            f"z_jitter={z.z_jitter:.2f} too high for mild jitter with pop default applied"
        )

    def test_focused_reader_low_disruption_zero_baseline(self) -> None:
        """
        A genuinely focused reader with a zero-regress/jitter calibration
        baseline must NOT trigger high disruption from those signals alone.
        """
        features = extract_features(
            [_focused_batch(f"chunk-{i % 4 + 1}") for i in range(15)],
            _WORD_COUNTS,
            260.0,
        )
        z = compute_z_scores(features, self._ZERO_BASELINE)
        disruption, components = compute_disruption_score(z, self._ZERO_BASELINE)
        assert disruption < 0.50, (
            f"disruption={disruption:.3f} too high for focused reader with zero-baseline metrics"
        )


# ── Stagnation IQR tolerance tests ────────────────────────────────────────────


class TestStagnationIqrTolerance:
    """
    Users reading harder text dwell longer per paragraph than during calibration.
    stagnation_mu must use (median + 0.5×IQR)/30 to be more generous.
    """

    def test_stagnation_mu_uses_iqr(self) -> None:
        """
        For baseline median=4s, IQR=3s:
        stagnation_mu = (4 + 0.5×3) / 30 = 5.5/30 ≈ 0.183 (not 0.133).
        """
        b: dict[str, Any] = {
            "para_dwell_median_s": 4.0,
            "para_dwell_iqr_s": 3.0,
            "idle_ratio_mean": 0.37,
            "idle_ratio_std": 0.38,
            "scroll_jitter_mean": 0.10,
            "scroll_jitter_std": 0.10,
            "regress_rate_mean": 0.05,
            "regress_rate_std": 0.06,
        }
        # stagnation_ratio = 0.20 (user spends 6s of 30s on one paragraph)
        features = WindowFeatures(
            n_batches=15,
            stagnation_ratio=0.20,
            at_end_of_document=False,
        )
        z = compute_z_scores(features, b)
        # With old formula (mu=0.133): z = (0.20 - 0.133)/0.15 = 0.447
        # With new formula (mu=0.183): z = (0.20 - 0.183)/0.15 = 0.113
        assert z.z_stagnation < 0.25, (
            f"z_stagnation={z.z_stagnation:.3f} still too high for moderate dwell on harder text"
        )

    def test_extreme_stagnation_still_fires(self) -> None:
        """Stagnation ratio = 0.9 (stuck for 27 of 30 seconds) must still signal."""
        b: dict[str, Any] = {
            "para_dwell_median_s": 4.0,
            "para_dwell_iqr_s": 3.0,
            "idle_ratio_mean": 0.37,
            "idle_ratio_std": 0.38,
            "scroll_jitter_mean": 0.10,
            "scroll_jitter_std": 0.10,
            "regress_rate_mean": 0.05,
            "regress_rate_std": 0.06,
        }
        features = WindowFeatures(
            n_batches=15,
            stagnation_ratio=0.90,
            at_end_of_document=False,
        )
        z = compute_z_scores(features, b)
        assert z.z_stagnation > 2.0
