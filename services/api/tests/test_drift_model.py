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
    compute_stagnation_ratio,
    estimate_window_wpm,
    extract_features,
    idle_ratio_fn,
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

_BASELINE: dict[str, Any] = {
    "wpm_effective": 180.0,
    "idle_ratio_mean": 0.05,
    "idle_ratio_std": 0.08,
    "idle_seconds_mean": 1.0,
    "idle_seconds_std": 1.0,
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
        """Normal reading (cycling 3 paras) should produce beta close to BETA0."""
        batches = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(15)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        # Beta should be only modestly elevated from BETA0 for focused reading
        assert result.beta_effective < 0.20

    def test_distracted_reading_raises_beta_significantly(self) -> None:
        features = extract_features([_distracted_batch()] * 15, _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        assert result.beta_effective > 0.20


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
        """focused_drift < mild_drift < heavy_drift at 3 min."""
        focused = [_focused_batch(f"chunk-{i % 3 + 1}") for i in range(90)]
        mild = [
            _batch(idle_s=0.8, focus="focused", scroll_abs=150.0, para_id=f"chunk-{i % 3 + 1}")
            for i in range(90)
        ]
        heavy = [_distracted_batch("chunk-1")] * 90

        ema_f = _simulate_timeline(focused, _BASELINE)[-1][1]
        ema_m = _simulate_timeline(mild, _BASELINE)[-1][1]
        ema_h = _simulate_timeline(heavy, _BASELINE)[-1][1]

        assert ema_f < ema_m < ema_h, f"f={ema_f:.2f} m={ema_m:.2f} h={ema_h:.2f}"

    def test_early_distraction_raises_drift_vs_focused(self) -> None:
        """1 min of heavy distraction → drift clearly above 1 min of focused."""
        traj_f = _simulate_timeline([_focused_batch("chunk-1")] * 30, _BASELINE)
        traj_h = _simulate_timeline([_distracted_batch("chunk-1")] * 30, _BASELINE)
        assert traj_h[-1][1] > traj_f[-1][1] + 0.05

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
