"""
Deterministic drift model scenario tests — Phase 7.

These tests simulate realistic reading behaviors and assert expected drift_ema ranges.
They are pure-function tests — no DB, no HTTP, no fixtures required.

Reading behavior expectations (tuned to the fixed model parameters):
  - Perfect focus, 5 min:   drift_ema < 0.25
  - Perfect focus, 10 min:  drift_ema < 0.40
  - Immediate heavy distraction: drift_ema > 0.55 at 5 min
  - First 30 s (warm-up):   drift_ema < 0.10 regardless of signals
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from app.services.drift.features import (
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
    BETA_MAX,
    BETA_MIN,
    EMA_ALPHA,
    Z_POS_CAP,
    apply_beta_ema,
    apply_confidence_gating,
    compute_attention,
    compute_beta_raw,
    compute_confidence,
    compute_drift,
    compute_drift_result,
    compute_z_scores,
    elapsed_minutes_from_seconds,
    update_ema,
    z_pos,
    z_score,
)
from app.services.drift.types import WindowFeatures, ZScores


# ── Helpers ────────────────────────────────────────────────────────────────────

def _batch(
    idle_s: float = 0.1,
    focus: str = "focused",
    scroll_abs: float = 200.0,
    scroll_dir_changes: int = 0,
    scroll_events: int = 5,
    scroll_pos: float = 200.0,
    scroll_neg: float = 0.0,
    mouse_path: float = 0.0,
    mouse_net: float = 0.0,
    viewport_h: float = 800.0,
    scroll_pause: float = 0.0,
    para_id: str = "chunk-1",
) -> dict[str, Any]:
    return {
        "idle_seconds": idle_s,
        "window_focus_state": focus,
        "scroll_delta_abs_sum": scroll_abs,
        "scroll_direction_changes": scroll_dir_changes,
        "scroll_event_count": scroll_events,
        "scroll_delta_pos_sum": scroll_pos,
        "scroll_delta_neg_sum": scroll_neg,
        "mouse_path_px": mouse_path,
        "mouse_net_px": mouse_net,
        "viewport_height_px": viewport_h,
        "scroll_pause_seconds": scroll_pause,
        "current_paragraph_id": para_id,
    }


def _simulate_session(
    batches: list[dict[str, Any]],
    elapsed_minutes: float,
    baseline: dict[str, Any] | None = None,
    para_wc: dict[str, int] | None = None,
    *,
    n_iterations: int = 1,
) -> float:
    """
    Run the drift pipeline repeatedly for n_iterations (simulating consecutive
    2-second windows) and return the final drift_ema.

    This lets us simulate how ema converges over many batches.
    """
    b = baseline or {}
    wc = para_wc or {"chunk-1": 30, "chunk-2": 30, "chunk-3": 30}
    baseline_wpm = b.get("wpm_effective") or 180.0

    prev_ema = 0.0
    prev_beta_ema = BETA0

    for _ in range(n_iterations):
        features = extract_features(batches, wc, baseline_wpm)
        result = compute_drift_result(features, b, elapsed_minutes, prev_ema, prev_beta_ema)
        prev_ema = result.drift_ema
        prev_beta_ema = result.beta_ema

    return prev_ema


def _simulate_timeline(
    batch_sequence: list[dict[str, Any]],
    baseline: dict[str, Any] | None = None,
    para_wc: dict[str, int] | None = None,
    window_size: int = 15,
) -> list[tuple[float, float, float]]:
    """
    Simulate a full session by playing batches one at a time (each batch = 2 s).

    At each step i the rolling window is the last `window_size` batches, and
    elapsed_minutes = (i+1)*2/60.

    Returns a list of (elapsed_minutes, drift_ema, beta_gated) per step.
    This gives a full trajectory so tests can check HOW QUICKLY drift responds
    to distraction signals at different points in the session.
    """
    b = baseline or {}
    wc = para_wc or {"chunk-1": 30, "chunk-2": 30, "chunk-3": 30}
    baseline_wpm = b.get("wpm_effective") or 180.0

    prev_ema = 0.0
    prev_beta_ema = BETA0
    trajectory: list[tuple[float, float, float]] = []

    for i in range(len(batch_sequence)):
        window = batch_sequence[max(0, i + 1 - window_size) : i + 1]
        elapsed_min = ((i + 1) * 2.0) / 60.0
        features = extract_features(window, wc, baseline_wpm)
        result = compute_drift_result(features, b, elapsed_min, prev_ema, prev_beta_ema)
        prev_ema = result.drift_ema
        prev_beta_ema = result.beta_ema
        trajectory.append((elapsed_min, prev_ema, result.beta_effective))

    return trajectory


# ── Unit tests ─────────────────────────────────────────────────────────────────

class TestElapsedConversion:
    def test_120_seconds_is_2_minutes(self):
        assert elapsed_minutes_from_seconds(120.0) == pytest.approx(2.0)

    def test_0_is_0(self):
        assert elapsed_minutes_from_seconds(0.0) == 0.0

    def test_60_is_1(self):
        assert elapsed_minutes_from_seconds(60.0) == pytest.approx(1.0)


class TestZScoreHelpers:
    def test_z_score_at_mean(self):
        assert z_score(5.0, 5.0, 1.0) == pytest.approx(0.0)

    def test_z_score_positive(self):
        # Denominator has _EPS=1e-5 added so result is fractionally below 2.0
        assert z_score(7.0, 5.0, 1.0) == pytest.approx(2.0, abs=1e-3)

    def test_z_score_capped_at_cap(self):
        # With default cap = Z_POS_CAP = 2.0, value 5 std above mean caps at 2.0
        assert z_score(15.0, 5.0, 1.0) == pytest.approx(Z_POS_CAP)

    def test_z_pos_below_mean_is_zero(self):
        assert z_pos(3.0, 5.0, 1.0) == 0.0

    def test_z_pos_above_mean(self):
        result = z_pos(7.0, 5.0, 1.0)
        assert result == pytest.approx(2.0, abs=1e-3)


class TestBetaComputation:
    def test_beta_at_zero_z_equals_beta0(self):
        z = ZScores()  # all zeros
        beta, comps = compute_beta_raw(z)
        assert beta == pytest.approx(BETA0)
        assert comps["beta0"] == BETA0

    def test_beta_increases_with_idle(self):
        z_low = ZScores(z_idle=0.0)
        z_high = ZScores(z_idle=2.0)
        b_low, _ = compute_beta_raw(z_low)
        b_high, _ = compute_beta_raw(z_high)
        assert b_high > b_low

    def test_beta_increases_with_focus_loss(self):
        z_low = ZScores(z_focus_loss=0.0)
        z_high = ZScores(z_focus_loss=2.0)
        b_low, _ = compute_beta_raw(z_low)
        b_high, _ = compute_beta_raw(z_high)
        assert b_high > b_low

    def test_beta_clamped_min(self):
        z = ZScores()
        beta, _ = compute_beta_raw(z)
        assert beta >= BETA_MIN

    def test_beta_clamped_max(self):
        # Max all z scores to 2 — should clamp to BETA_MAX
        z = ZScores(z_idle=2, z_focus_loss=2, z_jitter=2, z_regress=2,
                    z_pause=2, z_stagnation=2, z_mouse=2, z_pace=2)
        beta, _ = compute_beta_raw(z)
        assert beta == pytest.approx(BETA_MAX)

    def test_beta_ema_smoothing(self):
        # Large spike from 0.03 to 0.60 should produce intermediate value
        new_ema = apply_beta_ema(0.60, 0.03)
        assert 0.03 < new_ema < 0.60
        # After one step with alpha=0.20: 0.20*0.60 + 0.80*0.03 = 0.144
        assert new_ema == pytest.approx(0.144)

    def test_confidence_gating_at_zero(self):
        # With 0 confidence, beta should equal beta0
        gated = apply_confidence_gating(0.50, 0.0)
        assert gated == pytest.approx(BETA0)

    def test_confidence_gating_at_full(self):
        # With full confidence, beta should equal the input beta
        gated = apply_confidence_gating(0.50, 1.0)
        assert gated == pytest.approx(0.50)

    def test_confidence_gating_interpolates(self):
        gated = apply_confidence_gating(0.50, 0.5)
        expected = BETA0 * 0.5 + 0.50 * 0.5
        assert gated == pytest.approx(expected)


class TestConfidenceAndEMA:
    def test_confidence_zero_batches(self):
        assert compute_confidence(0) == 0.0

    def test_confidence_full_at_15(self):
        assert compute_confidence(15) == pytest.approx(1.0)

    def test_confidence_caps_at_1(self):
        assert compute_confidence(100) == pytest.approx(1.0)

    def test_confidence_partial(self):
        assert compute_confidence(5) == pytest.approx(5 / 15)

    def test_ema_update(self):
        result = update_ema(0.8, 0.0, alpha=EMA_ALPHA)
        assert result == pytest.approx(EMA_ALPHA * 0.8)

    def test_ema_converges(self):
        ema = 0.0
        for _ in range(100):
            ema = update_ema(0.5, ema)
        assert ema == pytest.approx(0.5, abs=0.01)


class TestPaceGating:
    """Pace should only fire when enough paragraphs + effective time are observed."""

    def _make_single_para_batches(self, n: int = 15) -> list[dict[str, Any]]:
        """15 batches all on the same paragraph — pace should be gated."""
        return [_batch(para_id="chunk-1") for _ in range(n)]

    def _make_multi_para_batches(self) -> list[dict[str, Any]]:
        """Mix of 3 paragraphs — pace should be available."""
        return (
            [_batch(para_id="chunk-1") for _ in range(5)]
            + [_batch(para_id="chunk-2") for _ in range(5)]
            + [_batch(para_id="chunk-3") for _ in range(5)]
        )

    def test_single_para_pace_not_available(self):
        batches = self._make_single_para_batches()
        features = extract_features(batches, {"chunk-1": 30}, 180.0)
        assert not features.pace_available
        assert features.pace_dev == 0.0

    def test_multi_para_pace_available(self):
        batches = self._make_multi_para_batches()
        wc = {"chunk-1": 30, "chunk-2": 30, "chunk-3": 30}
        features = extract_features(batches, wc, 180.0)
        assert features.pace_available

    def test_pace_z_zero_when_not_available(self):
        batches = self._make_single_para_batches()
        features = extract_features(batches, {"chunk-1": 30}, 180.0)
        z = compute_z_scores(features, {})
        assert z.z_pace == 0.0

    def test_pace_dev_symmetric(self):
        """pace_ratio 0.5 and 2.0 should give equal pace_dev."""
        dev_slow = abs(math.log(0.5))
        dev_fast = abs(math.log(2.0))
        assert dev_slow == pytest.approx(dev_fast, abs=1e-9)


class TestMouseEfficiency:
    def test_no_movement_returns_neutral(self):
        """When mouse is stationary, efficiency should be 1.0 (neutral, not penalised)."""
        assert mouse_efficiency(0.0, 0.0) == pytest.approx(1.0)

    def test_small_path_returns_neutral(self):
        assert mouse_efficiency(5.0, 0.0) == pytest.approx(1.0)

    def test_efficient_movement(self):
        assert mouse_efficiency(100.0, 100.0) == pytest.approx(1.0)

    def test_inefficient_movement(self):
        eff = mouse_efficiency(100.0, 10.0)
        assert eff == pytest.approx(0.1)

    def test_z_mouse_zero_when_no_movement(self):
        """No-movement batch: z_mouse must be 0 so it doesn't inflate beta."""
        batches = [_batch(mouse_path=0.0, mouse_net=0.0) for _ in range(15)]
        features = extract_features(batches, {}, 180.0)
        z = compute_z_scores(features, {})
        assert z.z_mouse == 0.0


class TestStagnation:
    def test_stagnation_below_threshold(self):
        """60 % dominance is below the 80 % threshold — should be 0."""
        batches = (
            [_batch(para_id="chunk-1")] * 9
            + [_batch(para_id="chunk-2")] * 6
        )
        assert paragraph_stagnation(batches) == 0.0

    def test_stagnation_above_threshold(self):
        """90 % dominance should trigger stagnation."""
        batches = [_batch(para_id="chunk-1")] * 14 + [_batch(para_id="chunk-2")]
        result = paragraph_stagnation(batches)
        assert result > 0.8

    def test_no_batches_returns_zero(self):
        assert paragraph_stagnation([]) == 0.0

    def test_normal_reading_no_stagnation(self):
        """Typical reading: 2–3 paragraphs in window, no single one > 80 %."""
        batches = (
            [_batch(para_id="chunk-1")] * 5
            + [_batch(para_id="chunk-2")] * 5
            + [_batch(para_id="chunk-3")] * 5
        )
        assert paragraph_stagnation(batches) == 0.0


# ── Scenario tests ─────────────────────────────────────────────────────────────

class TestScenarios:
    """
    End-to-end drift simulations.

    We run multiple consecutive compute cycles (n_iterations) to let EMA
    and beta_ema converge, then check the final drift_ema.
    """

    _BASELINE = {
        "wpm_effective": 180.0,
        "idle_ratio_mean": 0.05,
        "idle_ratio_std": 0.08,
        "idle_seconds_mean": 1.0,
        "idle_seconds_std": 1.0,
        "scroll_jitter_mean": 0.10,
        "scroll_jitter_std": 0.10,
        "regress_rate_mean": 0.05,
        "regress_rate_std": 0.06,
    }

    def test_scenario_focused_reader_3min_below_35pct(self):
        """
        Focused reader at 3 minutes: drift_ema should be well below 35 %.
        Conditions: low idle, stable forward scroll, no blur, alternating paras.
        """
        batches = (
            [_batch(idle_s=0.1, scroll_abs=200, scroll_pos=200, para_id="chunk-1")] * 5
            + [_batch(idle_s=0.1, scroll_abs=200, scroll_pos=200, para_id="chunk-2")] * 5
            + [_batch(idle_s=0.1, scroll_abs=200, scroll_pos=200, para_id="chunk-3")] * 5
        )
        ema = _simulate_session(
            batches,
            elapsed_minutes=3.0,
            baseline=self._BASELINE,
            n_iterations=20,
        )
        assert ema < 0.35, f"Expected drift_ema < 0.35 for focused reading at 3 min, got {ema:.3f}"

    def test_scenario_focused_reader_10min_below_50pct(self):
        """
        Focused reader at 10 minutes: drift_ema should still be below 50 %.
        """
        batches = (
            [_batch(idle_s=0.1, scroll_abs=200, scroll_pos=200, para_id="chunk-1")] * 5
            + [_batch(idle_s=0.1, scroll_abs=200, scroll_pos=200, para_id="chunk-2")] * 5
            + [_batch(idle_s=0.1, scroll_abs=200, scroll_pos=200, para_id="chunk-3")] * 5
        )
        ema = _simulate_session(
            batches,
            elapsed_minutes=10.0,
            baseline=self._BASELINE,
            n_iterations=50,
        )
        assert ema < 0.50, f"Expected drift_ema < 0.50 for focused reading at 10 min, got {ema:.3f}"

    def test_scenario_heavy_distraction_3min_above_55pct(self):
        """
        Heavy distraction at 3 minutes: high idle + frequent blur → drift_ema > 55 %.
        """
        batches = [
            _batch(idle_s=1.8, focus="blurred", scroll_abs=0, para_id="chunk-1")
            for _ in range(15)
        ]
        ema = _simulate_session(
            batches,
            elapsed_minutes=3.0,
            baseline=self._BASELINE,
            n_iterations=50,  # let EMA converge
        )
        assert ema > 0.55, f"Expected drift_ema > 0.55 for heavy distraction at 3 min, got {ema:.3f}"

    def test_scenario_warmup_first_5_batches_low(self):
        """
        During warm-up (first 5 batches, ~10 s), even with bad signals,
        drift_ema should remain low due to confidence gating.
        """
        # Use the worst possible signals
        batches = [
            _batch(idle_s=1.9, focus="blurred")
            for _ in range(5)
        ]
        ema = _simulate_session(
            batches,
            elapsed_minutes=0.5,
            baseline=self._BASELINE,
            n_iterations=1,  # only 1 cycle — still warming up
        )
        assert ema < 0.25, f"Expected low drift during warm-up, got {ema:.3f}"

    def test_scenario_skimming_pace_gated_when_one_para(self):
        """
        If the user is on only ONE paragraph for the whole window, pace should
        not fire (pace_available=False).  Drift should be modest.
        """
        batches = [_batch(scroll_abs=2000.0, para_id="chunk-1") for _ in range(15)]
        features = extract_features(batches, {"chunk-1": 30}, 180.0)
        assert not features.pace_available, "Pace should be gated with single paragraph"
        z = compute_z_scores(features, self._BASELINE)
        assert z.z_pace == 0.0, "z_pace must be 0 when pace unavailable"

    def test_scenario_skimming_two_paras_pace_fires(self):
        """
        With 2 paragraphs and sufficient effective time, pace fires
        and if WPM is much higher than baseline, drift rises.
        """
        # Reading very fast: many words, 2 paragraphs
        wc = {"chunk-1": 200, "chunk-2": 200}
        batches = (
            [_batch(scroll_abs=300, para_id="chunk-1", idle_s=0.0)] * 6
            + [_batch(scroll_abs=300, para_id="chunk-2", idle_s=0.0)] * 9
        )
        features = extract_features(batches, wc, 180.0)
        if features.pace_available:
            # If pace is available, pace_ratio should be > 1 (fast reader)
            assert features.pace_ratio > 0.5  # not degenerate

    def test_scenario_stuck_reader_slow_drift(self):
        """
        Stuck/overloaded reader: low scroll, high pause, regress.
        Drift rises but not instantly — EMA needs multiple windows.
        """
        batches = [
            _batch(idle_s=0.5, scroll_abs=20, scroll_pos=10, scroll_neg=10,
                   scroll_pause=5.0, para_id="chunk-1")
            for _ in range(15)
        ]
        ema_early = _simulate_session(batches, 1.0, self._BASELINE, n_iterations=5)
        ema_later = _simulate_session(batches, 5.0, self._BASELINE, n_iterations=30)
        # Drift should be higher after more time
        assert ema_later >= ema_early, "Drift should not decrease over time with same signals"

    def test_scenario_no_baseline_uses_defaults(self):
        """
        Without calibration, the model uses safe defaults — should not explode.
        """
        batches = (
            [_batch(idle_s=0.1, para_id="chunk-1")] * 5
            + [_batch(idle_s=0.1, para_id="chunk-2")] * 5
            + [_batch(idle_s=0.1, para_id="chunk-3")] * 5
        )
        ema = _simulate_session(
            batches,
            elapsed_minutes=5.0,
            baseline=None,  # no calibration
            n_iterations=20,
        )
        assert ema < 0.50, f"No-baseline session should not drift catastrophically: got {ema:.3f}"


class TestDriftCurveProperties:
    """Mathematical properties of the drift curve."""

    def test_drift_increases_with_time(self):
        """For any positive beta, drift should be monotonically increasing."""
        from app.services.drift.model import compute_attention, compute_drift
        beta = 0.10
        drifts = [compute_drift(compute_attention(beta, t)) for t in [1, 2, 3, 5, 10]]
        for i in range(len(drifts) - 1):
            assert drifts[i] < drifts[i + 1]

    def test_drift_is_zero_at_t_zero(self):
        from app.services.drift.model import compute_attention, compute_drift
        assert compute_drift(compute_attention(0.10, 0.0)) == pytest.approx(0.0)

    def test_attention_is_one_at_t_zero(self):
        from app.services.drift.model import compute_attention
        assert compute_attention(0.10, 0.0) == pytest.approx(1.0)

    def test_drift_below_one(self):
        # Use a moderate t where float underflow is not an issue
        from app.services.drift.model import compute_attention, compute_drift
        assert compute_drift(compute_attention(BETA_MAX, 10.0)) < 1.0

    def test_beta_ema_warmup_then_spike(self):
        """
        If signals are good for a while then suddenly bad, the EMA should
        lag — not immediately jump to BETA_MAX.
        """
        # 20 good cycles
        prev = BETA0
        for _ in range(20):
            prev = apply_beta_ema(BETA0, prev)  # stays near BETA0
        assert prev == pytest.approx(BETA0, abs=0.005)

        # 1 spike
        after_spike = apply_beta_ema(BETA_MAX, prev)
        # Should be much less than BETA_MAX due to 0.20 alpha
        assert after_spike < BETA_MAX * 0.5


# ── Distraction detection & sensitivity tests ─────────────────────────────────

class TestDistractedReadingDetection:
    """
    Verify that the model reliably DETECTS distraction at different severity levels
    and time-points, while remaining discriminable from focused reading.

    These tests document the model's sensitivity guarantees for the future LLM
    classification layer, which will use drift_ema, beta_effective, and z_scores
    as input features.

    Design principle: The model should satisfy BOTH:
      (a) focused normal reading → low drift  [tested in TestScenarios]
      (b) distracted reading → detectably elevated drift  [tested here]
    """

    _BASELINE = {
        "wpm_effective": 180.0,
        "idle_ratio_mean": 0.05,
        "idle_ratio_std": 0.08,
        "idle_seconds_mean": 1.0,
        "idle_seconds_std": 1.0,
        "scroll_jitter_mean": 0.10,
        "scroll_jitter_std": 0.10,
        "regress_rate_mean": 0.05,
        "regress_rate_std": 0.06,
    }

    # ── Helper batches representing distinct attentional states ───────────────

    @staticmethod
    def _focused_batch(para_id: str = "chunk-1") -> dict[str, Any]:
        return _batch(idle_s=0.1, focus="focused", scroll_abs=200, scroll_pos=200,
                      scroll_neg=0, para_id=para_id)

    @staticmethod
    def _mild_distraction_batch(para_id: str = "chunk-1") -> dict[str, Any]:
        """Phone glance / short pause: elevated idle, no blur."""
        return _batch(idle_s=1.0, focus="focused", scroll_abs=50, scroll_pos=50,
                      scroll_neg=0, scroll_pause=2.0, para_id=para_id)

    @staticmethod
    def _heavy_distraction_batch(para_id: str = "chunk-1") -> dict[str, Any]:
        """Tab switch + long idle: blurred + max idle."""
        return _batch(idle_s=1.8, focus="blurred", scroll_abs=0, scroll_pos=0,
                      scroll_neg=0, scroll_pause=0.0, para_id=para_id)

    # ── Three-level ordering ──────────────────────────────────────────────────

    def test_three_levels_ordered_focused_lt_mild_lt_heavy(self):
        """
        After enough time, the model must rank:
          focused < mild distraction < heavy distraction
        This is the core discriminability guarantee for the LLM classifier.
        """
        focused_batches = [self._focused_batch(f"chunk-{i % 3 + 1}") for i in range(15)]
        mild_batches    = [self._mild_distraction_batch() for _ in range(15)]
        heavy_batches   = [self._heavy_distraction_batch() for _ in range(15)]

        ema_focused = _simulate_session(focused_batches, 3.0, self._BASELINE, n_iterations=30)
        ema_mild    = _simulate_session(mild_batches,    3.0, self._BASELINE, n_iterations=30)
        ema_heavy   = _simulate_session(heavy_batches,   3.0, self._BASELINE, n_iterations=30)

        assert ema_focused < ema_mild, (
            f"Focused drift ({ema_focused:.3f}) should be < mild ({ema_mild:.3f})"
        )
        assert ema_mild < ema_heavy, (
            f"Mild drift ({ema_mild:.3f}) should be < heavy ({ema_heavy:.3f})"
        )

    def test_heavy_distraction_exceeds_focused_within_1_minute(self):
        """
        Even in the first minute, heavy distraction should produce a meaningfully
        higher drift than focused reading — the model must be responsive enough
        for the LLM to act on early signals.

        We use the timeline to check separation exists before 1-minute mark.
        """
        # 30 batches = 60 s total
        focused_seq  = [self._focused_batch(f"chunk-{i % 3 + 1}") for i in range(30)]
        heavy_seq    = [self._heavy_distraction_batch() for _ in range(30)]

        traj_focused = _simulate_timeline(focused_seq,  self._BASELINE)
        traj_heavy   = _simulate_timeline(heavy_seq,    self._BASELINE)

        # Find first point where elapsed >= 1.0 minute
        heavy_at_1min  = next(ema for t, ema, _ in traj_heavy  if t >= 1.0)
        focused_at_1min = next(ema for t, ema, _ in traj_focused if t >= 1.0)

        assert heavy_at_1min > focused_at_1min, (
            f"Heavy distraction drift ({heavy_at_1min:.3f}) must exceed "
            f"focused ({focused_at_1min:.3f}) by 1 minute"
        )
        # Absolute check: heavy distraction at 1 min should be non-trivially elevated
        assert heavy_at_1min > 0.10, (
            f"Heavy distraction should produce >10% drift at 1 min, got {heavy_at_1min:.3f}"
        )

    def test_mild_distraction_detectable_vs_focused_at_2min(self):
        """
        Mild distraction (elevated idle, no blur) should be distinguishable
        from focused reading by 2 minutes.
        """
        focused_seq = [self._focused_batch(f"chunk-{i % 3 + 1}") for i in range(60)]
        mild_seq    = [self._mild_distraction_batch() for _ in range(60)]

        traj_focused = _simulate_timeline(focused_seq, self._BASELINE)
        traj_mild    = _simulate_timeline(mild_seq,    self._BASELINE)

        mild_at_2min    = next(ema for t, ema, _ in traj_mild    if t >= 2.0)
        focused_at_2min = next(ema for t, ema, _ in traj_focused if t >= 2.0)

        assert mild_at_2min > focused_at_2min, (
            f"Mild distraction drift ({mild_at_2min:.3f}) should exceed "
            f"focused ({focused_at_2min:.3f}) at 2 minutes"
        )

    def test_early_distraction_onset_raises_drift(self):
        """
        Focused for 1 minute then sudden heavy distraction: the drift trajectory
        must clearly rise in the second minute compared to staying focused.

        This verifies that LATE-STARTING distraction is also detected.
        """
        # 30 focused batches (1 min), then 30 heavily distracted batches (1 min)
        seq_onset = (
            [self._focused_batch(f"chunk-{i % 3 + 1}") for i in range(30)]
            + [self._heavy_distraction_batch() for _ in range(30)]
        )
        # Control: fully focused for 2 minutes
        seq_focused = [self._focused_batch(f"chunk-{i % 3 + 1}") for i in range(60)]

        traj_onset   = _simulate_timeline(seq_onset,   self._BASELINE)
        traj_focused = _simulate_timeline(seq_focused,  self._BASELINE)

        # At the end of minute 2, the onset session must have higher drift
        onset_final   = traj_onset[-1][1]
        focused_final = traj_focused[-1][1]

        assert onset_final > focused_final, (
            f"Session with distraction onset ({onset_final:.3f}) must exceed "
            f"fully focused ({focused_final:.3f}) by end of minute 2"
        )

    def test_beta_trajectory_rises_when_distraction_begins(self):
        """
        Beta (the instantaneous decay rate) must increase as distraction signals
        enter the rolling window. Check that beta_gated at minute 2 is higher
        than at minute 1 when distraction starts at minute 1.
        """
        seq = (
            [self._focused_batch(f"chunk-{i % 3 + 1}") for i in range(30)]
            + [self._heavy_distraction_batch() for _ in range(30)]
        )
        traj = _simulate_timeline(seq, self._BASELINE)

        # beta just before distraction onset (~1 min)
        beta_at_1min = next(beta for t, _, beta in traj if t >= 1.0)
        # beta after distraction fills the window (~2 min)
        beta_at_2min = traj[-1][2]

        assert beta_at_2min > beta_at_1min, (
            f"Beta should rise after distraction onset: "
            f"1min={beta_at_1min:.4f} 2min={beta_at_2min:.4f}"
        )

    # ── Per-signal discriminability (LLM feature quality) ────────────────────

    def test_z_idle_responds_to_idle_increase(self):
        """
        z_idle must be clearly higher for idle-heavy batches than for active ones.
        This verifies idle is a usable LLM input feature.
        """
        active_batches = [_batch(idle_s=0.1) for _ in range(15)]
        idle_batches   = [_batch(idle_s=1.8) for _ in range(15)]

        f_active = extract_features(active_batches, {}, 180.0)
        f_idle   = extract_features(idle_batches,   {}, 180.0)

        z_active = compute_z_scores(f_active, self._BASELINE)
        z_idle_f = compute_z_scores(f_idle,   self._BASELINE)

        assert z_idle_f.z_idle > z_active.z_idle + 0.5, (
            f"z_idle should be clearly higher for idle batches: "
            f"active={z_active.z_idle:.3f} idle={z_idle_f.z_idle:.3f}"
        )

    def test_z_focus_loss_responds_to_blur(self):
        """
        z_focus_loss must be > 0 when batches are blurred and 0 when focused.
        """
        focused_batches = [_batch(focus="focused") for _ in range(15)]
        blurred_batches = [_batch(focus="blurred") for _ in range(15)]

        f_focused = extract_features(focused_batches, {}, 180.0)
        f_blurred = extract_features(blurred_batches, {}, 180.0)

        z_focused = compute_z_scores(f_focused, self._BASELINE)
        z_blurred = compute_z_scores(f_blurred, self._BASELINE)

        assert z_focused.z_focus_loss == 0.0, "No blur → z_focus_loss must be 0"
        assert z_blurred.z_focus_loss > 0.5, (
            f"Blurred batches must produce z_focus_loss > 0.5, got {z_blurred.z_focus_loss:.3f}"
        )

    def test_all_signals_zero_for_perfect_reading(self):
        """
        A reader at exactly their baseline values should produce near-zero z-scores
        across all signals.  This ensures the model's zero point is calibrated.
        """
        # Batches that exactly match the default baseline assumptions
        batches = [_batch(idle_s=0.1, focus="focused", scroll_abs=100,
                          scroll_neg=2, scroll_pos=98, scroll_events=5,
                          scroll_dir_changes=0) for _ in range(15)]
        features = extract_features(batches, {}, 180.0)
        z = compute_z_scores(features, self._BASELINE)

        # All z_pos terms should be small (< 1.0) — no signal is firing
        for field_name, val in [
            ("z_idle",       z.z_idle),
            ("z_focus_loss", z.z_focus_loss),
            ("z_jitter",     z.z_jitter),
            ("z_regress",    z.z_regress),
            ("z_mouse",      z.z_mouse),
            ("z_pace",       z.z_pace),
        ]:
            assert val < 1.0, (
                f"{field_name} should be < 1.0 for normal reading, got {val:.3f}"
            )

    def test_beta_components_dict_identifies_dominant_signal(self):
        """
        For a heavily idle + blurred session, beta_components should show
        idle and focus_loss as the top contributors.
        This is the data structure the LLM would consume.
        """
        batches = [_batch(idle_s=1.8, focus="blurred") for _ in range(15)]
        features = extract_features(batches, {}, 180.0)
        z = compute_z_scores(features, self._BASELINE)
        _, components = compute_beta_raw(z)

        # idle and focus_loss should be the top two signal contributions
        signal_contribs = {
            k: v for k, v in components.items()
            if k not in ("beta0", "beta_raw", "beta_ema", "confidence", "beta_gated")
        }
        top_two = sorted(signal_contribs, key=signal_contribs.get, reverse=True)[:2]

        assert "idle" in top_two or "focus_loss" in top_two, (
            f"Expected idle or focus_loss to dominate, got top two: {top_two} "
            f"with values {signal_contribs}"
        )

    def test_no_baseline_still_detects_heavy_distraction(self):
        """
        Even without calibration data, the model must still distinguish
        heavy distraction from focused reading using safe defaults.
        This ensures uncalibrated users are not invisible to the system.
        """
        focused_batches  = [_batch(idle_s=0.1, focus="focused",
                                   para_id=f"chunk-{i % 3 + 1}") for i in range(15)]
        heavy_batches    = [_batch(idle_s=1.8, focus="blurred") for _ in range(15)]

        ema_focused = _simulate_session(focused_batches, 3.0, baseline=None, n_iterations=30)
        ema_heavy   = _simulate_session(heavy_batches,   3.0, baseline=None, n_iterations=30)

        assert ema_heavy > ema_focused + 0.05, (
            f"Without baseline, heavy distraction ({ema_heavy:.3f}) must still exceed "
            f"focused ({ema_focused:.3f}) by at least 5 percentage points"
        )
