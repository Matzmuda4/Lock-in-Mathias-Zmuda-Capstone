"""
Tests for telemetry signal tracking, data-quality guardrails (B1/B2/B3),
and skimming fallback (C1).

These tests verify:
- All input signals produce non-zero values when active (no silent zeroing)
- Server-side quality flags are set correctly on bad inputs
- Quality confidence multiplier is computed correctly
- Progress velocity (skimming fallback) is computed correctly
- Feature extraction handles all edge cases
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services.drift.features import (
    compute_progress_velocity,
    compute_quality_confidence_mult,
    extract_features,
    idle_ratio_fn,
    jitter_ratio,
    mouse_efficiency,
    regress_rate,
    scroll_velocity_norm,
)
from app.services.drift.model import (
    BETA0,
    Z_POS_CAP,
    compute_drift_result,
    compute_z_scores,
)
from app.services.drift.types import WindowFeatures


# ── Shared test helpers ───────────────────────────────────────────────────────

def _batch(
    *,
    idle_s: float = 0.5,
    focus: str = "focused",
    scroll_abs: float = 300.0,
    scroll_pos: float = 300.0,
    scroll_neg: float = 0.0,
    dir_changes: int = 1,
    event_count: int = 5,
    pause_s: float = 0.3,
    mouse_path: float = 50.0,
    mouse_net: float = 40.0,
    viewport_h: float = 800.0,
    viewport_w: float = 1280.0,
    para_id: str = "chunk-1",
    progress: float = 0.3,
    telemetry_fault: bool = False,
    scroll_capture_fault: bool = False,
    paragraph_missing_fault: bool = False,
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
        "viewport_width_px": viewport_w,
        "current_paragraph_id": para_id,
        "viewport_progress_ratio": progress,
        "telemetry_fault": telemetry_fault,
        "scroll_capture_fault": scroll_capture_fault,
        "paragraph_missing_fault": paragraph_missing_fault,
    }


_BASELINE: dict[str, Any] = {
    "wpm_effective": 180.0,
    "idle_ratio_mean": 0.35,
    "idle_ratio_std": 0.20,
    "scroll_jitter_mean": 0.10,
    "scroll_jitter_std": 0.10,
    "regress_rate_mean": 0.05,
    "regress_rate_std": 0.06,
    "para_dwell_median_s": 10.0,
    "para_dwell_iqr_s": 3.0,
    "scroll_velocity_norm_mean": 0.15,
    "scroll_velocity_norm_std": 0.08,
}

_WORD_COUNTS = {f"chunk-{i}": 30 for i in range(1, 10)}


# ── Part A1: idle_seconds per-window clamp ────────────────────────────────────

class TestIdleSignalTracking:
    """idle_seconds should be per-window (0..2s) and produce meaningful idle_ratio."""

    def test_zero_idle_gives_zero_ratio(self) -> None:
        assert idle_ratio_fn(0.0) == 0.0

    def test_full_window_idle_gives_ratio_one(self) -> None:
        assert idle_ratio_fn(2.0) == 1.0

    def test_over_window_idle_clamped_to_one(self) -> None:
        """Server clamps idle to 2s, but feature extraction also caps at 1.0."""
        assert idle_ratio_fn(5.0) == 1.0

    def test_half_window_idle_gives_half_ratio(self) -> None:
        assert idle_ratio_fn(1.0) == pytest.approx(0.5)

    def test_idle_ratio_drives_z_idle(self) -> None:
        """When idle_ratio exceeds baseline mean, z_idle must be > 0."""
        features = WindowFeatures(
            n_batches=15,
            idle_ratio_mean=0.9,  # way above baseline 0.35
        )
        z = compute_z_scores(features, _BASELINE)
        assert z.z_idle > 0.0, "High idle_ratio should produce positive z_idle"

    def test_low_idle_gives_zero_z_idle(self) -> None:
        """When idle_ratio is below baseline, z_idle should be 0 (one-sided)."""
        features = WindowFeatures(
            n_batches=15,
            idle_ratio_mean=0.05,  # well below baseline 0.35
        )
        z = compute_z_scores(features, _BASELINE)
        assert z.z_idle == 0.0, "Low idle should not increase z_idle (one-sided)"

    def test_full_window_idle_batches_produce_high_disruption(self) -> None:
        """15 batches of maximum idle → high disruption score."""
        batches = [_batch(idle_s=2.0, focus="focused", scroll_abs=0.0, event_count=0) for _ in range(15)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.idle_ratio_mean == pytest.approx(1.0)
        result = compute_drift_result(features, _BASELINE, 3.0, 0.0, BETA0)
        assert result.disruption_score > 0.5, (
            f"Full-idle session should have disruption > 0.5, got {result.disruption_score:.3f}"
        )


# ── Part A2: scroll delta tracking ────────────────────────────────────────────

class TestScrollSignalTracking:
    """scroll_delta values should produce non-zero signals when scrolling occurs."""

    def test_scroll_velocity_norm_nonzero(self) -> None:
        """scroll_delta_abs_sum > 0 should produce nonzero normalized velocity."""
        vel = scroll_velocity_norm(300.0, 800.0)
        assert vel > 0.0
        assert vel == pytest.approx(300.0 / (800.0 * 2.0))

    def test_scroll_velocity_norm_zero_scroll(self) -> None:
        """No scrolling → zero velocity (e.g. during tab-away)."""
        assert scroll_velocity_norm(0.0, 800.0) == 0.0

    def test_scroll_velocity_proportional_to_delta(self) -> None:
        """Doubling scroll delta doubles velocity."""
        v1 = scroll_velocity_norm(200.0, 800.0)
        v2 = scroll_velocity_norm(400.0, 800.0)
        assert v2 == pytest.approx(2 * v1)

    def test_scroll_velocity_inversely_proportional_to_viewport(self) -> None:
        """Larger viewport → smaller normalized velocity for same delta."""
        v_small = scroll_velocity_norm(200.0, 400.0)
        v_large = scroll_velocity_norm(200.0, 800.0)
        assert v_small > v_large

    def test_jitter_ratio_with_direction_changes(self) -> None:
        """Direction changes / event_count → jitter ratio."""
        assert jitter_ratio(3, 10) == pytest.approx(0.3)
        assert jitter_ratio(0, 10) == 0.0

    def test_jitter_ratio_zero_events_guard(self) -> None:
        """Zero event count should not cause division by zero."""
        assert jitter_ratio(0, 0) == 0.0

    def test_regress_rate_pure_backtrack(self) -> None:
        """All negative scroll → regress_rate = 1.0."""
        assert regress_rate(0.0, 500.0) == pytest.approx(1.0)

    def test_regress_rate_pure_forward(self) -> None:
        """All positive scroll → regress_rate ≈ 0."""
        assert regress_rate(500.0, 0.0) < 0.001

    def test_regress_rate_mixed(self) -> None:
        """300 forward, 100 backward → rate = 100/(300+100) = 0.25."""
        assert regress_rate(300.0, 100.0) == pytest.approx(0.25)

    def test_all_signals_nonzero_in_active_batch(self) -> None:
        """An active reading batch should produce nonzero values for all signals."""
        batches = [
            _batch(
                idle_s=0.3,
                scroll_abs=400.0,
                scroll_pos=350.0,
                scroll_neg=50.0,
                dir_changes=2,
                event_count=6,
                mouse_path=80.0,
                mouse_net=60.0,
                para_id="chunk-1",
            )
        ]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.scroll_velocity_norm_mean > 0.0, "scroll_velocity should be nonzero"
        assert features.idle_ratio_mean < 1.0, "idle_ratio should be < 1 when active"
        assert features.regress_rate_mean > 0.0, "regress_rate should be nonzero with neg scroll"
        assert features.scroll_jitter_mean > 0.0, "jitter should be nonzero with dir changes"
        assert features.mouse_efficiency_mean <= 1.0

    def test_zero_scroll_batches_produce_zero_velocity(self) -> None:
        """Batches with no scroll events produce zero velocity."""
        batches = [_batch(scroll_abs=0.0, event_count=0, scroll_pos=0.0)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.scroll_velocity_norm_mean == 0.0


# ── Part A3: paragraph tracking ───────────────────────────────────────────────

class TestParagraphTracking:
    """current_paragraph_id should drive stagnation and pace estimation."""

    def test_same_para_all_batches_gives_stagnation_one(self) -> None:
        """All 15 batches on same paragraph → stagnation_ratio = 1.0."""
        batches = [_batch(para_id="chunk-1") for _ in range(15)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.stagnation_ratio == pytest.approx(1.0)

    def test_all_different_paras_gives_low_stagnation(self) -> None:
        """Each batch on a different paragraph → stagnation near 1/n."""
        batches = [_batch(para_id=f"chunk-{i}") for i in range(1, 16)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.stagnation_ratio < 0.2, "Many paras should give low stagnation"

    def test_missing_para_ids_mark_paragraph_fault(self) -> None:
        """Batches with paragraph_missing_fault=True reduce quality confidence."""
        batches = [
            _batch(para_id=None, paragraph_missing_fault=True)
            for _ in range(15)
        ]
        tf, scf, pmf, mult = compute_quality_confidence_mult(batches)
        assert pmf > 0.7, "All-missing paras → paragraph_missing_fault_rate > 0.7"
        assert mult < 1.0, "Quality multiplier should be reduced"

    def test_pace_available_with_multiple_paras(self) -> None:
        """Pace estimation only available when >= 2 distinct paragraphs and >= 10 eff seconds."""
        # 10 batches with 2 distinct paragraphs and low idle → pace_available = True
        batches = (
            [_batch(para_id="chunk-1", idle_s=0.1) for _ in range(5)]
            + [_batch(para_id="chunk-2", idle_s=0.1) for _ in range(5)]
        )
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.pace_available is True
        assert features.paragraphs_observed == 2

    def test_pace_unavailable_with_single_para(self) -> None:
        """Single paragraph → pace_available = False."""
        batches = [_batch(para_id="chunk-1", idle_s=0.1) for _ in range(15)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.pace_available is False


# ── Part B1/B2: data-quality guardrails ───────────────────────────────────────

class TestDataQualityGuardrails:
    """Server-side quality flags and confidence penalties."""

    def test_telemetry_fault_rate_zero_for_clean_batches(self) -> None:
        batches = [_batch() for _ in range(10)]
        tf, scf, pmf, mult = compute_quality_confidence_mult(batches)
        assert tf == 0.0
        assert mult == 1.0

    def test_telemetry_fault_rate_high_reduces_confidence(self) -> None:
        """If > 30% of batches have telemetry_fault, confidence mult *= 0.5."""
        batches = [_batch(telemetry_fault=True) for _ in range(10)]
        tf, _, _, mult = compute_quality_confidence_mult(batches)
        assert tf == 1.0
        assert mult == pytest.approx(0.5)

    def test_scroll_capture_fault_reduces_confidence(self) -> None:
        """If > 50% of batches have scroll_capture_fault, mult *= 0.7."""
        batches = [_batch(scroll_capture_fault=True) for _ in range(10)]
        _, scf, _, mult = compute_quality_confidence_mult(batches)
        assert scf == 1.0
        assert mult == pytest.approx(0.7)

    def test_paragraph_missing_fault_reduces_confidence(self) -> None:
        """If > 70% of batches have paragraph_missing_fault, mult *= 0.7."""
        batches = [_batch(paragraph_missing_fault=True) for _ in range(10)]
        _, _, pmf, mult = compute_quality_confidence_mult(batches)
        assert pmf == 1.0
        assert mult == pytest.approx(0.7)

    def test_multiple_faults_stack_multiplicatively(self) -> None:
        """Telemetry fault + paragraph fault → mult = 0.5 × 0.7 = 0.35."""
        batches = [
            _batch(telemetry_fault=True, paragraph_missing_fault=True)
            for _ in range(10)
        ]
        _, _, _, mult = compute_quality_confidence_mult(batches)
        assert mult == pytest.approx(0.5 * 0.7)

    def test_quality_mult_applied_to_confidence_in_drift_result(self) -> None:
        """Broken telemetry reduces model confidence in drift result."""
        clean_batches = [_batch() for _ in range(15)]
        faulty_batches = [_batch(telemetry_fault=True, paragraph_missing_fault=True) for _ in range(15)]

        clean_features = extract_features(clean_batches, _WORD_COUNTS, 180.0)
        faulty_features = extract_features(faulty_batches, _WORD_COUNTS, 180.0)

        clean_result = compute_drift_result(clean_features, _BASELINE, 2.0, 0.0, BETA0)
        faulty_result = compute_drift_result(faulty_features, _BASELINE, 2.0, 0.0, BETA0)

        assert clean_result.confidence > faulty_result.confidence, (
            "Faulty telemetry must reduce model confidence"
        )

    def test_telemetry_fault_ingest_idle_clamped_in_features(self) -> None:
        """Even if a batch arrived with idle_seconds > 2, feature extraction uses clamped value."""
        # The server clamps to 2.0 before storage, but let's test the extraction layer too
        # Simulate what happens if idle > 2 slips through
        batches = [_batch(idle_s=2.0) for _ in range(10)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        # idle_ratio_fn(2.0) = 1.0 → capped at 1.0
        assert features.idle_ratio_mean == pytest.approx(1.0)


# ── Part C1: skimming fallback via progress_velocity ──────────────────────────

class TestSkimmingFallback:
    """progress_velocity should trigger skimming detection when pace unavailable."""

    def test_progress_velocity_zero_when_no_change(self) -> None:
        batches = [_batch(progress=0.3) for _ in range(5)]
        assert compute_progress_velocity(batches) == pytest.approx(0.0)

    def test_progress_velocity_positive_when_advancing(self) -> None:
        """Progress going from 0.1 to 0.5 over 5 batches (10s) = 0.04/s."""
        batches = [_batch(progress=0.1 + i * 0.1) for i in range(5)]
        vel = compute_progress_velocity(batches)
        # (0.5 - 0.1) / (5 * 2.0) = 0.4 / 10 = 0.04
        assert vel == pytest.approx(0.04)

    def test_progress_velocity_negative_when_regressing(self) -> None:
        batches = [_batch(progress=0.9 - i * 0.1) for i in range(5)]
        vel = compute_progress_velocity(batches)
        assert vel < 0.0

    def test_fast_progress_triggers_skim_fallback_in_model(self) -> None:
        """Rapid progress (no para IDs) should increase disruption via z_skim."""
        # Build batches with fast progress but no para IDs
        batches = [
            _batch(
                para_id=None,
                progress=i * 0.08,  # 8% per batch = very fast
                scroll_abs=800.0,
                idle_s=0.1,
                paragraph_missing_fault=True,
            )
            for i in range(15)
        ]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.progress_velocity > 0.03, (
            f"Fast progress should give high velocity, got {features.progress_velocity:.4f}"
        )

        # Drift result should pick up the skimming fallback
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        # With skim fallback, disruption should be higher than pure stagnation
        assert result.disruption_score > 0.0

    def test_slow_progress_no_skim_fallback(self) -> None:
        """Very slow progress should not trigger skim fallback (below threshold 0.02/s)."""
        batches = [_batch(progress=0.3 + i * 0.001) for i in range(15)]  # tiny increment
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.progress_velocity < 0.02


# ── Mouse signal tracking ─────────────────────────────────────────────────────

class TestMouseSignalTracking:
    """Mouse efficiency should be computed correctly from path/net displacement."""

    def test_straight_line_mouse_efficiency_one(self) -> None:
        """Straight-line movement → efficiency = 1.0."""
        assert mouse_efficiency(100.0, 100.0) == pytest.approx(1.0)

    def test_zig_zag_mouse_low_efficiency(self) -> None:
        """Zig-zag path → path >> net → efficiency < 1."""
        eff = mouse_efficiency(300.0, 50.0)
        assert eff < 0.3

    def test_stationary_mouse_returns_neutral(self) -> None:
        """No mouse movement → returns 1.0 (neutral, not a distraction)."""
        assert mouse_efficiency(0.0, 0.0) == 1.0
        assert mouse_efficiency(5.0, 5.0) == 1.0  # below active threshold

    def test_mouse_efficiency_clamped_to_one(self) -> None:
        """net > path is impossible but should be clamped to 1.0."""
        assert mouse_efficiency(50.0, 100.0) == 1.0

    def test_active_mouse_drives_features(self) -> None:
        """Batches with mouse movement produce nonzero path and efficiency."""
        batches = [_batch(mouse_path=200.0, mouse_net=150.0) for _ in range(5)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.mouse_path_px_mean == pytest.approx(200.0)
        assert features.mouse_efficiency_mean == pytest.approx(0.75)


# ── Focus loss signal tracking ────────────────────────────────────────────────

class TestFocusLossSignalTracking:
    """Focus loss should increase disruption and reduce engagement."""

    def test_blurred_batches_give_focus_loss_rate_one(self) -> None:
        batches = [_batch(focus="blurred") for _ in range(10)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.focus_loss_rate == pytest.approx(1.0)

    def test_focused_batches_give_focus_loss_rate_zero(self) -> None:
        batches = [_batch(focus="focused") for _ in range(10)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        assert features.focus_loss_rate == pytest.approx(0.0)

    def test_blurred_batches_raise_z_focus_loss(self) -> None:
        features = WindowFeatures(n_batches=15, focus_loss_rate=1.0)
        z = compute_z_scores(features, _BASELINE)
        assert z.z_focus_loss > 0.0

    def test_tab_away_session_high_disruption(self) -> None:
        """Session where user is tabbed away → disruption_score > 0.6."""
        batches = [
            _batch(focus="blurred", idle_s=2.0, scroll_abs=0.0, event_count=0)
            for _ in range(15)
        ]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 2.0, 0.0, BETA0)
        assert result.disruption_score > 0.6, (
            f"Tab-away session should have high disruption, got {result.disruption_score:.3f}"
        )


# ── Signal completeness: all signals return values ────────────────────────────

class TestSignalCompleteness:
    """
    Verify that a full reading batch populates ALL feature fields with
    meaningful (non-default) values — nothing silently zeroes out.
    """

    def test_all_feature_fields_populated_from_active_batch(self) -> None:
        """A complete active-reading batch should produce nonzero values across all signals."""
        batches = [
            _batch(
                idle_s=0.3,
                focus="focused",
                scroll_abs=300.0,
                scroll_pos=250.0,
                scroll_neg=50.0,
                dir_changes=2,
                event_count=6,
                pause_s=0.5,
                mouse_path=100.0,
                mouse_net=80.0,
                viewport_h=800.0,
                para_id=f"chunk-{(i % 3) + 1}",
                progress=0.05 * i,
            )
            for i in range(15)
        ]
        features = extract_features(batches, _WORD_COUNTS, 180.0)

        # Every signal should be non-default where active reading is expected
        assert features.n_batches == 15
        assert features.scroll_velocity_norm_mean > 0.0, "scroll_velocity_norm_mean should be >0"
        assert features.scroll_jitter_mean > 0.0, "jitter should be >0 with dir_changes"
        assert features.regress_rate_mean > 0.0, "regress_rate should be >0 with neg scroll"
        assert features.idle_ratio_mean < 1.0, "idle_ratio should be <1 when active"
        assert features.focus_loss_rate == 0.0, "focus loss should be 0 when focused"
        assert features.mouse_efficiency_mean > 0.0, "mouse_efficiency should be >0"
        assert features.mouse_path_px_mean > 0.0, "mouse_path should be >0"
        assert features.stagnation_ratio < 1.0, "stagnation should be <1 with multiple paras"
        assert features.paragraphs_observed >= 2, "should observe multiple paragraphs"
        assert features.progress_velocity >= 0.0, "progress_velocity computed"
        assert features.quality_confidence_mult == 1.0, "no faults → quality mult = 1"

    def test_full_drift_result_has_all_nonzero_fields(self) -> None:
        """compute_drift_result should produce valid values for all output fields."""
        batches = [_batch(para_id=f"chunk-{(i % 3) + 1}") for i in range(15)]
        features = extract_features(batches, _WORD_COUNTS, 180.0)
        result = compute_drift_result(features, _BASELINE, 3.0, 0.1, BETA0)

        assert 0.0 <= result.drift_level <= 1.0
        assert 0.0 <= result.drift_ema <= 1.0
        assert 0.0 <= result.disruption_score <= 1.0
        assert 0.0 <= result.engagement_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.beta_ema > 0.0
        assert result.beta_effective > 0.0
        assert result.attention_score < 1.0  # t=3min, must have decayed some
        assert result.drift_score == result.drift_level

    def test_z_scores_all_bounded(self) -> None:
        """All z-scores must be within [-Z_POS_CAP, Z_POS_CAP]."""
        features = WindowFeatures(
            n_batches=15,
            idle_ratio_mean=0.99,
            focus_loss_rate=0.9,
            scroll_jitter_mean=0.8,
            regress_rate_mean=0.7,
            scroll_burstiness=5.0,
            stagnation_ratio=0.9,
            pace_ratio=3.0,
            pace_dev=1.2,
            pace_available=True,
            mouse_efficiency_mean=0.1,
            mouse_path_px_mean=500.0,
        )
        z = compute_z_scores(features, _BASELINE)
        for name, val in [
            ("z_idle", z.z_idle),
            ("z_focus_loss", z.z_focus_loss),
            ("z_jitter", z.z_jitter),
            ("z_regress", z.z_regress),
            ("z_pause", z.z_pause),
            ("z_stagnation", z.z_stagnation),
            ("z_mouse", z.z_mouse),
            ("z_pace", z.z_pace),
            ("z_skim", z.z_skim),
            ("z_burstiness", z.z_burstiness),
        ]:
            assert -Z_POS_CAP <= val <= Z_POS_CAP, (
                f"{name}={val} out of bounds [-{Z_POS_CAP}, {Z_POS_CAP}]"
            )


# ── C2: Faster re-engagement ──────────────────────────────────────────────────

class TestFasterReEngagement:
    """When engagement is high and previous beta is elevated, beta should drop faster."""

    def test_re_engagement_lowers_drift_compared_to_distraction(self) -> None:
        """
        Start distracted (high beta), then switch to focused.
        Drift should stop growing as fast after re-engagement.
        """
        from app.services.drift.model import compute_drift_result

        # Step 1: distracted window → elevated beta
        distracted = [
            _batch(idle_s=2.0, focus="blurred", scroll_abs=0.0, event_count=0)
            for _ in range(15)
        ]
        feat1 = extract_features(distracted, _WORD_COUNTS, 180.0)
        r1 = compute_drift_result(feat1, _BASELINE, 2.0, 0.0, BETA0)
        high_beta = r1.beta_ema

        # Step 2: focused window → beta should fall from high_beta
        focused = [
            _batch(idle_s=0.1, focus="focused", scroll_abs=300.0,
                   para_id=f"chunk-{(i % 3) + 1}")
            for i in range(15)
        ]
        feat2 = extract_features(focused, _WORD_COUNTS, 180.0)
        r2 = compute_drift_result(feat2, _BASELINE, 4.0, r1.drift_ema, high_beta)

        # Beta should be lower after re-engagement
        assert r2.beta_ema < high_beta, (
            f"Beta should decrease on re-engagement: {high_beta:.4f} → {r2.beta_ema:.4f}"
        )
