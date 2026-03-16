"""
Pure feature extraction from a list of telemetry_batch payloads.

All functions are side-effect-free and independently testable.
"""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any

from app.services.drift.types import WindowFeatures

_EPS: float = 1e-9
_BATCH_WINDOW_S: float = 2.0
_DEFAULT_VIEWPORT_H: float = 800.0
_DEFAULT_WORDS_PER_PARA: int = 20
_PAUSE_CAP_S: float = 10.0
_EFFECTIVE_IDLE_THRESHOLD_S: float = 1.5

# Minimum evidence required before pace is computed
_PACE_MIN_EFF_S: float = 10.0
_PACE_MIN_PARAGRAPHS: int = 2

# Mouse path below this threshold = stationary = neutral (not penalised)
_MOUSE_PATH_ACTIVE_PX: float = 10.0


# ── Individual helpers (exported for unit tests) ──────────────────────────────


def scroll_velocity_norm(scroll_delta_abs_sum: float, viewport_height_px: float) -> float:
    """Viewport-normalised scroll velocity per batch: px / (vh * 2s)."""
    vh = viewport_height_px if viewport_height_px > 0 else _DEFAULT_VIEWPORT_H
    return scroll_delta_abs_sum / (vh * _BATCH_WINDOW_S)


def jitter_ratio(direction_changes: int, event_count: int) -> float:
    return direction_changes / max(event_count, 1)


def regress_rate(pos_sum: float, neg_sum: float) -> float:
    return neg_sum / (pos_sum + neg_sum + _EPS)


def mouse_efficiency(path_px: float, net_px: float) -> float:
    """
    net_px / path_px clamped [0,1].
    Returns 1.0 (neutral) when mouse is stationary — normal during reading.
    """
    if path_px < _MOUSE_PATH_ACTIVE_PX:
        return 1.0
    return min(1.0, net_px / max(path_px, _EPS))


def idle_ratio_fn(idle_seconds: float) -> float:
    return min(1.0, idle_seconds / _BATCH_WINDOW_S)


def compute_stagnation_ratio(batches: list[dict[str, Any]]) -> float:
    """
    Raw fraction of the window dominated by the single most common paragraph_id.
    Returns a value in [0, 1] — z-scoring in model.py decides if it's anomalous.
    """
    n = len(batches)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for b in batches:
        pid = b.get("current_paragraph_id")
        if pid:
            counts[pid] = counts.get(pid, 0) + 1
    if not counts:
        return 0.0
    return max(counts.values()) / n


# Keep old name as alias for backward compat with test_drift.py
def paragraph_stagnation(batches: list[dict[str, Any]]) -> float:
    return compute_stagnation_ratio(batches)


# ── Pace estimation ───────────────────────────────────────────────────────────


def estimate_window_wpm(
    batches: list[dict[str, Any]],
    paragraph_word_counts: dict[str, int],
) -> tuple[float, bool, int]:
    """
    Estimate effective WPM from paragraph transitions in the window.

    Returns (wpm, pace_available, n_distinct_paragraphs).

    pace_available is True only when:
    - >= _PACE_MIN_PARAGRAPHS distinct paragraph IDs observed, AND
    - >= _PACE_MIN_EFF_S effective (focused + low-idle) seconds.
    """
    seen_ids: set[str] = {
        b["current_paragraph_id"]
        for b in batches
        if b.get("current_paragraph_id")
    }
    eff_batches = [
        b for b in batches
        if b.get("window_focus_state", "focused") == "focused"
        and b.get("idle_seconds", 0.0) <= _EFFECTIVE_IDLE_THRESHOLD_S
    ]
    eff_seconds = len(eff_batches) * _BATCH_WINDOW_S
    n_paragraphs = len(seen_ids)
    pace_available = n_paragraphs >= _PACE_MIN_PARAGRAPHS and eff_seconds >= _PACE_MIN_EFF_S

    if not pace_available or eff_seconds < _EPS:
        return 0.0, False, n_paragraphs

    words_read = sum(
        paragraph_word_counts.get(pid, _DEFAULT_WORDS_PER_PARA) for pid in seen_ids
    )
    return (words_read / eff_seconds) * 60.0, True, n_paragraphs


# ── Quality flag helpers ──────────────────────────────────────────────────────


def compute_quality_confidence_mult(batches: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """
    Compute data-quality rates and a combined confidence multiplier.

    Returns:
        (telemetry_fault_rate, scroll_capture_fault_rate,
         paragraph_missing_fault_rate, quality_confidence_mult)

    Confidence penalties (multiplicative):
        - telemetry_fault (idle > 2s)       → ×0.5
        - scroll_capture_fault              → ×0.7
        - paragraph_missing for all batches → ×0.7
    """
    n = len(batches)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0

    tf = sum(1 for b in batches if b.get("telemetry_fault", False)) / n
    scf = sum(1 for b in batches if b.get("scroll_capture_fault", False)) / n
    pmf = sum(1 for b in batches if b.get("paragraph_missing_fault", b.get("current_paragraph_id") is None)) / n

    mult = 1.0
    if tf > 0.3:   # more than 30% of batches had idle > 2s
        mult *= 0.5
    if scf > 0.5:  # more than 50% of batches had no scroll events
        mult *= 0.7
    if pmf > 0.7:  # more than 70% of batches missing paragraph id
        mult *= 0.7

    return tf, scf, pmf, mult


def compute_progress_velocity(batches: list[dict[str, Any]]) -> float:
    """
    Rate of progress_ratio change per second across the window.
    Used as a skimming fallback when pace estimation is unavailable.

    Returns progress delta / window_seconds (0 if insufficient data).
    """
    progresses = [b.get("viewport_progress_ratio", 0.0) for b in batches]
    if len(progresses) < 2:
        return 0.0
    # Change from first to last batch in window
    total_change = progresses[-1] - progresses[0]
    window_seconds = len(progresses) * _BATCH_WINDOW_S
    return total_change / max(window_seconds, _EPS)


# ── Main extractor ────────────────────────────────────────────────────────────


def extract_features(
    batches: list[dict[str, Any]],
    paragraph_word_counts: dict[str, int],
    baseline_wpm_effective: float,
    pause_threshold_s: float = 2.0,
) -> WindowFeatures:
    """
    Build a WindowFeatures dataclass from a list of batch payloads.

    pause_threshold_s is the personalized threshold for "long pause" detection;
    callers should pass `max(2.0, baseline_para_dwell_median_s / 3)`.

    Also computes data-quality flags and confidence multiplier from
    server-side telemetry fault tags.
    """
    n = len(batches)
    if n == 0:
        return WindowFeatures()

    sv_norms: list[float] = []
    jitters: list[float] = []
    regresses: list[float] = []
    idle_ratios_: list[float] = []
    pauses: list[float] = []
    mouse_effs: list[float] = []
    mouse_paths: list[float] = []
    focus_losses: list[float] = []
    long_pauses: list[float] = []

    for b in batches:
        sv = scroll_velocity_norm(
            b.get("scroll_delta_abs_sum", 0.0),
            b.get("viewport_height_px", _DEFAULT_VIEWPORT_H),
        )
        sv_norms.append(sv)
        jitters.append(jitter_ratio(
            b.get("scroll_direction_changes", 0),
            b.get("scroll_event_count", 0),
        ))
        regresses.append(regress_rate(
            b.get("scroll_delta_pos_sum", 0.0),
            b.get("scroll_delta_neg_sum", 0.0),
        ))
        idle_ratios_.append(idle_ratio_fn(b.get("idle_seconds", 0.0)))
        pause_s = min(b.get("scroll_pause_seconds", 0.0), _PAUSE_CAP_S)
        pauses.append(pause_s)
        long_pauses.append(1.0 if pause_s >= pause_threshold_s else 0.0)

        path_px = b.get("mouse_path_px", 0.0)
        net_px = b.get("mouse_net_px", 0.0)
        mouse_effs.append(mouse_efficiency(path_px, net_px))
        mouse_paths.append(path_px)

        focus_losses.append(
            0.0 if b.get("window_focus_state", "focused") == "focused" else 1.0
        )

    sv_mean = mean(sv_norms)
    sv_std = stdev(sv_norms) if n > 1 else 0.0
    # Burstiness: coefficient of variation of scroll velocity
    burstiness = sv_std / max(sv_mean, _EPS) if sv_mean > 1e-6 else 0.0

    # Pace (gated)
    base_wpm = baseline_wpm_effective if baseline_wpm_effective > 0 else 200.0
    win_wpm, pace_avail, n_paras = estimate_window_wpm(batches, paragraph_word_counts)

    if pace_avail and win_wpm > 0:
        pace_ratio = win_wpm / base_wpm
        pace_dev = abs(math.log(max(pace_ratio, 1e-3)))
    else:
        pace_ratio, pace_dev = 1.0, 0.0

    stag = compute_stagnation_ratio(batches)

    # Progress velocity (skimming fallback)
    prog_vel = compute_progress_velocity(batches)

    # Data-quality flags and confidence multiplier
    tf_rate, scf_rate, pmf_rate, qual_mult = compute_quality_confidence_mult(batches)

    return WindowFeatures(
        n_batches=n,
        scroll_velocity_norm_mean=sv_mean,
        scroll_velocity_norm_std=sv_std,
        scroll_burstiness=min(burstiness, 5.0),
        scroll_jitter_mean=mean(jitters),
        regress_rate_mean=mean(regresses),
        window_wpm_effective=win_wpm,
        pace_ratio=pace_ratio,
        pace_dev=pace_dev,
        pace_available=pace_avail,
        paragraphs_observed=n_paras,
        idle_ratio_mean=mean(idle_ratios_),
        scroll_pause_mean=mean(pauses),
        long_pause_share=mean(long_pauses),
        focus_loss_rate=mean(focus_losses),
        mouse_efficiency_mean=mean(mouse_effs),
        mouse_path_px_mean=mean(mouse_paths),
        stagnation_ratio=stag,
        paragraph_stagnation=stag,  # alias
        progress_velocity=prog_vel,
        telemetry_fault_rate=tf_rate,
        scroll_capture_fault_rate=scf_rate,
        paragraph_missing_fault_rate=pmf_rate,
        quality_confidence_mult=qual_mult,
    )
