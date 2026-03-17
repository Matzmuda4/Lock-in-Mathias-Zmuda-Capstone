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


_REGRESSION_ABS_MIN_PX: float = 50.0   # minimum absolute scroll to count as regression event
_REGRESSION_FRAC_THRESH: float = 0.50  # neg_sum > 50% of abs_sum → major regression


def estimate_window_wpm(
    batches: list[dict[str, Any]],
    paragraph_word_counts: dict[str, int],
) -> tuple[float, bool, int]:
    """
    Estimate effective WPM from paragraph transitions in the window.

    Returns (wpm, pace_available, n_distinct_paragraphs).

    pace_available is True only when:
    - >= _PACE_MIN_PARAGRAPHS distinct paragraph IDs observed, AND
    - >= _PACE_MIN_EFF_S effective (focused + low-idle) seconds, AND
    - The window contains NO major regression event.

    Regression gate: if any single batch has scroll_neg > 50% of scroll_abs
    AND scroll_abs > 50 px, the user is navigating (re-read + re-advance), not
    reading linearly.  All paragraph IDs in the window are then untrustworthy
    as a WPM proxy — the same paragraphs appear both during backtrack and
    re-advance, inflating the estimate by 3–6×.
    """
    # Regression gate: check every batch for a major backward scroll event
    for b in batches:
        abs_sum = b.get("scroll_delta_abs_sum", 0.0)
        neg_sum = b.get("scroll_delta_neg_sum", 0.0)
        n_paragraphs_early = len({
            bb["current_paragraph_id"]
            for bb in batches
            if bb.get("current_paragraph_id")
        })
        if abs_sum >= _REGRESSION_ABS_MIN_PX and neg_sum >= _REGRESSION_FRAC_THRESH * abs_sum:
            # Navigation event in window — pace estimate is unreliable
            return 0.0, False, n_paragraphs_early

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


def is_at_end_of_document(batches: list[dict[str, Any]], threshold: float = 0.97) -> bool:
    """
    Returns True when the majority of the window is at the end of the document.

    When the user has finished reading and is sitting at progress >= 0.97,
    stagnation and scroll-absence are EXPECTED, not signs of distraction.
    We suppress those signals to avoid penalising users for finishing.
    """
    if not batches:
        return False
    at_end = sum(
        1 for b in batches if b.get("viewport_progress_ratio", 0.0) >= threshold
    )
    return at_end / len(batches) >= 0.60   # 60%+ of window at near-end


def compute_scroll_capture_fault_rate(batches: list[dict[str, Any]]) -> float:
    """
    Detect genuine scroll-capture failures using consecutive batch comparisons.

    A fault is: viewport_progress_ratio changed between two consecutive batches,
    but no scroll events were recorded in the second batch.  This indicates the
    scroll listener missed events — data is unreliable for that transition.

    Reading pauses (no scroll + no progress change) are NOT faults.
    """
    n = len(batches)
    if n < 2:
        return 0.0
    faults = 0
    for i in range(1, n):
        prev_prog = batches[i - 1].get("viewport_progress_ratio", 0.0)
        curr_prog = batches[i].get("viewport_progress_ratio", 0.0)
        scroll_abs = batches[i].get("scroll_delta_abs_sum", 0.0)
        scroll_ev = batches[i].get("scroll_event_count", 0)
        if scroll_abs < 0.1 and scroll_ev == 0 and abs(curr_prog - prev_prog) > 0.005:
            faults += 1
    return faults / (n - 1)


def compute_quality_confidence_mult(batches: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """
    Compute data-quality rates and a combined confidence multiplier.

    Returns:
        (telemetry_fault_rate, scroll_capture_fault_rate,
         paragraph_missing_fault_rate, quality_confidence_mult)

    Confidence penalties (multiplicative):
        - telemetry_fault (idle > 2s)       → ×0.5
        - scroll_capture_fault (genuine)    → ×0.7  [window-context comparison]
        - paragraph_missing for all batches → ×0.7

    scroll_capture_fault is now computed from consecutive progress comparisons
    rather than the stored per-batch flag, which was a false positive every time
    the user simply paused to read.
    """
    n = len(batches)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0

    tf = sum(1 for b in batches if b.get("telemetry_fault", False)) / n
    scf = compute_scroll_capture_fault_rate(batches)
    pmf = sum(1 for b in batches if b.get("paragraph_missing_fault", b.get("current_paragraph_id") is None)) / n

    mult = 1.0
    if tf > 0.3:
        mult *= 0.5
    if scf > 0.5:
        mult *= 0.7
    if pmf > 0.7:
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
    # Use 250 WPM as the population-average fallback when no calibration baseline
    # exists.  200 WPM was too low — anyone reading at an average pace (~250 WPM)
    # would score pace_ratio ≈ 1.25, just barely below SKIM_THRESHOLD, yet a
    # small burst would push them over and fire a false skim signal.
    base_wpm = baseline_wpm_effective if baseline_wpm_effective > 0 else 250.0
    win_wpm, pace_avail, n_paras = estimate_window_wpm(batches, paragraph_word_counts)

    if pace_avail and win_wpm > 0:
        pace_ratio = win_wpm / base_wpm
        pace_dev = abs(math.log(max(pace_ratio, 1e-3)))
    else:
        pace_ratio, pace_dev = 1.0, 0.0

    # End-of-document: user has finished reading; stagnation/scroll-absence
    # are expected here, not signs of distraction.
    at_end = is_at_end_of_document(batches)

    # Suppress stagnation when at end of document — we cannot tell "stuck"
    # from "done" without this, and the false signal is severe.
    stag = compute_stagnation_ratio(batches)
    if at_end:
        stag = 0.0

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
        at_end_of_document=at_end,
    )
