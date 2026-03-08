"""
Pure feature extraction from a list of telemetry_batch payloads.

All functions are side-effect-free and independently testable.

Key design principles
---------------------
- Pace is only computed when there is sufficient evidence (>=2 paragraphs, >=10 s effective).
- Mouse efficiency z-score is skipped when the user's mouse is stationary (path < threshold).
- Stagnation fires only when a single paragraph dominates >80% of the window.
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

# Minimum effective seconds required to compute pace (prevents explosion from 1 paragraph)
_PACE_MIN_EFF_S: float = 10.0
# Minimum distinct paragraphs to compute pace
_PACE_MIN_PARAGRAPHS: int = 2
# Minimum total mouse path before efficiency is considered meaningful
_MOUSE_PATH_ACTIVE_PX: float = 10.0
# Stagnation only fires when ONE paragraph dominates this much of the window
_STAGNATION_THRESHOLD: float = 0.80


# ── Individual feature helpers (exported for unit tests) ──────────────────────


def scroll_velocity_norm(
    scroll_delta_abs_sum: float, viewport_height_px: float
) -> float:
    """px / (viewport_height * batch_window_s) — dimensionless."""
    vh = viewport_height_px if viewport_height_px > 0 else _DEFAULT_VIEWPORT_H
    return scroll_delta_abs_sum / (vh * _BATCH_WINDOW_S)


def jitter_ratio(direction_changes: int, event_count: int) -> float:
    """direction_changes / max(event_count, 1)."""
    return direction_changes / max(event_count, 1)


def regress_rate(pos_sum: float, neg_sum: float) -> float:
    """neg_sum / (pos_sum + neg_sum + eps)."""
    return neg_sum / (pos_sum + neg_sum + _EPS)


def mouse_efficiency(path_px: float, net_px: float) -> float:
    """
    net_px / max(path_px, eps), clamped [0, 1].

    Returns 1.0 (fully efficient) when path is near zero — the user is not moving
    the mouse, which is normal during reading and carries no negative signal.
    """
    if path_px < _MOUSE_PATH_ACTIVE_PX:
        return 1.0  # no movement → not applicable → default to neutral
    return min(1.0, net_px / max(path_px, _EPS))


def idle_ratio_fn(idle_seconds: float) -> float:
    """idle_seconds / 2.0, clamped [0, 1]."""
    return min(1.0, idle_seconds / _BATCH_WINDOW_S)


# ── Paragraph stagnation ──────────────────────────────────────────────────────


def paragraph_stagnation(batches: list[dict[str, Any]]) -> float:
    """
    Return the max single-paragraph dominance fraction IF it exceeds
    _STAGNATION_THRESHOLD (0.80), otherwise 0.

    A threshold of 0.80 means a single paragraph dominated >24 s of a 30 s window.
    This filters out normal reading where the user naturally dwells on paragraphs.
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
    max_count = max(counts.values())
    frac = max_count / n
    return frac if frac >= _STAGNATION_THRESHOLD else 0.0


# ── Pace estimation ───────────────────────────────────────────────────────────


def estimate_window_wpm(
    batches: list[dict[str, Any]],
    paragraph_word_counts: dict[str, int],
) -> tuple[float, bool, int]:
    """
    Estimate effective WPM from paragraph transitions in the window.

    Returns (wpm, pace_available, n_distinct_paragraphs).

    pace_available is True only when:
    - At least _PACE_MIN_PARAGRAPHS distinct paragraph IDs were observed, AND
    - At least _PACE_MIN_EFF_S effective (focused, low-idle) seconds were counted.

    This prevents pace explosions when the user is sitting on one paragraph for the
    entire window, which is normal reading behaviour for longer text blocks.
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
    pace_available = (n_paragraphs >= _PACE_MIN_PARAGRAPHS and eff_seconds >= _PACE_MIN_EFF_S)

    if not pace_available or eff_seconds < _EPS:
        return 0.0, False, n_paragraphs

    words_read = sum(
        paragraph_word_counts.get(pid, _DEFAULT_WORDS_PER_PARA)
        for pid in seen_ids
    )
    wpm = (words_read / eff_seconds) * 60.0
    return wpm, True, n_paragraphs


# ── Main extractor ────────────────────────────────────────────────────────────


def extract_features(
    batches: list[dict[str, Any]],
    paragraph_word_counts: dict[str, int],
    baseline_wpm_effective: float,
) -> WindowFeatures:
    """
    Build a WindowFeatures dataclass from a list of batch payloads.

    Parameters
    ----------
    batches:
        Ordered list of telemetry_batch payloads (oldest first).
    paragraph_word_counts:
        Mapping of paragraph_id (str) → word count.
    baseline_wpm_effective:
        User's calibration-derived effective WPM (used for pace_ratio).
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

    for b in batches:
        sv_norms.append(scroll_velocity_norm(
            b.get("scroll_delta_abs_sum", 0.0),
            b.get("viewport_height_px", _DEFAULT_VIEWPORT_H),
        ))
        jitters.append(jitter_ratio(
            b.get("scroll_direction_changes", 0),
            b.get("scroll_event_count", 0),
        ))
        regresses.append(regress_rate(
            b.get("scroll_delta_pos_sum", 0.0),
            b.get("scroll_delta_neg_sum", 0.0),
        ))
        idle_ratios_.append(idle_ratio_fn(b.get("idle_seconds", 0.0)))
        pauses.append(min(b.get("scroll_pause_seconds", 0.0), _PAUSE_CAP_S))

        path_px = b.get("mouse_path_px", 0.0)
        net_px = b.get("mouse_net_px", 0.0)
        mouse_effs.append(mouse_efficiency(path_px, net_px))
        mouse_paths.append(path_px)

        focus_losses.append(0.0 if b.get("window_focus_state", "focused") == "focused" else 1.0)

    sv_mean = mean(sv_norms)
    sv_std = stdev(sv_norms) if n > 1 else 0.0

    # ── Pace (gated) ─────────────────────────────────────────────────────────
    base_wpm = baseline_wpm_effective if baseline_wpm_effective > 0 else 200.0
    win_wpm, pace_avail, _ = estimate_window_wpm(batches, paragraph_word_counts)

    if pace_avail and win_wpm > 0:
        pace_ratio = win_wpm / base_wpm
        # Symmetric: |log| so 2x speed == 0.5x speed in deviation
        pace_dev = abs(math.log(max(pace_ratio, 1e-3)))
    else:
        pace_ratio = 1.0
        pace_dev = 0.0

    return WindowFeatures(
        n_batches=n,
        scroll_velocity_norm_mean=sv_mean,
        scroll_velocity_norm_std=sv_std,
        scroll_jitter_mean=mean(jitters),
        regress_rate_mean=mean(regresses),
        window_wpm_effective=win_wpm,
        pace_ratio=pace_ratio,
        pace_dev=pace_dev,
        pace_available=pace_avail,
        idle_ratio_mean=mean(idle_ratios_),
        scroll_pause_mean=mean(pauses),
        focus_loss_rate=mean(focus_losses),
        mouse_efficiency_mean=mean(mouse_effs),
        mouse_path_px_mean=mean(mouse_paths),
        paragraph_stagnation=paragraph_stagnation(batches),
    )
