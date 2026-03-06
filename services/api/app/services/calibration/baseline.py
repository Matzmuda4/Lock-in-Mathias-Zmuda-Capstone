"""
Baseline computation from telemetry_batch activity events.

Takes the raw list of telemetry_batch payload dicts (one per 2-second window)
collected during a calibration session and produces a stable reading profile.

All values are exported for unit testing.
"""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any

_BATCH_WINDOW_S = 2.0


# ─── Pure helpers (exported for unit tests) ──────────────────────────────────

def scroll_velocities(batches: list[dict[str, Any]]) -> list[float]:
    """px/s for each batch window."""
    return [b.get("scroll_delta_abs_sum", 0.0) / _BATCH_WINDOW_S for b in batches]


def scroll_jitter_values(batches: list[dict[str, Any]]) -> list[float]:
    """Direction-change ratio per batch: changes / events (0 when no events)."""
    vals = []
    for b in batches:
        events = b.get("scroll_event_count", 0)
        changes = b.get("scroll_direction_changes", 0)
        if events > 0:
            vals.append(changes / events)
    return vals


def idle_ratios(batches: list[dict[str, Any]]) -> list[float]:
    """idle_seconds / window_seconds, clamped to [0, 1]."""
    return [min(b.get("idle_seconds", 0.0) / _BATCH_WINDOW_S, 1.0) for b in batches]


def paragraph_dwells(batches: list[dict[str, Any]]) -> dict[str, int]:
    """
    Map {paragraph_id → batch count} — how many 2-second windows each
    paragraph was the most-visible element.
    """
    counts: dict[str, int] = {}
    for b in batches:
        pid = b.get("current_paragraph_id")
        if pid:
            counts[pid] = counts.get(pid, 0) + 1
    return counts


def estimate_wpm(
    batches: list[dict[str, Any]],
    chunk_word_counts: dict[int, int],
    duration_seconds: int,
    total_words_override: int | None = None,
) -> float:
    """
    WPM estimate.

    When ``total_words_override`` is provided (calibration case where all text
    was read), WPM = total_words / duration_minutes.

    Otherwise falls back to paragraph-dwell tracking (normal sessions), parsing
    ``chunk-N`` or ``calib-N`` paragraph IDs to look up word counts.
    """
    duration_min = max(duration_seconds / 60.0, 0.1)

    if total_words_override is not None and total_words_override > 0:
        return total_words_override / duration_min

    # Dwell-based estimate: accumulate words from visited paragraphs
    seen_ids = set(paragraph_dwells(batches).keys())
    total_words = 0
    for pid in seen_ids:
        # Accept "chunk-N" and "calib-N" formats
        for prefix in ("chunk-", "calib-"):
            if pid.startswith(prefix):
                try:
                    chunk_id = int(pid[len(prefix):])
                    total_words += chunk_word_counts.get(chunk_id, 0)
                except (ValueError, AttributeError):
                    pass
                break
    return total_words / duration_min


# ─── Public entry point ───────────────────────────────────────────────────────

def compute_baseline(
    batches: list[dict[str, Any]],
    chunk_word_counts: dict[int, int],
    duration_seconds: int,
    total_words: int | None = None,
) -> dict[str, Any]:
    """
    Compute the full reading baseline from a list of telemetry_batch payloads.

    Parameters
    ----------
    batches:
        Ordered list of payload dicts from activity_events
        (event_type = "telemetry_batch").
    chunk_word_counts:
        {chunk_id → word_count} from the calibration document's chunks.
    duration_seconds:
        Total active session duration in seconds.

    Returns
    -------
    A dict matching the BaselineData schema.
    """
    if not batches:
        return {
            "wpm_mean": 0.0,
            "wpm_std": 0.0,
            "scroll_velocity_mean": 0.0,
            "scroll_velocity_std": 0.0,
            "scroll_jitter_mean": 0.0,
            "idle_ratio_mean": 0.0,
            "regress_rate_mean": 0.0,
            "paragraph_dwell_mean": 0.0,
            "calibration_duration_seconds": duration_seconds,
        }

    # ── Scroll velocity ───────────────────────────────────────────────────────
    sv = scroll_velocities(batches)
    sv_mean = round(mean(sv), 2)
    sv_std = round(stdev(sv) if len(sv) > 1 else 0.0, 2)

    # ── Jitter / regression rate ──────────────────────────────────────────────
    jitter = scroll_jitter_values(batches)
    jitter_mean = round(mean(jitter) if jitter else 0.0, 4)

    # ── Idle ratio ────────────────────────────────────────────────────────────
    idle = idle_ratios(batches)
    idle_mean = round(mean(idle) if idle else 0.0, 4)

    # ── Paragraph dwell ───────────────────────────────────────────────────────
    dwells = paragraph_dwells(batches)
    dwell_seconds = [c * _BATCH_WINDOW_S for c in dwells.values()]
    dwell_mean = round(mean(dwell_seconds) if dwell_seconds else 0.0, 2)

    # ── WPM ───────────────────────────────────────────────────────────────────
    wpm = round(
        estimate_wpm(batches, chunk_word_counts, duration_seconds, total_words),
        1,
    )

    return {
        "wpm_mean": wpm,
        "wpm_std": 0.0,  # v1: single-pass estimate; no per-batch granularity yet
        "scroll_velocity_mean": sv_mean,
        "scroll_velocity_std": sv_std,
        "scroll_jitter_mean": jitter_mean,
        "idle_ratio_mean": idle_mean,
        "regress_rate_mean": jitter_mean,  # v1: same proxy as jitter
        "paragraph_dwell_mean": dwell_mean,
        "calibration_duration_seconds": duration_seconds,
    }
