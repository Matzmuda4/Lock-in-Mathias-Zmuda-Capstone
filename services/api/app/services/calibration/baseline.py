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
) -> float:
    """
    WPM estimate using paragraph dwell tracking.

    A paragraph is counted as "read" when it appears as current_paragraph_id
    in at least one 2-second batch (≥2 s dwell).  Its word count is looked up
    from the document chunks.
    """
    seen_ids = set(paragraph_dwells(batches).keys())
    total_words = 0
    for pid in seen_ids:
        try:
            chunk_id = int(pid.replace("chunk-", ""))
            total_words += chunk_word_counts.get(chunk_id, 0)
        except (ValueError, AttributeError):
            pass
    duration_min = max(duration_seconds / 60.0, 0.1)
    return total_words / duration_min


# ─── Public entry point ───────────────────────────────────────────────────────

def compute_baseline(
    batches: list[dict[str, Any]],
    chunk_word_counts: dict[int, int],
    duration_seconds: int,
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
    wpm = round(estimate_wpm(batches, chunk_word_counts, duration_seconds), 1)

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
