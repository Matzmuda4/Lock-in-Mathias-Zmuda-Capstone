"""
Baseline computation from telemetry_batch activity events — Phase 6 v2.

Takes the raw list of telemetry_batch payload dicts (one per 2-second window)
collected during a calibration session and produces a stable, generalisable
reading profile stored in user_baselines.baseline_json.

All functions are pure and exported for unit testing.

Baseline schema (v2)
--------------------
{
  "wpm_gross": float,               # total_words / total_duration_min
  "wpm_effective": float,           # words_read / effective_reading_min
  "words_read_estimated": int,      # words in paragraphs with >= 1 dwell window
  "effective_reading_seconds": float,

  "scroll_velocity_px_s_mean": float,   # px/s (raw, for debugging)
  "scroll_velocity_px_s_std": float,
  "scroll_velocity_norm_mean": float,   # px / (viewport_height * s), dimensionless
  "scroll_velocity_norm_std": float,

  "scroll_jitter_mean": float,      # direction_changes / scroll_events per window
  "scroll_jitter_std": float,

  "idle_ratio_mean": float,         # idle_seconds / 2.0 per window
  "idle_ratio_std": float,
  "idle_seconds_mean": float,
  "idle_seconds_std": float,

  "regress_rate_mean": float,       # neg_sum / (neg_sum + pos_sum) per window
  "regress_rate_std": float,

  "para_dwell_mean_s": float,
  "para_dwell_median_s": float,
  "para_dwell_iqr_s": float,        # Q3 - Q1
  "paragraph_count_observed": int,  # paragraphs with >= 1 dwell window (>=2 s)

  "presentation_profile": {
    "viewport_height_px_mean": float,
    "viewport_height_px_std": float,
    "viewport_width_px_mean": float,
    "viewport_width_px_std": float,
    "reader_container_height_px_mean": float,
    "calibration_text_word_count": int,
    "paragraph_count_total": int,
  },

  "calibration_duration_seconds": int,

  # Kept for backwards compatibility
  "wpm_mean": float,                # alias for wpm_gross
  "wpm_std": float,
  "scroll_velocity_mean": float,    # alias for scroll_velocity_px_s_mean
  "scroll_velocity_std": float,
  "paragraph_dwell_mean": float,    # alias for para_dwell_mean_s
  "regress_rate_mean_legacy": float,
}
"""

from __future__ import annotations

from statistics import mean, median, stdev
from typing import Any

# Window size — must match the frontend flush interval (2 000 ms).
_BATCH_WINDOW_S: float = 2.0

# Idle threshold: windows with idle_seconds > this are excluded from
# effective reading time.
_EFFECTIVE_IDLE_THRESHOLD_S: float = 1.5

_EPS: float = 1e-9   # prevents division by zero


# ─── Helpers (all pure, all exported) ─────────────────────────────────────────


def scroll_velocities(batches: list[dict[str, Any]]) -> list[float]:
    """Raw px/s velocity per batch window."""
    return [b.get("scroll_delta_abs_sum", 0.0) / _BATCH_WINDOW_S for b in batches]


def scroll_velocities_norm(
    batches: list[dict[str, Any]],
    fallback_viewport_px: float = 800.0,
) -> list[float]:
    """
    Viewport-normalised scroll velocity: px / (viewport_height_px * window_s).

    Values are dimensionless (fraction of viewport per second), making them
    comparable across window sizes.  Uses ``fallback_viewport_px`` when the
    batch does not carry viewport_height_px.
    """
    result = []
    for b in batches:
        vh = b.get("viewport_height_px") or fallback_viewport_px
        if vh <= 0:
            vh = fallback_viewport_px
        vel_norm = b.get("scroll_delta_abs_sum", 0.0) / (vh * _BATCH_WINDOW_S)
        result.append(vel_norm)
    return result


def scroll_jitter_values(batches: list[dict[str, Any]]) -> list[float]:
    """Direction-change ratio: changes / events per window (skips 0-event windows)."""
    vals = []
    for b in batches:
        events = b.get("scroll_event_count", 0)
        changes = b.get("scroll_direction_changes", 0)
        if events > 0:
            vals.append(changes / events)
    return vals


def idle_ratios(batches: list[dict[str, Any]]) -> list[float]:
    """idle_seconds / window_seconds, clamped [0, 1]."""
    return [min(b.get("idle_seconds", 0.0) / _BATCH_WINDOW_S, 1.0) for b in batches]


def paragraph_dwells(batches: list[dict[str, Any]]) -> dict[str, int]:
    """
    Map {paragraph_id → window_count}: how many 2-second windows each
    paragraph was the most-visible element.
    """
    counts: dict[str, int] = {}
    for b in batches:
        pid = b.get("current_paragraph_id")
        if pid:
            counts[pid] = counts.get(pid, 0) + 1
    return counts


def compute_paragraph_dwell_distribution(
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute dwell time distribution across observed paragraphs.

    Returns
    -------
    dict with keys:
      para_dwell_mean_s, para_dwell_median_s, para_dwell_iqr_s,
      paragraph_count_observed
    """
    dwells = paragraph_dwells(batches)
    dwell_secs = sorted(c * _BATCH_WINDOW_S for c in dwells.values())
    n = len(dwell_secs)
    if n == 0:
        return {
            "para_dwell_mean_s": 0.0,
            "para_dwell_median_s": 0.0,
            "para_dwell_iqr_s": 0.0,
            "paragraph_count_observed": 0,
        }
    med = median(dwell_secs)
    avg = mean(dwell_secs)
    # IQR using index-based quartiles
    q1 = _percentile(dwell_secs, 25)
    q3 = _percentile(dwell_secs, 75)
    return {
        "para_dwell_mean_s": round(avg, 2),
        "para_dwell_median_s": round(med, 2),
        "para_dwell_iqr_s": round(q3 - q1, 2),
        "paragraph_count_observed": n,
    }


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Return the p-th percentile (0–100) from a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def compute_regress_rate_stats(batches: list[dict[str, Any]]) -> dict[str, float]:
    """
    True regress rate using signed scroll sums.

    regress_rate_window = neg_sum / (neg_sum + pos_sum + eps)

    This measures what fraction of total absolute scrolling was *backward*
    (up) motion, which is a cleaner signal than direction-change count.
    """
    rates = []
    for b in batches:
        pos = b.get("scroll_delta_pos_sum", 0.0)
        neg = b.get("scroll_delta_neg_sum", 0.0)
        total = pos + neg
        if total > 0:
            rates.append(neg / total)
    if not rates:
        return {"regress_rate_mean": 0.0, "regress_rate_std": 0.0}
    return {
        "regress_rate_mean": round(mean(rates), 4),
        "regress_rate_std": round(stdev(rates) if len(rates) > 1 else 0.0, 4),
    }


def compute_effective_wpm(
    batches: list[dict[str, Any]],
    paragraph_word_counts: dict[str, int],
    total_words_override: int | None = None,
    idle_threshold_s: float = _EFFECTIVE_IDLE_THRESHOLD_S,
) -> dict[str, Any]:
    """
    Compute both gross and effective WPM.

    wpm_gross:
        total_words / total_duration_min
        (uses total_words_override when provided — most accurate for calibration)

    wpm_effective:
        words_read_estimated / effective_reading_min
        where effective_reading_min sums only windows that were:
          - window_focus_state == "focused"
          - idle_seconds <= idle_threshold_s

    words_read_estimated:
        sum of word counts for paragraphs that appeared as current_paragraph_id
        in at least one batch window.
    """
    n = len(batches)
    total_duration_s = n * _BATCH_WINDOW_S

    # Effective windows
    effective_windows = [
        b for b in batches
        if b.get("window_focus_state", "focused") == "focused"
        and b.get("idle_seconds", 0.0) <= idle_threshold_s
    ]
    effective_s = len(effective_windows) * _BATCH_WINDOW_S

    # Words read (paragraphs seen for >= 1 batch = >= 2 s)
    seen_ids = set(paragraph_dwells(batches).keys())
    words_read = 0
    for pid in seen_ids:
        if pid in paragraph_word_counts:
            words_read += paragraph_word_counts[pid]
        else:
            # Fallback: try to extract index from "calib-N" or "chunk-N"
            for prefix in ("calib-", "chunk-"):
                if pid.startswith(prefix):
                    try:
                        words_read += paragraph_word_counts.get(
                            f"{prefix}{pid[len(prefix):]}", 0
                        )
                    except Exception:
                        pass
                    break

    total_words = total_words_override if (
        total_words_override is not None and total_words_override > 0
    ) else sum(paragraph_word_counts.values()) or words_read

    wpm_gross = round(total_words / max(total_duration_s / 60.0, 0.1), 1)
    wpm_effective = round(words_read / max(effective_s / 60.0, 0.1), 1) if words_read > 0 else 0.0

    return {
        "wpm_gross": wpm_gross,
        "wpm_effective": wpm_effective,
        "words_read_estimated": words_read,
        "effective_reading_seconds": round(effective_s, 1),
    }


def compute_scroll_velocity_stats(batches: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean/std for both raw px/s and viewport-normalised velocity."""
    raw = scroll_velocities(batches)
    norm = scroll_velocities_norm(batches)
    return {
        "scroll_velocity_px_s_mean": round(mean(raw) if raw else 0.0, 2),
        "scroll_velocity_px_s_std": round(stdev(raw) if len(raw) > 1 else 0.0, 2),
        "scroll_velocity_norm_mean": round(mean(norm) if norm else 0.0, 4),
        "scroll_velocity_norm_std": round(stdev(norm) if len(norm) > 1 else 0.0, 4),
    }


def compute_idle_stats(batches: list[dict[str, Any]]) -> dict[str, float]:
    """Mean/std for idle ratio and raw idle seconds."""
    ratios = idle_ratios(batches)
    secs = [b.get("idle_seconds", 0.0) for b in batches]
    return {
        "idle_ratio_mean": round(mean(ratios) if ratios else 0.0, 4),
        "idle_ratio_std": round(stdev(ratios) if len(ratios) > 1 else 0.0, 4),
        "idle_seconds_mean": round(mean(secs) if secs else 0.0, 2),
        "idle_seconds_std": round(stdev(secs) if len(secs) > 1 else 0.0, 2),
    }


def compute_jitter_stats(batches: list[dict[str, Any]]) -> dict[str, float]:
    """Mean/std for direction-change jitter ratio."""
    vals = scroll_jitter_values(batches)
    return {
        "scroll_jitter_mean": round(mean(vals) if vals else 0.0, 4),
        "scroll_jitter_std": round(stdev(vals) if len(vals) > 1 else 0.0, 4),
    }


def compute_presentation_profile_stats(
    batches: list[dict[str, Any]],
    calibration_text_word_count: int = 0,
    paragraph_count_total: int = 0,
) -> dict[str, Any]:
    """
    Aggregate viewport dimension statistics from all batches.

    Stored so that Phase 7 can detect if a session is running at a very
    different window size from calibration and adjust normalisation accordingly.
    """
    vhs = [b["viewport_height_px"] for b in batches if b.get("viewport_height_px")]
    vws = [b["viewport_width_px"] for b in batches if b.get("viewport_width_px")]
    rchs = [b["reader_container_height_px"] for b in batches if b.get("reader_container_height_px")]

    def _stats(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        return round(mean(vals), 1), round(stdev(vals) if len(vals) > 1 else 0.0, 1)

    vh_mean, vh_std = _stats(vhs)
    vw_mean, vw_std = _stats(vws)
    rch_mean, _ = _stats(rchs)

    return {
        "viewport_height_px_mean": vh_mean,
        "viewport_height_px_std": vh_std,
        "viewport_width_px_mean": vw_mean,
        "viewport_width_px_std": vw_std,
        "reader_container_height_px_mean": rch_mean,
        "calibration_text_word_count": calibration_text_word_count,
        "paragraph_count_total": paragraph_count_total,
    }


# ─── Public entry point ────────────────────────────────────────────────────────


def estimate_wpm(
    batches: list[dict[str, Any]],
    chunk_word_counts: dict[int, int],
    duration_seconds: int,
    total_words_override: int | None = None,
) -> float:
    """
    Backwards-compatible WPM estimate (used by existing tests).

    When total_words_override is provided: wpm = total_words / duration_min.
    Otherwise: dwell-based — sum word counts for paragraphs seen >= 1 window,
    divide by duration_min (using the passed duration_seconds, not batch count).
    """
    duration_min = max(duration_seconds / 60.0, 0.1)

    if total_words_override is not None and total_words_override > 0:
        return round(total_words_override / duration_min, 1)

    # Build both int-index and "chunk-N" / "calib-N" keyed lookups
    para_wc: dict[str, int] = {}
    for cid, wc in chunk_word_counts.items():
        para_wc[str(cid)] = wc
        para_wc[f"chunk-{cid}"] = wc
        para_wc[f"calib-{cid}"] = wc

    seen_ids = set(paragraph_dwells(batches).keys())
    total_words = sum(para_wc.get(pid, 0) for pid in seen_ids)
    if total_words == 0:
        total_words = sum(chunk_word_counts.values())
    return round(total_words / duration_min, 1)


def compute_baseline(
    batches: list[dict[str, Any]],
    chunk_word_counts: dict[int, int],
    duration_seconds: int,
    total_words: int | None = None,
    paragraph_word_counts: dict[str, int] | None = None,
    calibration_text_word_count: int = 0,
    paragraph_count_total: int = 0,
) -> dict[str, Any]:
    """
    Compute the full reading baseline from a list of telemetry_batch payloads.

    Parameters
    ----------
    batches:
        Ordered list of payload dicts (event_type = "telemetry_batch").
    chunk_word_counts:
        {chunk_id (int) → word_count} from the calibration document's DB chunks.
    duration_seconds:
        Total active session duration.
    total_words:
        Known total word count for the calibration text (for gross WPM).
    paragraph_word_counts:
        {paragraph_id (str, e.g. "calib-0") → word_count} for effective WPM.
    calibration_text_word_count:
        Stored in presentation_profile for Phase 7 reference.
    paragraph_count_total:
        Total number of paragraphs in the calibration text.
    """
    _zero = {
        "wpm_gross": 0.0, "wpm_effective": 0.0,
        "words_read_estimated": 0, "effective_reading_seconds": 0.0,
        "scroll_velocity_px_s_mean": 0.0, "scroll_velocity_px_s_std": 0.0,
        "scroll_velocity_norm_mean": 0.0, "scroll_velocity_norm_std": 0.0,
        "scroll_jitter_mean": 0.0, "scroll_jitter_std": 0.0,
        "idle_ratio_mean": 0.0, "idle_ratio_std": 0.0,
        "idle_seconds_mean": 0.0, "idle_seconds_std": 0.0,
        "regress_rate_mean": 0.0, "regress_rate_std": 0.0,
        "para_dwell_mean_s": 0.0, "para_dwell_median_s": 0.0,
        "para_dwell_iqr_s": 0.0, "paragraph_count_observed": 0,
        "presentation_profile": compute_presentation_profile_stats(
            [], calibration_text_word_count, paragraph_count_total
        ),
        "calibration_duration_seconds": duration_seconds,
        # Legacy aliases
        "wpm_mean": 0.0, "wpm_std": 0.0,
        "scroll_velocity_mean": 0.0, "scroll_velocity_std": 0.0,
        "paragraph_dwell_mean": 0.0, "regress_rate_mean_legacy": 0.0,
    }

    if not batches:
        return _zero

    # ── Build paragraph word count map (string-keyed) ─────────────────────
    para_wc: dict[str, int] = paragraph_word_counts or {}
    if not para_wc:
        # Build from chunk_word_counts with both "calib-N" and "chunk-N" keys
        for cid, wc in chunk_word_counts.items():
            para_wc[f"calib-{cid}"] = wc
            para_wc[f"chunk-{cid}"] = wc
            para_wc[str(cid)] = wc

    # ── WPM ───────────────────────────────────────────────────────────────
    wpm_stats = compute_effective_wpm(
        batches, para_wc, total_words_override=total_words
    )

    # ── Scroll velocity ───────────────────────────────────────────────────
    sv_stats = compute_scroll_velocity_stats(batches)

    # ── Jitter ────────────────────────────────────────────────────────────
    jitter_stats = compute_jitter_stats(batches)

    # ── Idle ──────────────────────────────────────────────────────────────
    idle_stats = compute_idle_stats(batches)

    # ── Regress rate ──────────────────────────────────────────────────────
    regress_stats = compute_regress_rate_stats(batches)

    # ── Paragraph dwell distribution ──────────────────────────────────────
    dwell_stats = compute_paragraph_dwell_distribution(batches)

    # ── Presentation profile ──────────────────────────────────────────────
    profile = compute_presentation_profile_stats(
        batches, calibration_text_word_count, paragraph_count_total
    )

    return {
        **wpm_stats,
        **sv_stats,
        **jitter_stats,
        **idle_stats,
        **regress_stats,
        **dwell_stats,
        "presentation_profile": profile,
        "calibration_duration_seconds": duration_seconds,
        # Legacy aliases (kept for backwards compatibility with existing code/tests)
        "wpm_mean": wpm_stats["wpm_gross"],
        "wpm_std": 0.0,
        "scroll_velocity_mean": sv_stats["scroll_velocity_px_s_mean"],
        "scroll_velocity_std": sv_stats["scroll_velocity_px_s_std"],
        "paragraph_dwell_mean": dwell_stats["para_dwell_mean_s"],
        "regress_rate_mean_legacy": jitter_stats["scroll_jitter_mean"],
    }
