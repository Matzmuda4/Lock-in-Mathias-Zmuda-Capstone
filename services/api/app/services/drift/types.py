"""
Shared data-classes for the Phase 7 drift pipeline.

All types are pure dataclasses — no DB or FastAPI coupling here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class WindowFeatures:
    """
    Feature vector extracted from the last N telemetry batches (rolling 30 s window).
    """

    n_batches: int = 0

    # ── Scroll signals ────────────────────────────────────────────────────────
    scroll_velocity_norm_mean: float = 0.0
    scroll_velocity_norm_std: float = 0.0
    # std/mean of per-batch scroll velocity — high = erratic bursting
    scroll_burstiness: float = 0.0
    scroll_jitter_mean: float = 0.0
    regress_rate_mean: float = 0.0

    # ── Pace / WPM signals ────────────────────────────────────────────────────
    window_wpm_effective: float = 0.0
    pace_ratio: float = 1.0
    pace_dev: float = 0.0
    # True only when >= 2 distinct paragraphs AND >= 10 s effective time
    pace_available: bool = False
    paragraphs_observed: int = 0

    # ── Idle / engagement signals ─────────────────────────────────────────────
    idle_ratio_mean: float = 0.0
    scroll_pause_mean: float = 0.0
    # Fraction of batches with scroll_pause >= personalized threshold
    long_pause_share: float = 0.0
    focus_loss_rate: float = 0.0

    # ── Mouse signals ─────────────────────────────────────────────────────────
    # Returns 1.0 (neutral) when mouse is stationary (< 10 px path)
    mouse_efficiency_mean: float = 1.0
    mouse_path_px_mean: float = 0.0

    # ── Reading-position signals ──────────────────────────────────────────────
    # Raw fraction of window dominated by one paragraph_id (0..1)
    stagnation_ratio: float = 0.0
    # Keep old name as alias for backward compatibility
    paragraph_stagnation: float = 0.0

    # ── Progress signals ──────────────────────────────────────────────────────
    # Count of "progress_marker" events in the last 30 s
    progress_markers_count: int = 0


@dataclass
class ZScores:
    """
    Normalized deviations from the user's calibration baseline.
    Each term is capped at Z_POS_CAP (3.0).
    """
    z_idle: float = 0.0
    z_focus_loss: float = 0.0
    z_jitter: float = 0.0
    z_regress: float = 0.0
    z_pause: float = 0.0
    z_stagnation: float = 0.0
    z_mouse: float = 0.0
    z_pace: float = 0.0
    # Asymmetric skimming signal (only > 0 when pace_ratio > SKIM_THRESHOLD)
    z_skim: float = 0.0
    z_burstiness: float = 0.0


@dataclass
class DriftResult:
    """Full output of one drift computation cycle."""

    # ── Primary bidirectional drift state ─────────────────────────────────────
    drift_level: float = 0.0   # raw level [0,1], updated every cycle
    drift_ema: float = 0.0     # EMA-smoothed for UI display
    disruption_score: float = 0.0
    engagement_score: float = 0.0
    confidence: float = 0.0
    pace_ratio: Optional[float] = None
    pace_available: bool = False

    # ── Legacy / compatibility fields ─────────────────────────────────────────
    # beta_effective: computed decay rate before EMA smoothing
    beta_effective: float = 0.0
    # beta_ema: smoothed decay rate (pass back as prev_beta_ema next cycle)
    beta_ema: float = 0.0
    # attention_score = 1 - drift_level  (A(t) = exp(-beta_ema*t))
    attention_score: float = 1.0
    # drift_score = drift_level (same value, alternate name for API compat)
    drift_score: float = 0.0

    # ── Debug breakdown ───────────────────────────────────────────────────────
    beta_components: dict[str, Any] = field(default_factory=dict)
    features: WindowFeatures = field(default_factory=WindowFeatures)
    z_scores: ZScores = field(default_factory=ZScores)
