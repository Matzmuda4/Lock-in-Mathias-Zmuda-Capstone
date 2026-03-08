"""
Shared data-classes for the Phase 7 drift pipeline.

All types are pure dataclasses — no DB or FastAPI coupling here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WindowFeatures:
    """
    Feature vector extracted from the last N telemetry batches (rolling 30 s window).
    All values are already in [0, 1] or natural scales as documented.
    """

    # Number of 2-second batches that fed into this window
    n_batches: int = 0

    # ── Scroll signals ────────────────────────────────────────────────────────
    scroll_velocity_norm_mean: float = 0.0
    scroll_velocity_norm_std: float = 0.0
    scroll_jitter_mean: float = 0.0
    regress_rate_mean: float = 0.0

    # ── Pace / WPM signals ────────────────────────────────────────────────────
    window_wpm_effective: float = 0.0
    pace_ratio: float = 1.0
    pace_dev: float = 0.0
    # True only when sufficient evidence exists (>=2 paragraphs, >=10 s effective)
    pace_available: bool = False

    # ── Engagement signals ────────────────────────────────────────────────────
    idle_ratio_mean: float = 0.0
    scroll_pause_mean: float = 0.0
    focus_loss_rate: float = 0.0

    # ── Mouse signals ─────────────────────────────────────────────────────────
    mouse_efficiency_mean: float = 1.0  # default to 1 (efficient) when no movement
    # Mean mouse path in px over the window — used to decide if mouse is active
    mouse_path_px_mean: float = 0.0

    # ── Reading-position signals ──────────────────────────────────────────────
    # Proportion of window dominated by a single paragraph_id (> 0.8 threshold)
    paragraph_stagnation: float = 0.0


@dataclass
class ZScores:
    """
    Normalized deviations from the user's calibration baseline.
    Positive values mean "worse than baseline" for signals where high = bad.
    Capped at 2.0 per term (prevents single-term domination).
    """
    z_idle: float = 0.0
    z_focus_loss: float = 0.0
    z_jitter: float = 0.0
    z_regress: float = 0.0
    z_pause: float = 0.0
    z_stagnation: float = 0.0
    z_mouse: float = 0.0
    z_pace: float = 0.0


@dataclass
class DriftResult:
    """Full output of one drift computation cycle."""
    # Final beta after confidence gating
    beta_effective: float = 0.03
    # Smoothed beta EMA — stored in DB for next cycle
    beta_ema: float = 0.03
    attention_score: float = 1.0
    drift_score: float = 0.0
    drift_ema: float = 0.0
    confidence: float = 0.0
    # Per-term breakdown for the debug endpoint
    beta_components: dict[str, Any] = field(default_factory=dict)

    features: WindowFeatures = field(default_factory=WindowFeatures)
    z_scores: ZScores = field(default_factory=ZScores)
