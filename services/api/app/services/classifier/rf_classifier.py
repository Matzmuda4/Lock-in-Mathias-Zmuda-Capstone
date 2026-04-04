"""
RFClassifier — production attentional-state classifier.

Loads the pre-trained Random Forest + isotonic calibration bundle saved by
the Colab training notebook (rf_classifier_v2.pkl).

Bundle structure (joblib dict):
  {
    "calibrated_clf":  CalibratedClassifierCV (the model used for inference),
    "rf_base":         RandomForestClassifier  (kept for feature importances),
    "feature_cols":    list[str]               (19 column names — for sanity check),
    "state_keys":      list[str]               (class order from the RF),
    "best_params":     dict                    (hyperparams for reproducibility),
    "seed":            int,
    "n_train":         int,
  }

Classification is synchronous (sklearn predict_proba is pure CPU) but wrapped
in asyncio.to_thread so the FastAPI event loop is never blocked.

SOLID adherence:
  Single Responsibility — feature extraction delegated to feature_extractor.py.
  Open/Closed          — new classifiers implement AbstractClassifier without
                         touching this file.
  Dependency Inversion — callers depend on AbstractClassifier, not this class.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Expected 19-column order — validated at load time against bundle["feature_cols"]
from .feature_extractor import FEATURE_COLS, build_feature_vector
from .base import ATTENTIONAL_STATES, ClassificationResult

# Lazy imports so sklearn is only required when RFClassifier is actually used
try:
    import numpy as np
    import joblib
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class RFClassifier:
    """
    Random Forest attentional-state classifier.

    Implements AbstractClassifier (duck-typed via Protocol).
    """

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self._clf: Any = None          # CalibratedClassifierCV
        self._state_keys: list[str] = list(ATTENTIONAL_STATES)
        self._loaded = False
        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the pkl bundle.  Logs an error but never raises."""
        if not _SKLEARN_AVAILABLE:
            log.error(
                "RFClassifier: scikit-learn / joblib not installed. "
                "Run: pip install scikit-learn joblib numpy"
            )
            return

        if not self._model_path.exists():
            log.error(
                "RFClassifier: model file not found at '%s'. "
                "Copy rf_classifier_v2.pkl from Google Drive to the repo root.",
                self._model_path,
            )
            return

        try:
            bundle = joblib.load(self._model_path)
            self._clf = bundle["calibrated_clf"]

            # Sanity-check feature column order matches what we compute live
            bundle_cols = list(bundle.get("feature_cols", []))
            if bundle_cols and bundle_cols != list(FEATURE_COLS):
                log.warning(
                    "RFClassifier: feature column mismatch!\n"
                    "  bundle : %s\n"
                    "  live   : %s\n"
                    "This means the model was trained on a different feature set. "
                    "Retrain or update feature_extractor.py.",
                    bundle_cols,
                    list(FEATURE_COLS),
                )

            # Determine class order from bundle (RF may reorder alphabetically)
            self._state_keys = list(bundle.get("state_keys", ATTENTIONAL_STATES))

            self._loaded = True
            log.info(
                "RFClassifier: loaded '%s'  n_train=%s  best_params=%s",
                self._model_path.name,
                bundle.get("n_train", "?"),
                bundle.get("best_params", {}),
            )
        except Exception as exc:
            log.error("RFClassifier: failed to load model — %s", exc)

    def is_loaded(self) -> bool:
        return self._loaded

    # ── AbstractClassifier interface ──────────────────────────────────────────

    async def classify(self, packet_json: dict[str, Any]) -> ClassificationResult:
        """
        Classify one full-window state packet.

        The caller (drift._run_classification) already gates on is_full_window()
        so this method trusts the packet is complete.  It returns parse_ok=False
        if the model is not loaded (should never happen in production).
        """
        if not self._loaded or self._clf is None:
            return _make_fallback(rationale="[rf_error] Model not loaded.")

        t0 = time.monotonic()
        fvec = build_feature_vector(packet_json)

        # Run predict_proba in a thread so the event loop is not blocked
        proba: list[float] = await asyncio.to_thread(self._predict, fvec)

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Map probabilities to ATTENTIONAL_STATES order (the RF may have
        # a different class_ order than our canonical order)
        dist = dict(zip(self._state_keys, proba))
        focused            = float(dist.get("focused",            0.0))
        drifting           = float(dist.get("drifting",           0.0))
        hyperfocused       = float(dist.get("hyperfocused",       0.0))
        cognitive_overload = float(dist.get("cognitive_overload", 0.0))

        # ── Engagement-context focused boosts ────────────────────────────────
        # Three post-classification heuristic corrections that prevent the RF
        # from misinterpreting deliberate system engagement as attentional drift.
        # All are analogous to Bayesian prior updating (Hattie & Timperley, 2007):
        # we inject domain knowledge to correct known training-distribution gaps
        # without retraining the model.  Each boost is logged in the rationale.
        ft: dict[str, Any] = packet_json.get("features") or {}

        # 1. Panel interaction — idle-in-panel misread as drift
        panel_share = float(ft.get("panel_interaction_share", 0.0))
        focused, drifting, hyperfocused, cognitive_overload, boost_note = (
            _apply_panel_boost(focused, drifting, hyperfocused, cognitive_overload, panel_share)
        )

        # 2. Section summary active — user is deliberately reading inline summary
        if ft.get("section_summary_active"):
            focused, drifting, hyperfocused, cognitive_overload, note = (
                _apply_fixed_boost(
                    focused, drifting, hyperfocused, cognitive_overload,
                    boost=0.20, label="summary_active",
                )
            )
            boost_note += note

        # 3. Text reformat active — altered layout changes expected scroll patterns
        if ft.get("text_reformat_active"):
            focused, drifting, hyperfocused, cognitive_overload, note = (
                _apply_fixed_boost(
                    focused, drifting, hyperfocused, cognitive_overload,
                    boost=0.12, label="reformat_active",
                )
            )
            boost_note += note
        # ─────────────────────────────────────────────────────────────────────

        primary_state = max(
            {"focused": focused, "drifting": drifting,
             "hyperfocused": hyperfocused, "cognitive_overload": cognitive_overload},
            key=lambda k: {"focused": focused, "drifting": drifting,
                           "hyperfocused": hyperfocused, "cognitive_overload": cognitive_overload}[k],
        )

        n_batches_norm = fvec[15]  # index 15 = n_batches_norm
        conf = max(focused, drifting, hyperfocused, cognitive_overload)
        rationale = (
            f"RF: {primary_state} {conf:.3f} | "
            f"window={round(n_batches_norm * 16)}/16 batches | "
            f"latency={latency_ms}ms"
            f"{boost_note}"
        )

        log.debug(
            "[rf_classify] primary=%s conf=%.3f panel_share=%.2f latency=%dms",
            primary_state, conf, panel_share, latency_ms,
        )

        return ClassificationResult(
            focused=focused,
            drifting=drifting,
            hyperfocused=hyperfocused,
            cognitive_overload=cognitive_overload,
            primary_state=primary_state,
            rationale=rationale,
            latency_ms=latency_ms,
            parse_ok=True,
        )

    def _predict(self, fvec: list[float]) -> list[float]:
        """Synchronous sklearn call — always run inside asyncio.to_thread."""
        X = np.array([fvec], dtype=float)
        proba = self._clf.predict_proba(X)[0]
        return [float(p) for p in proba]

    async def health_check(self) -> bool:
        """True if the model is loaded and can produce a prediction."""
        if not self._loaded:
            return False
        # Smoke-test with a zero vector
        try:
            fvec = [0.0] * len(FEATURE_COLS)
            await asyncio.to_thread(self._predict, fvec)
            return True
        except Exception as exc:
            log.warning("RFClassifier health check failed: %s", exc)
            return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_panel_boost(
    focused:             float,
    drifting:            float,
    hyperfocused:        float,
    cognitive_overload:  float,
    panel_interaction_share: float,
) -> tuple[float, float, float, float, str]:
    """
    Post-classification probability adjustment for panel engagement.

    Problem: the RF model was trained on data with very few panel-interaction
    examples (the panel UI did not exist during most data collection).  As a
    result, P(drifting) stays high when a user is actively reading the
    assistant panel because the model interprets reduced scroll activity as
    disengagement.

    Fix: linearly boost P(focused) when ``panel_interaction_share`` exceeds a
    threshold, redistributing the taken mass from the other three states
    proportionally so the distribution always sums to 1.0.  The boost
    disappears the moment the user scrolls back to the text.

    Thesis defensibility:
      Analogous to incorporating a Bayesian prior — we inject domain knowledge
      (deliberate panel engagement is a form of cognitive engagement, not drift)
      to correct a known training-distribution gap without retraining.
      The magnitude (max 30 pp at full panel engagement) is conservative enough
      to be overridden by strong drift signals in the other 18 features.

    Parameters
    ----------
    panel_interaction_share : fraction of the 30 s window spent in
                              PANEL_INTERACTING context (0.0 – 1.0).

    Returns
    -------
    (focused, drifting, hyperfocused, cognitive_overload, boost_note_str)
    """
    # Boost only kicks in when at least 15 % of the window was panel-engaged
    _THRESHOLD: float = 0.15
    # Maximum probability points transferred to focused at full engagement
    _MAX_BOOST: float = 0.30

    if panel_interaction_share <= _THRESHOLD:
        return focused, drifting, hyperfocused, cognitive_overload, ""

    # Boost strength: 0 at threshold, 1 at full panel engagement
    strength = min(
        (panel_interaction_share - _THRESHOLD) / (1.0 - _THRESHOLD),
        1.0,
    )
    add = strength * _MAX_BOOST

    new_focused   = min(1.0, focused + add)
    actual_added  = new_focused - focused   # may be < add if focused was near 1

    # Remove the same mass from the other three states proportionally
    other = drifting + hyperfocused + cognitive_overload
    if other > 1e-9:
        scale          = max(0.0, (other - actual_added) / other)
        drifting           *= scale
        hyperfocused       *= scale
        cognitive_overload *= scale

    # Re-normalise for floating-point safety
    total = new_focused + drifting + hyperfocused + cognitive_overload
    if total > 1e-9:
        focused            = new_focused / total
        drifting           = drifting    / total
        hyperfocused       = hyperfocused / total
        cognitive_overload = cognitive_overload / total
    else:
        focused, drifting, hyperfocused, cognitive_overload = 1.0, 0.0, 0.0, 0.0

    boost_note = f" | panel_boost={panel_interaction_share:.2f} +{actual_added:.2f}→F"
    return focused, drifting, hyperfocused, cognitive_overload, boost_note


def _apply_fixed_boost(
    focused:            float,
    drifting:           float,
    hyperfocused:       float,
    cognitive_overload: float,
    boost:              float,
    label:              str,
) -> tuple[float, float, float, float, str]:
    """
    Apply a fixed probability boost to P(focused) when a boolean context flag
    is True (e.g. section_summary_active, text_reformat_active).

    Unlike _apply_panel_boost (which uses a graduated share), this applies a
    constant ``boost`` amount regardless of magnitude — the flag is binary.

    The same proportional redistribution and re-normalisation as the panel
    boost is used to guarantee the distribution sums to 1.0.
    """
    new_focused  = min(1.0, focused + boost)
    actual_added = new_focused - focused

    other = drifting + hyperfocused + cognitive_overload
    if other > 1e-9:
        scale          = max(0.0, (other - actual_added) / other)
        drifting           *= scale
        hyperfocused       *= scale
        cognitive_overload *= scale

    total = new_focused + drifting + hyperfocused + cognitive_overload
    if total > 1e-9:
        focused            = new_focused / total
        drifting           = drifting    / total
        hyperfocused       = hyperfocused / total
        cognitive_overload = cognitive_overload / total
    else:
        focused, drifting, hyperfocused, cognitive_overload = 1.0, 0.0, 0.0, 0.0

    return focused, drifting, hyperfocused, cognitive_overload, f" | {label}+{actual_added:.2f}→F"


def _make_fallback(rationale: str) -> ClassificationResult:
    """Uniform fallback when the model cannot produce a result."""
    return ClassificationResult(
        focused=0.25,
        drifting=0.25,
        hyperfocused=0.25,
        cognitive_overload=0.25,
        primary_state="focused",
        rationale=rationale,
        latency_ms=0,
        parse_ok=False,
    )
