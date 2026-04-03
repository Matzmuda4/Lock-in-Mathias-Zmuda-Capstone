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

        primary_state = max(dist, key=dist.get)  # type: ignore[arg-type]

        n_batches_norm = fvec[15]  # index 15 = n_batches_norm
        rationale = (
            f"RF: {primary_state} {dist[primary_state]:.3f} | "
            f"window={round(n_batches_norm * 16)}/16 batches | "
            f"latency={latency_ms}ms"
        )

        log.debug(
            "[rf_classify] primary=%s conf=%.3f latency=%dms",
            primary_state, dist[primary_state], latency_ms,
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
