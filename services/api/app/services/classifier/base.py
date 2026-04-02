"""
Classifier contracts — pure data types and the AbstractClassifier Protocol.

Nothing here imports FastAPI, SQLAlchemy, or httpx.
All concrete classifiers (RFClassifier, MockClassifier) implement
AbstractClassifier so the rest of the system depends only on this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

ATTENTIONAL_STATES: tuple[str, ...] = (
    "focused",
    "drifting",
    "hyperfocused",
    "cognitive_overload",
)

# Tolerance for floating-point probability sum validation
_SUM_TOLERANCE: float = 1e-4


@dataclass
class ClassificationResult:
    """
    Output of one classifier call for a single 10-second state packet.

    Probabilities are calibrated floats in [0, 1] summing to 1.0.
    The RF classifier (CalibratedClassifierCV + isotonic regression) produces
    these directly — no rounding or int conversion is applied.

    The rationale field carries a human-readable confidence note for the RF
    (e.g. "RF: focused 0.912 | window=16 batches") that downstream consumers
    (intervention LLM, debug UI) can use as context.
    """

    focused: float
    drifting: float
    hyperfocused: float
    cognitive_overload: float
    primary_state: str       # argmax of the four probabilities
    rationale: str           # confidence note / chain-of-thought
    latency_ms: int          # end-to-end inference time in ms
    parse_ok: bool = True    # False when output could not be parsed (LLM fallback)

    def as_dict(self) -> dict[str, Any]:
        return {
            "focused":            self.focused,
            "drifting":           self.drifting,
            "hyperfocused":       self.hyperfocused,
            "cognitive_overload": self.cognitive_overload,
            "primary_state":      self.primary_state,
            "rationale":          self.rationale,
            "latency_ms":         self.latency_ms,
            "parse_ok":           self.parse_ok,
        }

    def distribution_valid(self) -> bool:
        """True iff all probabilities are non-negative and sum to ≈ 1.0."""
        vals = [self.focused, self.drifting, self.hyperfocused, self.cognitive_overload]
        return (
            all(v >= 0.0 for v in vals)
            and abs(sum(vals) - 1.0) < _SUM_TOLERANCE
        )

    def as_intervention_context(self) -> dict[str, Any]:
        """
        Structured context dict passed to the intervention LLM prompt builder.

        This method is the integration point between the classifier and the
        future intervention generator.  The intervention LLM receives this dict
        as part of its input prompt so it can tailor the response to the
        confidence and state uncertainty of the classification.

        Extend this method (not the LLM prompt) when new fields become available.
        """
        return {
            "primary_state":  self.primary_state,
            "confidence":     round(max(
                self.focused, self.drifting,
                self.hyperfocused, self.cognitive_overload
            ), 4),
            "distribution": {
                "focused":            round(self.focused, 4),
                "drifting":           round(self.drifting, 4),
                "hyperfocused":       round(self.hyperfocused, 4),
                "cognitive_overload": round(self.cognitive_overload, 4),
            },
            "ambiguous": (
                # Flag if the top-2 states are within 0.15 of each other
                sorted([self.focused, self.drifting,
                        self.hyperfocused, self.cognitive_overload],
                       reverse=True)[0]
                - sorted([self.focused, self.drifting,
                          self.hyperfocused, self.cognitive_overload],
                         reverse=True)[1]
            ) < 0.15,
        }


@runtime_checkable
class AbstractClassifier(Protocol):
    """
    Protocol every concrete classifier must satisfy.
    Depends only on plain Python dicts — no ORM or HTTP layer.
    """

    async def classify(self, packet_json: dict[str, Any]) -> ClassificationResult:
        """
        Classify one state packet.

        Parameters
        ----------
        packet_json : raw packet dict as produced by store._build_packet_json().
            Keys: session_id, user_id, features, z_scores, drift,
                  baseline_snapshot, ui_aggregates, debug.

        Returns
        -------
        ClassificationResult — always returns something; parse_ok=False
        when the underlying model could not produce a valid result.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the underlying model/service is ready."""
        ...
