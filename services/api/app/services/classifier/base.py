"""
Classifier contracts — pure data types and the AbstractClassifier Protocol.

Nothing here imports FastAPI, SQLAlchemy, or httpx.
All concrete classifiers (OllamaClassifier, MockClassifier) implement
AbstractClassifier so the rest of the system depends only on this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

ATTENTIONAL_STATES: tuple[str, ...] = (
    "focused",
    "drifting",
    "hyperfocused",
    "cognitive_overload",
)


@dataclass
class ClassificationResult:
    """
    Output of one classifier call for a 30-second state packet.

    focused + drifting + hyperfocused + cognitive_overload == 100
    (multiples of 5, soft probability distribution).
    """

    focused: int
    drifting: int
    hyperfocused: int
    cognitive_overload: int
    primary_state: str          # argmax of the four values
    rationale: str              # chain-of-thought from the LLM
    latency_ms: int             # end-to-end inference time in ms
    parse_ok: bool = True       # False when the model output couldn't be parsed

    def as_dict(self) -> dict[str, Any]:
        return {
            "focused": self.focused,
            "drifting": self.drifting,
            "hyperfocused": self.hyperfocused,
            "cognitive_overload": self.cognitive_overload,
            "primary_state": self.primary_state,
            "rationale": self.rationale,
            "latency_ms": self.latency_ms,
            "parse_ok": self.parse_ok,
        }

    def distribution_valid(self) -> bool:
        """True iff the four values are non-negative, sum to 100, and are multiples of 5."""
        vals = [self.focused, self.drifting, self.hyperfocused, self.cognitive_overload]
        return (
            all(v >= 0 for v in vals)
            and sum(vals) == 100
            and all(v % 5 == 0 for v in vals)
        )


@runtime_checkable
class AbstractClassifier(Protocol):
    """
    Protocol every concrete classifier must satisfy.
    Depends only on plain Python dicts — no ORM or HTTP layer.
    """

    async def classify(self, llm_input: dict[str, Any]) -> ClassificationResult:
        """
        Classify one state-packet input dict.

        Parameters
        ----------
        llm_input : dict produced by formatter.format_for_llm().
            Keys: meta, baseline_snapshot, features, z_scores,
                  ui_aggregates, drift.

        Returns
        -------
        ClassificationResult — always returns something; parse_ok=False
        when the model output couldn't be parsed.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the underlying model/service is reachable."""
        ...
