"""
MockClassifier — used during development and CI when no model file is available.

Set CLASSIFY_USE_MOCK=true in .env to activate.
Returns a deterministic focused distribution (probabilities sum to 1.0)
so the full API pipeline can be exercised without the pkl file.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import ClassificationResult


class MockClassifier:
    """
    Returns a deterministic focused distribution as calibrated floats.
    Useful to verify the full pipeline wiring before the RF model is deployed.
    """

    async def classify(self, packet_json: dict[str, Any]) -> ClassificationResult:
        await asyncio.sleep(0)  # yield to event loop
        return ClassificationResult(
            focused=0.70,
            drifting=0.20,
            hyperfocused=0.05,
            cognitive_overload=0.05,
            primary_state="focused",
            rationale=(
                "[MockClassifier] Development placeholder. "
                "Replace with RFClassifier when model file is available."
            ),
            latency_ms=0,
            parse_ok=True,
        )

    async def health_check(self) -> bool:
        return True
