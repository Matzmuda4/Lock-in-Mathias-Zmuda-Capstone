"""
MockClassifier — used during development and CI when no Ollama model is available.

Set CLASSIFY_USE_MOCK=true in .env to activate.
Returns a stable "focused" result so the endpoint works without a real model.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import ClassificationResult


class MockClassifier:
    """
    Returns a deterministic focused distribution.
    Useful to verify the full pipeline wiring before the adapter is ready.
    """

    async def classify(self, llm_input: dict[str, Any]) -> ClassificationResult:
        await asyncio.sleep(0)  # yield to event loop — keeps behaviour async
        return ClassificationResult(
            focused=70,
            drifting=20,
            hyperfocused=5,
            cognitive_overload=5,
            primary_state="focused",
            rationale=(
                "[MockClassifier] Development placeholder. "
                "Replace with OllamaClassifier when adapter is ready."
            ),
            latency_ms=0,
            parse_ok=True,
        )

    async def health_check(self) -> bool:
        return True
