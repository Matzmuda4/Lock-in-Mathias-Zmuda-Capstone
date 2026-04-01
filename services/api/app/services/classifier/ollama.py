"""
OllamaClassifier — production classifier that calls a locally-running
Ollama model (the fine-tuned + merged + GGUF adapter).

Setup when adapter is ready:
  1. In Colab:
       model = model.merge_and_unload()
       model.save_pretrained("/content/drive/MyDrive/exports/merged_model")
       tokenizer.save_pretrained("/content/drive/MyDrive/exports/merged_model")

  2. On your Mac, convert to GGUF:
       python llama.cpp/convert_hf_to_gguf.py /path/to/merged_model \\
           --outfile lock-in-classifier.gguf --outtype q4_k_m

  3. Create an Ollama Modelfile:
       FROM ./lock-in-classifier.gguf
       PARAMETER temperature 0.0
       PARAMETER stop "<|im_end|>"

  4. Register:
       ollama create lock-in-classifier -f Modelfile

  5. Set in .env:
       CLASSIFY_ENABLED=true
       OLLAMA_CLASSIFIER_MODEL=lock-in-classifier
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from .base import ATTENTIONAL_STATES, ClassificationResult
from .prompt import SYSTEM_PROMPT

log = logging.getLogger(__name__)

_FALLBACK_RESULT = ClassificationResult(
    focused=0,
    drifting=0,
    hyperfocused=0,
    cognitive_overload=0,
    primary_state="focused",
    rationale="[parse_error] Could not extract a valid distribution from model output.",
    latency_ms=0,
    parse_ok=False,
)


def _parse_output(text: str, latency_ms: int) -> ClassificationResult:
    """
    Parse the three-line model output:
        Rationale: ...
        Primary State: <state>
        {"focused": F, "drifting": D, "hyperfocused": H, "cognitive_overload": C}
    """
    rationale = ""
    primary_state: str | None = None

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("rationale:"):
            rationale = stripped[len("rationale:"):].strip()
        elif stripped.lower().startswith("primary state:"):
            raw = stripped[len("primary state:"):].strip().lower()
            for s in ATTENTIONAL_STATES:
                if s in raw:
                    primary_state = s
                    break

    # Extract JSON object
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        log.warning("Classifier: no JSON object found in output: %r", text[:200])
        return ClassificationResult(**{**_FALLBACK_RESULT.__dict__, "latency_ms": latency_ms})

    try:
        obj = json.loads(text[start:end + 1])
        dist: dict[str, int] = {}
        for k in ATTENTIONAL_STATES:
            dist[k] = int(obj.get(k, 0))

        total = sum(dist.values())
        if total != 100 or any(v < 0 for v in dist.values()):
            log.warning(
                "Classifier: distribution invalid (sum=%d): %s", total, dist
            )
            return ClassificationResult(**{**_FALLBACK_RESULT.__dict__, "latency_ms": latency_ms})

        # Round to nearest 5 if not already multiples of 5
        if any(v % 5 != 0 for v in dist.values()):
            dist = {k: round(v / 5) * 5 for k, v in dist.items()}
            diff = 100 - sum(dist.values())
            if diff != 0:
                max_k = max(dist, key=dist.get)  # type: ignore[arg-type]
                dist[max_k] += diff

        if primary_state is None:
            primary_state = max(dist, key=dist.get)  # type: ignore[arg-type]

        return ClassificationResult(
            focused=dist["focused"],
            drifting=dist["drifting"],
            hyperfocused=dist["hyperfocused"],
            cognitive_overload=dist["cognitive_overload"],
            primary_state=primary_state,
            rationale=rationale,
            latency_ms=latency_ms,
            parse_ok=True,
        )

    except Exception as exc:
        log.warning("Classifier: JSON parse error — %s. Raw: %r", exc, text[:200])
        return ClassificationResult(**{**_FALLBACK_RESULT.__dict__, "latency_ms": latency_ms})


class OllamaClassifier:
    """
    Calls the locally-running Ollama inference server.

    Uses the Ollama /api/chat endpoint with temperature=0 for
    deterministic output.  max_tokens is generous (~400) to fit the
    full rationale + primary state + JSON.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "lock-in-classifier",
        timeout: float = 45.0,
    ) -> None:
        self._chat_url   = f"{base_url}/api/chat"
        self._health_url = f"{base_url}/api/tags"
        self._model      = model
        self._timeout    = timeout

    async def classify(self, llm_input: dict[str, Any]) -> ClassificationResult:
        t0 = time.monotonic()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(llm_input, ensure_ascii=False)},
        ]
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(
                    self._chat_url,
                    json={
                        "model":    self._model,
                        "messages": messages,
                        "stream":   False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 450,  # enough for rationale + state + JSON
                        },
                    },
                )
                r.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Classifier: HTTP error — %s", exc)
            return ClassificationResult(
                **{**_FALLBACK_RESULT.__dict__,
                   "rationale": f"[http_error] {exc}",
                   "latency_ms": int((time.monotonic() - t0) * 1000)}
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        raw_text   = r.json()["message"]["content"]
        result     = _parse_output(raw_text, latency_ms)

        log.debug(
            "Classifier: session — primary=%s latency=%dms parse_ok=%s",
            result.primary_state,
            latency_ms,
            result.parse_ok,
        )
        return result

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(self._health_url)
                if r.status_code != 200:
                    return False
                # Check the model is actually loaded
                models = [m.get("name", "") for m in r.json().get("models", [])]
                return any(self._model in m for m in models)
        except Exception:
            return False
