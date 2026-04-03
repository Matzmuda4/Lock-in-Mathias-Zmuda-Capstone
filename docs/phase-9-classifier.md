# Phase 9 — Attentional-State LLM Classifier (`classify` branch)

> **Status:** Infrastructure complete. Classifier wired and running. DB persistence pending (no migration required yet — results served from in-memory cache until adapter is validated).

---

## Overview

Phase 9 introduces the attentional-state classifier: a fine-tuned large language model (Qwen 2.5-7B-Instruct with a QLoRA adapter) that consumes the 30-second state packets produced every ~10 seconds by the Phase 8 pipeline and outputs a soft probability distribution over four attentional states:

| State | Description |
|---|---|
| `focused` | Active, productive reading at or near calibration pace |
| `drifting` | Attentional lapse — idle, tab-switching, or erratic scroll |
| `hyperfocused` | Deeply absorbed reading significantly above calibration pace |
| `cognitive_overload` | Struggling — active rereading, stagnation, slow or no progress |

The output is a **soft distribution** (not a hard one-hot label), e.g.:

```json
{"focused": 55, "drifting": 25, "hyperfocused": 5, "cognitive_overload": 15}
```

Values are non-negative multiples of 5, summing to exactly 100. This design captures the inherent ambiguity of attentional states — a reader can be simultaneously somewhat focused and somewhat drifting.

The classifier uses **chain-of-thought output**: before emitting the JSON distribution the model writes a `Rationale:` line and a `Primary State:` line, making its reasoning auditable and improving accuracy on rare states.

---

## Architecture and SOLID Design

The classifier is designed as an independent, optional layer that sits **above** the existing drift pipeline. The drift model always runs unaffected — the classifier is fire-and-forget, never blocking the 2-second telemetry cycle.

```
services/api/app/
├── services/
│   └── classifier/
│       ├── __init__.py
│       ├── base.py         AbstractClassifier Protocol + ClassificationResult
│       ├── prompt.py       System prompt — single source of truth
│       ├── formatter.py    Converts packet_json → LLM input dict
│       ├── ollama.py       OllamaClassifier (production)
│       ├── mock.py         MockClassifier (dev / CI)
│       ├── cache.py        In-memory per-session result cache
│       └── registry.py     Module-level singleton (set at startup)
└── routers/
    └── classification.py   GET /classifier/health
                            GET /sessions/{id}/attentional-state
                            GET /sessions/{id}/attentional-state/history
```

### SOLID principles applied

**Single Responsibility**
Each module has one job. `formatter.py` only transforms data structures. `ollama.py` only handles HTTP and output parsing. `cache.py` only manages in-memory state. `registry.py` only holds the live singleton. No module touches the DB, HTTP routing, or drift maths.

**Open/Closed**
The system is open for extension, closed for modification. When a second model version is trained, a new concrete class (e.g. `OllamaClassifierV2`) is created that implements `AbstractClassifier`. No existing code changes. The same principle applies to the planned intervention generator: it will depend on the same `AbstractClassifier` interface.

**Liskov Substitution**
`MockClassifier` and `OllamaClassifier` both satisfy `AbstractClassifier`. The registry, router, and background task never call methods not on the protocol — any compliant implementation can be swapped in transparently.

**Interface Segregation**
The classifier protocol exposes only two methods: `classify(llm_input) → ClassificationResult` and `health_check() → bool`. Nothing about HTTP, DB, or Ollama internals leaks into the protocol.

**Dependency Inversion**
High-level modules (`drift.py`, `routers/classification.py`) depend on the `AbstractClassifier` abstraction registered in `registry.py`, not on `OllamaClassifier` directly. The concrete implementation is resolved once at startup in `lifespan()`.

---

## What was added

### 1. `services/classifier/base.py` — Contracts

**`ClassificationResult`** dataclass:

```python
@dataclass
class ClassificationResult:
    focused:            int     # % weight, multiple of 5
    drifting:           int
    hyperfocused:       int
    cognitive_overload: int
    primary_state:      str     # argmax of the four values
    rationale:          str     # chain-of-thought from the LLM
    latency_ms:         int     # end-to-end inference time
    parse_ok:           bool    # False if model output couldn't be parsed
```

**`AbstractClassifier`** is a `@runtime_checkable` `Protocol`. Any class with `async def classify(...)` and `async def health_check(...)` satisfies it — no inheritance required.

---

### 2. `services/classifier/prompt.py` — System prompt

The system prompt is defined **once** in this file. It is identical to the prompt used during SFT fine-tuning in the Colab notebook, ensuring that inference conditions match training conditions exactly.

The prompt covers:
- All ten z-score signals with their thresholds and research basis
- All key feature fields and their interpretation
- UI aggregate fields
- Drift field semantics (secondary hints, not primary drivers)
- All four state definitions with precise firing conditions
- Strict output format rules (three lines: Rationale, Primary State, JSON)

When the model is retrained with an updated prompt, this file is the only thing that changes — the router, formatter, and cache are unaffected.

---

### 3. `services/classifier/formatter.py` — Packet formatter

Converts the internal `_build_packet_json` dict (stored in `session_state_packets`) to the exact input structure the LLM was trained on.

Training (`extract_inputs` in Colab) produced:

```python
{
    "meta":              { user_id, session_id, packet_seq, window_start_at,
                           window_end_at, session_mode, document_id, drift_included },
    "baseline_snapshot": { baseline_json, baseline_updated_at, baseline_valid },
    "features":          { ...WindowFeatures... },
    "z_scores":          { ...ZScores... },
    "ui_aggregates":     { ...ui context fractions... },
    "drift":             { drift_ema, disruption_score, engagement_score, ... },
}
```

`format_for_llm(packet_json, packet_seq, window_start_at, window_end_at)` produces exactly this structure. The formatter is the only place where the training-time and inference-time representations are coupled — if the training format changes, only this file needs updating.

---

### 4. `services/classifier/ollama.py` — Production classifier

`OllamaClassifier` calls the Ollama `/api/chat` endpoint with:
- `temperature: 0.0` — fully deterministic output
- `num_predict: 450` — enough tokens for the full rationale + primary state + JSON
- The system prompt from `prompt.py`
- The formatted packet as the user message

Output parsing (`_parse_output`) is defensive:
1. Extracts the `Rationale:` line into `rationale`
2. Extracts the `Primary State:` line and normalises to one of the four valid state strings
3. Finds the outermost `{...}` JSON object and parses it
4. Validates that values sum to 100 and are non-negative
5. Rounds to the nearest 5 if the model produced non-multiples (rare but possible)
6. Falls back to `parse_ok=False` with zero distribution on any failure — never raises

`health_check()` calls `GET /api/tags` and confirms the configured model name is in the response.

---

### 5. `services/classifier/mock.py` — Development classifier

`MockClassifier` returns a stable `{"focused": 70, "drifting": 20, "hyperfocused": 5, "cognitive_overload": 5}` result immediately. Used to verify the full pipeline wiring without a real model available.

Activate with `CLASSIFY_USE_MOCK=true` in `.env`.

---

### 6. `services/classifier/cache.py` — In-memory cache

`ClassificationCache` stores the latest `CachedClassification` per `session_id`:

```python
@dataclass
class CachedClassification:
    result:        ClassificationResult
    session_id:    int
    packet_seq:    int
    classified_at: datetime
```

- Capped at 500 sessions (LRU eviction by `classified_at`)
- No locking required — FastAPI's asyncio event loop is single-threaded
- Serves the `GET /sessions/{id}/attentional-state` endpoint with sub-millisecond reads

**Why in-memory rather than DB?** DB tables for classification results are planned but not yet migrated. The cache avoids schema lock-in while the model output format is still being validated. Once the adapter is confirmed to produce reliable output, the DB table will be added and `cache.put()` will also write to it — the endpoint already has a comment indicating the fallback path.

---

### 7. `services/classifier/registry.py` — Singleton registry

```python
_classifier: Optional[AbstractClassifier] = None
_cache: ClassificationCache = ClassificationCache()

def set_classifier(clf: AbstractClassifier) -> None: ...
def get_classifier() -> Optional[AbstractClassifier]: ...
def get_cache() -> ClassificationCache: ...
def is_available() -> bool: ...
```

Module-level rather than `app.state` because `store.py` and `drift.py` are pure-service modules without access to the FastAPI `Request` object. This is the same pattern used by `_upsert_counters` in `store.py`. The registry is set **once** during `lifespan()` and read many times — effectively a read-only singleton after startup.

---

### 8. `store.py` — `PacketWrittenInfo` return value

`upsert_drift_state()` now returns `(SessionDriftState, Optional[PacketWrittenInfo])`.

`PacketWrittenInfo` is non-`None` only on the cycle where a `SessionStatePacket` was written (every 5th call, ~10 seconds):

```python
@dataclass
class PacketWrittenInfo:
    packet_json:     dict
    packet_seq:      int
    window_start_at: datetime
    window_end_at:   datetime
```

This keeps `store.py` as a pure persistence layer — it signals **what** was written without knowing anything about the classifier. The caller (`_recompute_and_save` in `drift.py`) decides whether to fire classification, maintaining the single-responsibility boundary.

---

### 9. `drift.py` — Classification trigger

`_recompute_and_save` fires classification as a background asyncio task when `PacketWrittenInfo` is returned and the classifier is available:

```python
drift_row, packet_info = await upsert_drift_state(...)

if packet_info is not None:
    from app.services.classifier.registry import is_available
    if is_available():
        asyncio.create_task(
            _run_classification(session.id, packet_info),
            name=f"classify-{session.id}-{packet_info.packet_seq}",
        )
```

`_run_classification` is the only place that couples the drift pipeline to the classifier:

```python
async def _run_classification(session_id, packet_info) -> None:
    llm_input = format_for_llm(packet_info.packet_json, ...)
    result    = await get_classifier().classify(llm_input)
    get_cache().put(session_id, packet_info.packet_seq, result)
```

All exceptions are caught and logged — the drift cycle is never affected by classifier failures.

**Cadence:** Classification fires every ~10 seconds (same as state packet writing). Inference on a locally-running GGUF model typically takes 1–3 seconds, well within the window.

---

### 10. `routers/classification.py` — API endpoints

#### `GET /classifier/health`

No auth required. Returns:

```json
{
  "available": true,
  "classifier_type": "OllamaClassifier",
  "model_reachable": true,
  "cache_size": 3,
  "reason": null
}
```

| Field | Meaning |
|---|---|
| `available` | True if classifier is configured AND the Ollama model is reachable |
| `classifier_type` | `"OllamaClassifier"`, `"MockClassifier"`, or absent if none configured |
| `model_reachable` | Result of `health_check()` |
| `cache_size` | Number of sessions with a cached classification |
| `reason` | Human-readable failure reason if `available=false` |

#### `GET /sessions/{session_id}/attentional-state`

JWT-authenticated. Returns the latest classification for the session:

```json
{
  "session_id": 177,
  "packet_seq": 12,
  "classified_at": "2026-03-31T14:05:22.441Z",
  "distribution": {
    "focused": 55,
    "drifting": 25,
    "hyperfocused": 5,
    "cognitive_overload": 15
  },
  "primary_state": "focused",
  "rationale": "[1] z_idle=0.18 is near baseline... [2] z_skim=0.00...",
  "latency_ms": 1842,
  "parse_ok": true
}
```

Returns `404` if no classification is available yet (session is new or classifier is off).
Returns `503` if the classifier is not enabled, with a message explaining how to enable it.

#### `GET /sessions/{session_id}/attentional-state/history`

Placeholder. Returns `{"records": [], "message": "..."}` until the DB table is migrated.

---

### 11. `core/config.py` — New settings

| Setting | Default | Description |
|---|---|---|
| `CLASSIFY_ENABLED` | `false` | Master switch. Set to `true` when adapter is ready. |
| `CLASSIFY_USE_MOCK` | `false` | Use `MockClassifier` instead of Ollama (dev / CI). |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL. |
| `OLLAMA_CLASSIFIER_MODEL` | `lock-in-classifier` | Name of the registered Ollama model. |

---

### 12. `main.py` — Lifespan wiring

```python
if settings.classify_enabled:
    if settings.classify_use_mock:
        set_classifier(MockClassifier())
    else:
        clf = OllamaClassifier(settings.ollama_url, settings.ollama_classifier_model)
        if await clf.health_check():
            set_classifier(clf)
        else:
            log.warning("Ollama unreachable — classification disabled.")
```

The classifier is loaded once at startup. If Ollama is unreachable at startup (e.g. the model is not yet installed), the API starts normally — `is_available()` returns `False` and no classification tasks are fired. Classification can be enabled without restarting the API only by setting the registry after the fact (useful for testing).

---

## Model training pipeline

The adapter was trained using SFT + QLoRA (PEFT) in Google Colab:

| Component | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Quantisation | 4-bit NF4 double-quant via BitsAndBytes |
| LoRA rank | r=32, alpha=64 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj` |
| Training data | `TrainingData/supervised.jsonl` — 629 labelled packets |
| Train/val/test split | Group-aware (no temporal leakage across same session) |
| Optimiser | `paged_adamw_8bit` |
| Epochs | 5 (extended for small dataset) |
| Effective batch | 4 (per_device=1, grad_accum=4) |
| LR schedule | Cosine with 4-step warmup from 2e-4 |
| Precision | bf16 compute |
| Output format | Chain-of-thought rationale + Primary State line + JSON distribution |
| Label discretisation | All percentages rounded to nearest 5% |

### Training data composition (629 packets, 8 sessions, 4 users)

| State | Hard-label count | % of total |
|---|---|---|
| focused | 273 | 43.4% |
| cognitive_overload | 165 | 26.2% |
| drifting | 150 | 23.8% |
| hyperfocused | 41 | 6.5% |

`hyperfocused` is severely under-represented (15% of majority class). Session 178 (user 247) is a fully synthetic hyperfocus session added to ensure the rare class is represented. Additional real sessions will be labelled and added as the app is used.

---

## Deployment: installing the adapter

When the fine-tuned adapter (`adapter_model.safetensors`) is validated:

**Step 1 — Merge adapter into base model weights (run in Colab):**
```python
model = model.merge_and_unload()
model.save_pretrained("/content/drive/MyDrive/exports/merged_model")
tokenizer.save_pretrained("/content/drive/MyDrive/exports/merged_model")
```

**Step 2 — Convert to GGUF and quantise (run on Mac):**
```bash
python llama.cpp/convert_hf_to_gguf.py /path/to/merged_model \
    --outfile lock-in-classifier.gguf \
    --outtype q4_k_m      # ~4.5 GB, fits in 6 GB RAM
```

**Step 3 — Register with Ollama:**
```bash
cat > Modelfile << 'EOF'
FROM ./lock-in-classifier.gguf
PARAMETER temperature 0.0
PARAMETER stop "<|im_end|>"
EOF

ollama create lock-in-classifier -f Modelfile
ollama run lock-in-classifier   # smoke test
```

**Step 4 — Enable in `.env`:**
```
CLASSIFY_ENABLED=true
OLLAMA_CLASSIFIER_MODEL=lock-in-classifier
```

Restart the API. Classification begins automatically on the next session.

---

## Live data flow

```
Every 2 seconds:
  Frontend → POST /activity/batch
  → _recompute_and_save() runs drift model
  → upsert_drift_state() → (SessionDriftState, None)   ← no packet this cycle

Every ~10 seconds (5th batch):
  → upsert_drift_state() → (SessionDriftState, PacketWrittenInfo)
  → asyncio.create_task(_run_classification(session_id, packet_info))
      ↓  (background, non-blocking)
      → format_for_llm(packet_json, seq, window_start, window_end)
      → OllamaClassifier.classify(llm_input)   ← ~1–3 s on local GGUF
      → ClassificationCache.put(session_id, seq, result)

Frontend polling (every 10 s):
  → GET /sessions/{id}/attentional-state
  → reads ClassificationCache → returns latest result
```

---

## Graceful degradation

| Failure mode | Behaviour |
|---|---|
| `CLASSIFY_ENABLED=false` | API starts normally, no classification, drift works as always |
| Ollama not running at startup | Warning logged, `is_available()=False`, no tasks fired |
| Ollama times out during inference | `parse_ok=False` result cached, logged as warning |
| Model output unparseable | `parse_ok=False` with zero distribution stored; endpoint still responds |
| Cache miss (very new session) | `GET /attentional-state` returns `404` with explanation |

Drift monitoring is **never** affected by classifier availability. The two systems are fully independent — the classifier reads from the same packets the drift model writes but does not modify or depend on any drift state.

---

## Pending — Phase 9b

| Task | Notes |
|---|---|
| DB migration: `session_attentional_states` | Store classification history persistently; update `/history` endpoint |
| DB migration: `session_interventions` | For the intervention layer |
| Intervention generator | `services/intervention/` — rule-based first, then LLM-generated |
| Frontend classification panel | Poll `/attentional-state` and display state + rationale |
| Retraining with additional data | More sessions → better generalisation, especially for `hyperfocused` and `cognitive_overload` |

---

## References

- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
- Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*.
- Qwen Team (2024). Qwen2.5 Technical Report. Alibaba Group.
