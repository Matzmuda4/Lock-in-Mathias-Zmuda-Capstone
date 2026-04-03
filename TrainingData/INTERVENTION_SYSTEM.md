# Lock-in Intervention System — Architecture & Implementation

This document describes the intervention engine architecture implemented in the
`intervention` branch, covering the data pipeline, training dataset, model
fine-tuning, and the runtime integration wired into the existing Lock-in API.

---

## 1. Overview

The intervention system is a locally-running fine-tuned LLM (Qwen 2.5 7B,
QLoRA) that receives a structured JSON snapshot of a student's reading session
every ~10 seconds and decides whether and how to intervene to support focus and
comprehension. It operates as a "planner-generator": it decides the intervention
type, tier, and generates the intervention content in a single inference pass.

---

## 2. Intervention Framework

### 2.1 Tiers

| Tier | Intent | Typical Trigger |
|---|---|---|
| **Subtle** | Gentle nudge, preserve flow | Early drift, focused state (positive reinforcement) |
| **Moderate** | Clear redirect | 2+ negative packets, rising drift EMA |
| **Strong** | Hard re-entry | 3 consecutive negative packets, sustained overload |
| **Special** | Hyperfocus check | Sustained hyperfocus (>8 min) |

### 2.2 Intervention Types

| Type | Category | LLM Role | Frontend Action |
|---|---|---|---|
| `focus_point` | Text-generative | Writes a curiosity-spark prompt from `text_window` | Shows inline panel |
| `section_summary` | Text-generative | Summarises `text_window` into title/summary/key_point | Shows collapsible panel |
| `comprehension_check` | Text-generative | Generates T/F or highlight question from `text_window` | Shows interactive quiz |
| `re_engagement` | Text-generative | Writes a personalised re-engagement message | Shows modal panel |
| `ambient_sound` | System-driven | Selects sound type + fade duration | Frontend plays audio |
| `chime` | System-driven | Selects chime type + short message | Frontend triggers audio cue |
| `text_reformat` | System-driven | Specifies layout params (spacing, chunk size) | Frontend adjusts CSS |
| `break_suggestion` | System-driven | Selects duration + message | Frontend auto-pauses |
| `gamification` | System-driven | Selects badge/XP event + message | Frontend updates XP/badges |

### 2.3 Cooldown Logic

The LLM outputs `intervene: true | false` on every call:
- `intervene: true` — cooldown is clear, fire the intervention now
- `intervene: false` (tier ≠ none) — cooldown active, content is generated but suppressed; system schedules for next clear window
- `intervene: false` (tier = none) — no intervention warranted

The `session_context.cooldown_status` in the input prompt tells the model the
current cooldown state. The backend enforces it independently.

---

## 3. Runtime Data Pipeline

### 3.1 Signal Flow (every ~10 seconds)

```
Frontend telemetry batches (every 2s)
    ↓
activity.py  →  _recompute_and_save (drift.py)
    ↓
Drift model computes: drift_level, drift_ema, engagement_score, disruption_score
    ↓
store.py._build_packet_json  →  reads current_chunk_index from last telemetry batch
    ↓
SessionStatePacket written to DB (every 5th batch = 10s cadence)
    ↓
asyncio.create_task(_run_classification)           [fire-and-forget]
    ↓
RF Classifier → ClassificationResult (focused | drifting | hyperfocused | cognitive_overload)
    ↓
paragraph_fetcher.fetch_text_window(document_id, chunk_index, db)
    → SELECT text FROM document_chunks WHERE chunk_index IN [n-1, n, n+1]
    ↓
classifier_store.save_attentional_state  →  session_attentional_states row
    intervention_context JSONB = {
        primary_state, confidence, distribution,
        drift_level, drift_ema, engagement_score,
        current_chunk_index, text_window: [str, str, str],
        packet_seq, session_id
    }
```

### 3.2 Key Files Modified

| File | Change |
|---|---|
| `services/drift/store.py` | `_build_packet_json` — embeds `current_chunk_index` (int) from last telemetry batch into `reading_position` block |
| `routers/drift.py` | `_run_classification` — fetches text window and passes to `save_attentional_state`; text_modified down-weighting in `_recompute_and_save` |
| `services/classifier/classifier_store.py` | `save_attentional_state` — accepts `text_window`, extends `intervention_context` with drift fields + `text_window` |
| `services/classifier/paragraph_fetcher.py` | **New file** — `fetch_text_window` (async, DB) and `text_window_from_dict` (sync, dict for dataset builder) |

### 3.3 Paragraph Identification

The frontend renderer assigns `data-chunk-index` (the sequential integer
`DocumentChunk.chunk_index`) to each rendered paragraph element. The telemetry
batch includes this as `current_chunk_index`. The API embeds it in the
`SessionStatePacket` → `reading_position.current_chunk_index`.

`paragraph_fetcher.fetch_text_window` uses this integer directly in a
`WHERE chunk_index BETWEEN n-1 AND n+4` query on `document_chunks`, returning
up to 3 non-empty prose paragraphs centred on the current position. Image and
table chunks (empty text) are automatically skipped.

### 3.4 Text-Modification Down-weighting

When any telemetry batch in the current 30-second window has `text_modified:
true` (set by the frontend when `text_reformat` is active), the drift model
reduces `quality_confidence_mult` proportionally:

```
mult = max(0.5, 1.0 - 0.4 × (modified_batches / total_batches))
```

A fully-modified window halves confidence. A single modified batch (~7%) causes
only ~3% reduction. This prevents false-positive drift detections during layout
transitions.

---

## 4. LLM Input Prompt Structure

At inference time, the intervention LLM receives:

```json
{
  "session_context": {
    "elapsed_minutes": float,
    "session_stage": "early" | "mid",
    "last_intervention": {"type": str, "tier": str, "seconds_ago": int},
    "cooldown_status": "clear" | "cooling",
    "xp": int,
    "badges_earned": [str, ...]
  },
  "attentional_state_window": [
    {"primary_state": str, "confidence": float,
     "distribution": {"focused": float, "drifting": float,
                      "hyperfocused": float, "cognitive_overload": float}},
    ...  // last 3 classifications (30 seconds)
  ],
  "drift_progression": {
    "drift_level": [float, float, float],
    "engagement_score": [float, float, float],
    "drift_ema": float
  },
  "user_baseline": {
    "wpm_effective": float,
    "idle_ratio_mean": float,
    "regress_rate_mean": float,
    "para_dwell_median_s": float
  },
  "reading_context": {
    "current_paragraph_index": int | null,
    "text_window": [str, str, str]  // ≤3 paragraphs around current position
  }
}
```

This prompt is assembled from the last 3 `session_attentional_states` rows,
each of which carries its own `intervention_context` JSONB — no additional
joins required at inference time.

The LLM outputs:

```json
{
  "intervene": true | false,
  "tier": "subtle" | "moderate" | "strong" | "special" | "none",
  "type": "focus_point" | "section_summary" | ... | "none",
  "content": { ... } | null
}
```

---

## 5. Training Dataset

### 5.1 Source Data

Real reading sessions from `supervised.jsonl` (1,500 sessions, RF classifier
training data). Each window maps to a `SessionStatePacket`; `activity_events`
are queried by timestamp to recover `current_chunk_index` and look up real
paragraph text from `document_chunks`.

- **69%** of examples use exact real paragraph text (DB lookup)
- **12%** use approximate position (fallback midpoint)
- **19%** use synthetic text templates (DB unreachable for that session)

### 5.2 Dataset Statistics

| Metric | Value |
|---|---|
| Total examples | 1,455 |
| Training split | 1,310 (90%) |
| Evaluation split | 145 (10%, stratified by type) |
| Avg sequence length | ~990 tokens |
| Max sequence length | ~1,770 tokens (fits within 2048) |

### 5.3 Intervention Type Distribution

| Type | Count | % |
|---|---|---|
| section_summary | 466 | 32.0% |
| comprehension_check | 224 | 15.4% |
| re_engagement | 188 | 12.9% |
| gamification | 129 | 8.9% |
| focus_point | 118 | 8.1% |
| break_suggestion | 117 | 8.0% |
| none | 114 | 7.8% |
| chime | 44 | 3.0% |
| text_reformat | 33 | 2.3% |
| ambient_sound | 22 | 1.5% |

### 5.4 Label Logic (Cooldown Rule)

The labelling rule ensures maximum content exposure for the fine-tuned model:

| Condition | `intervene` | `content` |
|---|---|---|
| `tier == "none"` | `false` | `null` |
| `tier != "none"` AND `cooldown == "clear"` | `true` | Full content generated |
| `tier != "none"` AND `cooldown == "cooling"` | `false` | Full content generated |

Cooling examples still have full content so the model learns intervention
generation in all contexts, while learning to suppress via `intervene: false`.

### 5.5 Dataset Files

| File | Description |
|---|---|
| `intervention_training_raw.jsonl` | Rule-based tier/type hints, no labels |
| `intervention_training_labeled.jsonl` | ChatGPT-labelled (raw, with metadata) |
| `intervention_training_final.jsonl` | Cleaned: correct output schema, fixed duplicate Qs, diversified content |
| `intervention_train.jsonl` | ChatML-formatted, 1,310 examples — feed to QLoRA |
| `intervention_eval.jsonl` | ChatML-formatted, 145 examples — evaluation split |

---

## 6. Model Training

### 6.1 Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Qwen 2.5 7B Instruct | Strong JSON instruction-following, Ollama-compatible |
| Method | QLoRA (4-bit) | Fits A100 40GB; preserves base model language quality |
| LoRA rank | 32 | Good quality/memory balance on A100 |
| LoRA alpha | 64 (2× rank) | Standard starting point |
| Learning rate | 2e-4 | Cosine schedule with 5% warmup |
| Effective batch | 16 (4 × 4 grad accum) | Stable gradient updates |
| Epochs | 3 | Sufficient for 1,310 examples |
| Max seq length | 2048 | All examples fit |
| Framework | Unsloth + TRL SFTTrainer | ~2× faster than plain HF on A100 |

### 6.2 Output Format

After training, the model is exported as GGUF Q4\_K\_M for local serving via
Ollama:
```
ollama create lockin-intervention -f Modelfile
```

### 6.3 Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| JSON validity | % outputs parseable as JSON | >95% |
| Field completeness | All 4 required fields present | >95% |
| Logic accuracy | `intervene` matches `cooldown_status` rule | >90% |
| Content schema | `content` keys match intervention type | >85% |

---

## 7. DB Schema Additions (This Branch)

### `session_attentional_states.intervention_context` (extended)

Previously contained only `primary_state`, `confidence`, `distribution`,
`ambiguous`. Now also contains:

```json
{
  "drift_level":         float,
  "drift_ema":           float,
  "engagement_score":    float,
  "packet_seq":          int,
  "session_id":          int,
  "current_chunk_index": int | null,
  "text_window":         [str, str, str]
}
```

No schema migration required — `JSONB` is schemaless; old rows remain valid.

### `interventions` table (pre-existing, logging use)

```sql
CREATE TABLE interventions (
    id          SERIAL PRIMARY KEY,
    session_id  INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    type        VARCHAR(50) NOT NULL,   -- intervention type
    intensity   VARCHAR(20) NOT NULL,   -- tier
    payload     JSONB NOT NULL,         -- full LLM output content
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

This table is used to log every fired intervention (where `intervene: true`)
for post-session analysis and the cooldown tracker.

---

## 8. What Remains for Phase 9 (Next)

The training data pipeline and model configuration are complete. The following
must be implemented before the system is end-to-end functional:

1. **Intervention service** (`services/api/app/services/intervention/`) — Ollama client, prompt assembler (reads last 3 `session_attentional_states` rows), cooldown tracker, gamification/XP state
2. **Intervention router** (`routers/intervention.py`) — endpoints for the frontend to receive intervention decisions and acknowledge completion
3. **Frontend intervention components** — panel overlays, badge modals, comprehension check UI, break suggestion auto-pause, ambient sound player
4. **Cooldown enforcement** — in-memory per-session cooldown clock, backed by `interventions` table for persistence across reconnects
