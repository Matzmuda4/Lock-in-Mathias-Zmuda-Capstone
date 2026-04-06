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

### 2.3 Cooldown & Gate Logic

The LLM outputs `intervene: true | false` on every call:
- `intervene: true` — fire the intervention now (subject to backend gate check)
- `intervene: false` (tier ≠ none) — cooldown active; content is generated but suppressed
- `intervene: false` (tier = none) — no intervention warranted

The `session_context.cooldown_status` in the input prompt tells the model the
current gate state (`"clear"` or `"cooling"`). The backend enforces gate rules
independently via `ActiveInterventionTracker` in `engine.py`.

#### Backend gate rules (applied in order)

| Rule | Condition | Effect |
|------|-----------|--------|
| 1 | `break_suggestion` is active | Nothing else fires until break is acknowledged |
| 2 | `break_suggestion` requested AND < 5 min since last break resumed | `break_suggestion` blocked |
| 3 | Type is `gamification` or `ambient_sound` (PASSIVE) | Always fires freely — no slot or gap check |
| 4 | **Minimum gap** — dynamically set by attentional state: `drifting`/`cognitive_overload` → **0 s**; `focused`/`hyperfocused` → **10 s** | Prevents burst-firing on the same window during focus; allows rapid response during drift |
| 5 | Type is `chime` (INSTANT) | Subject only to rule 4 — fires and disappears; consumes no UI slot |
| 6 | Same type already on screen | Blocked — no duplicate types simultaneously |
| 7 | Type is a text prompt AND 2 text prompts already on screen | Blocked — max 2 text prompts simultaneously |
| 8 | 3 or more foreground items already on screen | Blocked — max 3 foreground interventions |

**Auto-dismiss:** Text prompts unacknowledged for > 90 s are silently removed,
freeing their slot for new interventions. This prevents permanently blocked slots
if a user ignores a card.

**Post-break cooldown:** After the user resumes from a break suggestion, only
`break_suggestion` is blocked for 5 minutes. All other intervention types are
free to fire immediately.

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

Real reading sessions from `supervised.jsonl`. Signal blocks (attentional states,
drift progression, text windows, session context) are drawn from genuine session
data where available, with synthetic context rows used only to fill gaps for
types that had insufficient real examples.

- **82%** of examples use real session signal blocks (genuine RF classifier
  outputs, real transition patterns, real `last_intervention` history)
- **18%** use synthetic context rows (programmatically generated state/drift
  combinations used to reach the 80-per-type target for underrepresented types)

For system-driven intervention types (`chime`, `text_reformat`, `ambient_sound`,
`none`), real session signal blocks were reused with content assigned
deterministically from `primary_state` and `drift_ema`. No labelling required
for these types.

### 5.2 Dataset Statistics (V2)

| Metric | Value |
|---|---|
| Total examples | 800 |
| Per-type count | Exactly 80 per type (stratified) |
| Real signal rows | 656 (82%) |
| Synthetic context rows | 144 (18%) |
| Pre-labelled (system-driven) | 320 — no ChatGPT labelling needed |
| Pending ChatGPT labelling | 480 — text-generative types |
| Training split | ~720 (90%, stratified) |
| Evaluation split | ~80 (10%, stratified by type) |

### 5.3 Intervention Type Distribution (V2 — balanced)

| Type | Count | Source | Labelling |
|---|---|---|---|
| `focus_point` | 80 | 69 real + 11 synthetic context | ChatGPT |
| `section_summary` | 80 | 80 real | ChatGPT |
| `comprehension_check` | 80 | 80 real | ChatGPT |
| `re_engagement` | 80 | 67 real + 13 synthetic context | ChatGPT |
| `gamification` | 80 | 60 real + 20 synthetic context | ChatGPT |
| `break_suggestion` | 80 | 46 real + 34 synthetic context | ChatGPT |
| `chime` | 80 | 80 real signal blocks reused | Pre-labelled (deterministic) |
| `text_reformat` | 80 | 80 real signal blocks reused | Pre-labelled (deterministic) |
| `ambient_sound` | 80 | 52 real + 28 synthetic context | Pre-labelled (deterministic) |
| `none` | 80 | 42 real + 38 synthetic context | Pre-labelled (content = null) |

### 5.4 Label Logic (Cooldown Rule)

The labelling rule ensures maximum content exposure for the fine-tuned model:

| Condition | `intervene` | `content` |
|---|---|---|
| `tier == "none"` | `false` | `null` |
| `tier != "none"` AND `cooldown == "clear"` | `true` | Full content generated |
| `tier != "none"` AND `cooldown == "cooling"` | `false` | Full content generated |

Cooling examples still have full content so the model learns intervention
generation in all contexts, while learning to suppress via `intervene: false`.

### 5.5 Dataset Files (V2)

| File | Description |
|---|---|
| `intervention_training_raw.jsonl` | Original rule-based skeletons (v1, 1,455 rows) |
| `intervention_training_v2_skeletons.jsonl` | Clean v2 skeletons — 800 rows, 80/type, pre-labelled system rows included |
| `intervention_training_v2_prelabelled.jsonl` | The 320 pre-labelled system-driven rows (chime, text_reformat, ambient_sound, none) |
| `batches/batch_NNN.json` | 12 labelling batches (40 rows each) — paste ChatGPT labels directly into these files |
| `intervention_training_v2_labelled.jsonl` | Final merged dataset — produced by `merge_labels.py` after labelling |
| `intervention_train_v2.jsonl` | ChatML-formatted training split — feed to QLoRA |
| `intervention_eval_v2.jsonl` | ChatML-formatted evaluation split |
| `LABELLING_PROMPT_V2.md` | Full labelling instructions and content schemas for ChatGPT |

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

## 8. Implementation Status

All core system components are implemented and functional:

| Component | Status | Location |
|---|---|---|
| Intervention service (Ollama client) | ✅ Done | `services/intervention/engine.py` |
| Prompt assembler | ✅ Done | `services/intervention/prompt.py` |
| Active intervention tracker / gate | ✅ Done | `engine.py → ActiveInterventionTracker` |
| Intervention router (trigger, manual fire, acknowledge) | ✅ Done | `routers/interventions.py` |
| Frontend: Focus Point, Re-engagement, Comprehension Check | ✅ Done | `components/interventions/` |
| Frontend: Section Summary, Text Reformat | ✅ Done | `components/interventions/` |
| Frontend: Journey Widget (gamification) | ✅ Done | `JourneyWidget.tsx` |
| Frontend: Break Suggestion overlay | ✅ Done | `BreakSuggestionOverlay.tsx` |
| Frontend: Audioscape (ambient sound) | ✅ Done | `AudioscapeWidget.tsx` |
| Frontend: Chime notification | ✅ Done | `ChimeWidget.tsx` |
| Frontend: Badge system | ✅ Done | `BadgePopup.tsx`, `ReaderPage.tsx` |
| Panel interaction boost (RF classifier) | ✅ Done | `rf_classifier.py → _apply_panel_boost` |
| Text-modification down-weighting | ✅ Done | `drift.py → _recompute_and_save` |

## 9. Remaining Work

1. **Complete ChatGPT labelling** — 12 batches × 40 rows in `batches/` need labels for the 6 text-generative types. Paste ChatGPT responses directly into each `batch_NNN.json`, then run `merge_labels.py` and `format_for_training.py`.
2. **Retrain the LLM** — upload `intervention_train_v2.jsonl` to Colab and run the QLoRA fine-tuning notebook (`train_lockin_llm.ipynb`).
3. **Merge adapter + update Ollama** — run `run_merge_and_convert.sh` locally, then `ollama create lockin-intervention -f Modelfile` to replace the current model.
4. **System prompt** — once the retrained model is deployed, tune `INTERVENTION_SYSTEM_PROMPT` in `prompt.py` based on observed intervention balance during test sessions.
