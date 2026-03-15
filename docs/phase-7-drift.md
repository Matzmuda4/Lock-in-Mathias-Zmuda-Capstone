# Phase 7 — Personalised Mathematical Drift Modelling

> **Status:** Implemented (v3 — telemetry reliability fixes, data-quality guardrails, LLM-ready state packet)
>
> **No LLM, no interventions.** Pure mathematics + telemetry signals only.

---

## Changelog: v2 → v3 (current)

| Area | Change |
|---|---|
| **Frontend telemetry (A1)** | `idle_seconds` is now per-window (0–2 s), never cumulative. Cumulative value is sent separately as `idle_since_interaction_seconds` for debugging. |
| **Frontend telemetry (A2)** | Scroll deltas now computed from `scrollTop` of the actual scroll container; zero-delta events are skipped. |
| **Frontend telemetry (A3)** | `selectCurrentParagraph` uses a 3-level fallback (≥0.6 ratio → highest ratio → closest to viewport top) so `current_paragraph_id` is almost never `NONE` during reading. |
| **Frontend telemetry (A4)** | Dev-mode sanity warnings surface broken telemetry in the debug panel: idle > 2 s, scroll=0 with progress change, missing paragraph IDs. |
| **Backend guardrails (B1/B2)** | Each batch is tagged at ingestion with `telemetry_fault`, `scroll_capture_fault`, `paragraph_missing_fault`. Feature extraction reads these flags and applies a `quality_confidence_mult` penalty (×0.5 for idle fault, ×0.7 for scroll/para faults). |
| **Backend guardrails (B3)** | Calibration completeness check: at least one of duration ≥ 90 s, ≥ 3 paragraphs visited, or ≥ 3 batches recorded must be met; otherwise `POST /calibration/complete` returns `400`. Baseline validity flag `baseline_valid` (requires `wpm_effective > 0`) is included in all debug/state endpoints. |
| **Drift model (C1)** | Skimming fallback via `progress_velocity` (Δprogress_ratio / seconds) when pace estimation is unavailable due to missing paragraph IDs. |
| **Drift model (C2)** | Faster re-engagement: when `engagement_score > 0.6` and previous beta is elevated, the beta EMA alpha is temporarily doubled (0.20 → 0.40) so recovery is noticeably quicker. |
| **Drift history (C3)** | New `session_drift_history` Timescale hypertable stores a snapshot every ≈10 seconds so the full trajectory can be analysed and plotted (not just the latest state). |
| **LLM input packet (D)** | New endpoint `GET /sessions/{id}/state-packet` returns a structured JSON payload that the future Ollama LLM classifier will consume: baseline snapshot, window features, z-scores, drift state, quality flags, and reading context. No LLM is called here. |

---

## Overview

Drift is a continuous, real-valued score ∈ [0, 1) that estimates how far the user's current reading state has drifted from their calibrated attentional baseline. A drift of 0 means perfectly focused; a drift approaching 1 means severely inattentive.

This document describes **exactly** what is implemented as of the current codebase, including every constant, formula, weight, and research citation.

---

## 1. Mathematical Model — Hybrid Exponential Decay

Attention follows an exponential decay curve, originally described by Ebbinghaus (1885) for memory decay and adapted to sustained reading by Smallwood & Schooler (2006):

```
A(t)      = exp(−beta_ema × t_minutes)     ∈ (0, 1]
drift     = 1 − A(t)                        ∈ [0, 1)
```

**Properties guaranteed by this formulation:**

| Property | Explanation |
|---|---|
| `drift > 0` always | Because `exp(−x) < 1` for all `x > 0`. Drift is never stuck at zero. |
| Drift grows with time | Natural time-on-task fatigue: the longer you read, the more accumulated drift. |
| Drift can decrease | If `beta_ema` drops (user re-engages), the growth rate of `1 - exp(-beta*t)` decreases. At any point where `new_beta < prev_beta`, the drift curve has a lower slope. |
| Bounded to [0, 1) | Mathematical guarantee from exponential form. |

### Expected drift curves for a calibrated user

| Scenario | `beta_ema` | Drift at 1 min | Drift at 3 min | Drift at 10 min |
|---|---|---|---|---|
| Focused, below-baseline idle | ≈ 0.02 | 2% | 6% | 18% |
| Mild distraction (1.7× idle) | ≈ 0.12 | 11% | 30% | 70% |
| Heavy idle (no interaction) | ≈ 0.65 | 48% | 86% | ~100% |
| Tab away / window blur | ≈ 0.65 | 48% | 86% | ~100% |
| Sustained skimming (2× WPM) | ≈ 0.46 | 36% | 74% | ~99% |
| Stuck/confused (stagnation) | ≈ 0.27 | 24% | 55% | 93% |

---

## 2. Beta Computation — The Decay Rate

`beta_ema` is the key lever: it is recomputed every 2-second telemetry cycle and controls how fast drift grows.

```python
# Raw beta for this window:
beta_raw = BETA0
         + confidence × W_DISRUPT × disruption_score
         − confidence × W_ENGAGE  × engagement_score

# Clamp to safe range:
beta_effective = clamp(beta_raw, BETA_MIN, BETA_MAX)

# EMA-smooth for UI stability:
beta_ema = BETA_EMA_ALPHA × beta_effective
         + (1 − BETA_EMA_ALPHA) × prev_beta_ema
```

### Constants

| Constant | Value | Meaning |
|---|---|---|
| `BETA0` | `0.03` | Natural time-on-task decay at baseline attention |
| `BETA_MIN` | `0.02` | Floor: even perfect engagement cannot eliminate drift growth |
| `BETA_MAX` | `0.65` | Ceiling: maximum disruption |
| `BETA_EMA_ALPHA` | `0.20` | EMA smoothing for beta (≈50% convergence in 6 seconds) |
| `W_DISRUPT` | `0.70` | How much disruption_score can raise beta (max +0.70) |
| `W_ENGAGE` | `0.425` | How much engagement_score can lower beta (max −0.425) |

**Why BETA_EMA_ALPHA = 0.20?**
At 0.10 (previous value), it took ~45 seconds for beta to respond to strong distraction. At 0.20, 50% convergence happens in 3 update cycles (6 seconds), making the model responsive to rapid state changes while still preventing single-window spikes from dominating.

---

## 3. Telemetry Windowing

The model operates on a **rolling 30-second window** of `telemetry_batch` events stored in the `activity_events` TimescaleDB hypertable.

- Query: `SELECT created_at, payload FROM activity_events WHERE session_id=:id AND event_type='telemetry_batch' AND created_at >= now() - interval '30 seconds'`
- Each batch is one row, representing a 2-second window of aggregated interaction data
- Window contains ~15 batches at steady state (30s ÷ 2s = 15)

---

## 4. Feature Extraction

All features are extracted from the 30-second window batches by `services/api/app/services/drift/features.py`.

### 4.1 Scroll Behaviour

| Feature | Formula | Purpose |
|---|---|---|
| `scroll_velocity_norm_mean` | `mean(scroll_delta_abs_sum / (viewport_height_px × 2s))` | Normalised scroll speed — viewport-independent |
| `scroll_velocity_norm_std` | Standard deviation of above | Variability in scroll speed |
| `scroll_burstiness` | `std / max(mean, ε)` | Erratic bursting vs. steady scroll |
| `jitter_ratio` | `mean(scroll_direction_changes / max(scroll_event_count, 1))` | Oscillatory, restless scrolling |
| `regress_rate` | `mean(scroll_delta_neg_sum / max(pos+neg, ε))` | Fraction of scroll that is backward |

**Why normalise by viewport height?** A user with a small screen scrolls more frequently to cover the same content. Raw pixel deltas are screen-size-dependent; dividing by viewport height makes the metric content-progress-equivalent.

### 4.2 Idle and Pause Behaviour

| Feature | Formula | Purpose |
|---|---|---|
| `idle_ratio_mean` | `mean(clamp(idle_seconds / 2.0, 0, 1))` | Fraction of window time with no interaction |
| `long_pause_share` | fraction of batches with `scroll_pause_seconds ≥ pause_thresh` | Sustained pauses above personalised threshold |
| `stagnation_ratio` | fraction of window spent on the **same** `current_paragraph_id` | Stuck on one paragraph |

`pause_thresh` is personalised: `max(2.0, baseline_para_dwell_median_s / 3)`. A user whose median paragraph dwell is 15 seconds gets a pause threshold of 5 seconds; a user who reads fast gets the minimum threshold of 2 seconds.

### 4.3 Reading Pace (Content-Based)

Pace estimation uses actual paragraph word counts, not just scroll speed.

```
window_words_read_est  = Σ word_count(para_id) for paras dominant ≥ 1 batch
effective_read_seconds = Σ 2s for batches where:
                           focus_state == "focused"  AND
                           idle_seconds ≤ 1.5
window_wpm_effective   = (words_read / max(effective_seconds, ε)) × 60
pace_ratio             = window_wpm_effective / baseline_wpm_effective
pace_dev               = abs(log(max(pace_ratio, 0.001)))   ← symmetric: 2× fast = 0.5× slow
```

**Pace is only used if `pace_available = True`:**
- `effective_read_seconds ≥ 10` AND
- `paragraphs_observed ≥ 2`

Without at least 2 distinct paragraph transitions, pace estimation is too noisy (e.g., a user sitting on one long paragraph registers WPM = ∞ or 0).

### 4.4 Mouse and Focus

| Feature | Formula | Purpose |
|---|---|---|
| `mouse_efficiency` | `mean(clamp(mouse_net_px / max(mouse_path_px, ε), 0, 1))` | Deliberate directed mouse movement (vs. aimless wandering) |
| `focus_loss_rate` | fraction of batches where `window_focus_state ≠ "focused"` | Explicit context switching / tab away |

### 4.5 Progress Markers

When the user clicks "Load more" or advances a section, a `progress_marker` event is logged to `activity_events`. The feature `progress_markers_count` counts these in the last 30 seconds. This is the strongest single engagement signal (deliberate forward progress).

---

## 5. Personalised Z-Score Normalisation

All features are converted to **deviations from the user's own calibration baseline** using z-scores. The same idle_ratio = 0.80 means very different things for different users:

- User A (baseline: `idle_ratio_mean=0.20`) → z_idle = `(0.80 - 0.20) / 0.20 = 3.0` → extreme
- User B (baseline: `idle_ratio_mean=0.70`) → z_idle = `(0.80 - 0.70) / 0.15 = 0.67` → mild

### z_pos (one-sided, bad-high signals)

```python
z_pos(x, mu, sigma) = max(0, clamp((x - mu) / (sigma + 1e-5), 0, 3))
```

Used for: `idle_ratio`, `regress_rate`, `jitter`, `focus_loss_rate`, `stagnation_ratio`

### Pace z-score

```python
pace_scale = clamp(para_dwell_iqr_s / max(para_dwell_median_s, 1e-5), 0.15, 0.60)
z_pace     = clamp(pace_dev / pace_scale, 0, 3)
z_skim     = max(0, (pace_ratio - 1.0) / 0.5)     ← only fires when pace_ratio > SKIM_THRESHOLD=1.3
```

`pace_scale` is derived from the spread (IQR) of the user's paragraph dwell during calibration. A user with highly variable dwell times (large IQR) has a wider pace tolerance; they need to deviate further before triggering pace drift.

### Stagnation z-score

```python
stagnation_mu    = clamp(para_dwell_median_s / 30.0, 0.05, 0.80)
z_stagnation     = z_pos(stagnation_ratio, stagnation_mu, 0.15)
```

A user whose calibration shows median 10-second dwell per paragraph expects `stagnation_mu = 0.33` (one-third of the window on the same paragraph). Stagnating for the full window (`ratio=1.0`) gives `z = (1.0 - 0.33) / 0.15 = 4.47` → capped at 3.

### Fallback defaults (no calibration baseline)

| Field | Default | Rationale |
|---|---|---|
| `idle_ratio_mean` | `0.35` | Population average for reading sessions |
| `idle_ratio_std` | `0.20` | Typical window-to-window variability |
| `scroll_jitter_mean` | `0.10` | Low baseline direction-change rate |
| `regress_rate_mean` | `0.05` | 5% backward scroll in normal reading |

**Critical note:** The old codebase used `idle_ratio_mean = 0.05` as the fallback. This was too low: it made every reader appear to be in extreme idle relative to "baseline", so z_idle was always high and drift rose too fast for all users. The corrected default of `0.35` reflects actual measured idle rates.

---

## 6. Disruption Score — Research-Grounded Weights

```python
disruption_raw =
    W_D_IDLE        × z_idle          +   # 0.22
    W_D_FOCUS       × z_focus_loss    +   # 0.18
    W_D_STAGNATION  × z_stagnation    +   # 0.12
    W_D_PACE        × z_pace          +   # 0.15
    W_D_SKIM        × z_skim          +   # 0.18
    W_D_REGRESS     × w_regress_adj   +   # 0.10 (softened by baseline variability)
    W_D_JITTER      × w_jitter_adj    +   # 0.08 (softened by baseline variability)
    W_D_BURSTINESS  × z_burstiness        # 0.05

disruption_score = sigmoid((disruption_raw − 0.35) / 0.25)
```

### Why these weights? Research basis:

| Signal | Weight | Evidence |
|---|---|---|
| **Idle (mind-wandering)** | **0.22** | **Smallwood & Schooler (2006, Psych. Bull.):** sustained lack of interaction is the strongest behavioural predictor of stimulus-independent thought (mind-wandering). Unsworth & McMillan (2013) confirm scroll inactivity correlates with task-unrelated thought. **Idle weight is NEVER reduced by baseline variability — it's too important.** |
| **Focus loss (tab away)** | **0.18** | Direct explicit disengagement: the user has left the reading context entirely. Slightly lower than idle because brief accidental task-switches occur frequently. |
| **Stagnation** | **0.12** | Rayner (1998, Psych. Bull.): abnormally long fixation on a single region is a reliable indicator of comprehension difficulty or zoning out. |
| **Pace deviation (symmetric)** | **0.15** | Just & Carpenter (1980): reading rate tightly coupled to processing depth. Both under-paced (confused/stuck) and over-paced (skimming) indicate anomaly. |
| **Skimming (asymmetric)** | **0.18** | Receives a separate asymmetric signal because fast-forward reading can mask other signals (low idle, no focus loss) while still indicating poor encoding. |
| **Regress rate** | **0.10** | Rayner & Pollatsek (1989): high regressive saccade rate correlates with reading difficulty. Lower weight than idle because some backtracking is deliberate re-reading. |
| **Jitter** | **0.08** | Novel signal; erratic oscillatory scrolling associated with restlessness. Conservative weight due to absence of published norms. |
| **Scroll burstiness** | **0.05** | Secondary instability signal; supplements jitter. |

### Variability adjustment for jitter and regress only

Jitter and regress (not idle) use a softened variability penalty:

```python
w_adj = base_w / (1.0 + 0.5 × min(baseline_std / max(baseline_mean, 0.01), 2.0))
```

This reduces the weight by at most 50% for users with highly variable baseline jitter/regress. A user who naturally produces lots of back-and-forth scrolling is less penalised for high jitter than an unusually steady reader.

**Idle does NOT use this adjustment.** Regardless of how variable a user's idle pattern is, sustained idleness is always the primary attention signal.

### Sigmoid scale choice

- `CENTER = 0.35` means disruption_raw must exceed 0.35 to produce disruption_score > 0.50
- `SCALE = 0.25` provides a smooth transition region ±0.75 around the centre
- At disruption_raw = 0: `sigmoid((0 − 0.35) / 0.25) = sigmoid(−1.4) ≈ 0.20` (noise floor)
- At disruption_raw = 1.5 (heavy distraction): `sigmoid(4.6) ≈ 0.99`

---

## 7. Engagement Score — Multiplicative Design

```python
calm       = (1 − z_idle / 3) × (1 − z_focus_loss / 3)    ∈ [0, 1]

if pace_available:
    pace_align = 1 − max(z_pace, z_skim) / 3               ∈ [0, 1]
else:
    pace_align = 0.50                                        # neutral

progress_boost  = min(1.0, progress_markers_count / 2)
engagement      = calm × (0.80 × pace_align + 0.20 × progress_boost)
```

**Why multiplicative?** An additive design allows a user to score high engagement even while skimming (low pace_align) if they are merely "calm" (not idle, not blurred). The multiplicative structure forces BOTH calmness AND appropriate pace to be present simultaneously for high engagement. This ensures:

- Fast skimming → low pace_align → engagement drops → beta stays elevated
- Tab away → calm ≈ 0 → engagement ≈ 0 → beta stays elevated
- Progress marker without calmness → partial boost only

**Progress markers** receive a 20% bonus channel in the engagement formula. A progress marker is a strong signal: the user explicitly advanced the content, indicating deliberate forward reading progress.

---

## 8. Confidence Gating

Confidence prevents the model from making strong claims early in a session when only a few data points exist:

```python
confidence = min(1.0, n_batches_in_window / 15)
```

- After 6 seconds (3 batches): confidence = 0.20 → beta changes are heavily muted
- After 30 seconds (15 batches): confidence = 1.0 → full signal
- If `focusloss_share = 1.0` (user fully tabbed away): confidence overridden to `max(confidence, 0.8)` — we are certain about the disengagement

Confidence is applied to both W_DISRUPT and W_ENGAGE in the beta formula:

```python
beta_raw = BETA0
         + confidence × W_DISRUPT × disruption_score
         − confidence × W_ENGAGE  × engagement_score
```

---

## 9. Personalised Update Rates

The rate at which beta (and therefore drift) changes is personalised using baseline variability:

```python
var_factor = clamp(idle_ratio_std + scroll_jitter_std + regress_rate_std, 0.01, 0.50)
up_rate    = base_up_rate / (1 + 2 × var_factor)
down_rate  = base_down_rate / (1 + 1 × var_factor)
```

A highly variable user (large spread in calibration metrics) is less surprising in their deviations, so the model is less aggressive in raising beta. A very consistent user is more surprising when they deviate.

If a `progress_marker` occurred within the last 20 seconds: `down_rate *= 1.3` — deliberately enhanced recovery after deliberate forward progress.

---

## 10. Drift State Persistence

The computed drift state is stored in `session_drift_states` (PostgreSQL):

| Column | Type | Description |
|---|---|---|
| `session_id` | PK / FK | Links to session |
| `drift_level` | FLOAT | Raw `1 − exp(−beta_ema × t)` |
| `drift_ema` | FLOAT | EMA-smoothed drift for UI display |
| `disruption_score` | FLOAT | Snapshot of last disruption_score |
| `engagement_score` | FLOAT | Snapshot of last engagement_score |
| `confidence` | FLOAT | Data sufficiency [0, 1] |
| `beta_effective` | FLOAT | Instantaneous beta before EMA |
| `beta_ema` | FLOAT | Smoothed beta passed to next cycle |
| `pace_ratio` | FLOAT | WPM ratio vs. baseline (nullable) |
| `last_window_ends_at` | TIMESTAMPTZ | When this window ended |
| `updated_at` | TIMESTAMPTZ | Row last updated |

The state is upserted every time a new telemetry batch arrives (`POST /activity/batch`). If recomputation fails (e.g., DB error), the existing state is preserved — drift never resets to zero due to backend errors.

---

## 11. API Endpoints

### `GET /sessions/{id}/drift`

Returns the current drift state. If the state is stale (>10 seconds), recomputes from the latest 30-second window before returning.

```json
{
  "session_id": "...",
  "drift_level": 0.142,
  "drift_ema": 0.118,
  "disruption_score": 0.461,
  "engagement_score": 0.284,
  "confidence": 1.0,
  "beta_effective": 0.087,
  "beta_ema": 0.091,
  "pace_ratio": 1.12,
  "updated_at": "2026-02-25T10:30:00Z"
}
```

### `GET /sessions/{id}/drift/debug` (requires `DEBUG=true`)

Full explainability payload for the most recent window:

```json
{
  "session_id": "...",
  "user_id": "...",
  "baseline_used": true,
  "baseline_snapshot": { "wpm_effective": 180.0, "idle_ratio_mean": 0.35, ... },
  "window_stats": { "n_batches": 15, "window_start": "...", "window_end": "..." },
  "extracted_features": { "idle_ratio_mean": 0.41, "scroll_velocity_norm_mean": 0.23, ... },
  "z_scores": { "z_idle": 0.30, "z_focus_loss": 0.0, "z_pace": 0.72, ... },
  "beta_components": {
    "beta0": 0.03,
    "idle": 0.066,
    "focus_loss": 0.0,
    "stagnation": 0.043,
    "pace": 0.108,
    "skim": 0.0,
    ...
  },
  "beta_effective": 0.247,
  "beta_ema": 0.189,
  "t_minutes": 3.41,
  "drift_level": 0.473,
  "drift_ema": 0.392,
  "confidence": 1.0
}
```

---

## 12. Frontend Display

The `ReaderPage` top bar displays:

- **Drift: XX%** — from `drift_ema × 100`, colour-coded:
  - Green: `drift_ema < 0.30`
  - Yellow: `0.30 ≤ drift_ema < 0.60`
  - Red: `drift_ema ≥ 0.60`
- **Confidence: XX%** — from `confidence × 100`

In development mode (`import.meta.env.DEV`), a collapsible debug panel shows:
`pace_ratio`, `z_pace`, `z_idle`, `z_stagnation`, `z_regress`, `z_jitter`, `focusloss_share`, `disruption_score`, `engagement_score`, `baseline_used`.

Drift is polled from `GET /sessions/{id}/drift` every 3 seconds.

---

## 13. How to Pull and Analyse Session Data

This section explains how to extract the raw telemetry and drift data from any session for analysis.

### Option A: CSV Export (recommended for spreadsheet analysis)

Every session has a built-in CSV export endpoint:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/sessions/SESSION_ID/export.csv" \
  --output session_SESSION_ID.csv
```

The CSV contains one row per telemetry batch with columns:
`created_at`, `session_id`, `scroll_delta_sum`, `scroll_delta_abs_sum`, `scroll_delta_pos_sum`, `scroll_delta_neg_sum`, `scroll_event_count`, `scroll_direction_changes`, `scroll_pause_seconds`, `idle_seconds`, `mouse_path_px`, `mouse_net_px`, `window_focus_state`, `current_paragraph_id`, `current_chunk_index`, `viewport_progress_ratio`, `viewport_height_px`, `viewport_width_px`

The CSV file is also written to `services/api/exports/user_{user_id}/session_{session_id}.csv` on disk.

There is also an "Export CSV" button in the `ReaderPage` UI (visible in dev mode).

### Option B: Debug endpoint (real-time model explainability)

```bash
# Set DEBUG=true in services/api/.env first
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/sessions/SESSION_ID/drift/debug"
```

Returns the full feature extraction, z-scores, beta components, and drift values for the current 30-second window. Useful for understanding **why** drift is at a particular level right now.

### Option C: Direct database query

Connect to the PostgreSQL database:

```bash
psql postgresql://lockin:lockin@localhost:5432/lockin
```

#### All telemetry batches for a session (raw JSONB):

```sql
SELECT
  created_at,
  payload->>'scroll_delta_abs_sum'   AS scroll_abs,
  payload->>'idle_seconds'           AS idle_s,
  payload->>'window_focus_state'     AS focus,
  payload->>'current_paragraph_id'   AS para_id,
  payload->>'viewport_progress_ratio' AS progress
FROM activity_events
WHERE session_id = 'YOUR_SESSION_ID'
  AND event_type = 'telemetry_batch'
ORDER BY created_at ASC;
```

#### Drift state over time (requires `debug_blob` column if populated):

```sql
SELECT
  updated_at,
  drift_level,
  drift_ema,
  disruption_score,
  engagement_score,
  confidence,
  beta_effective,
  beta_ema,
  pace_ratio
FROM session_drift_states
WHERE session_id = 'YOUR_SESSION_ID';
```

Note: `session_drift_states` stores only the **latest** state (one row per session, upserted on each telemetry batch). To see drift trajectory over time, you need to either:
1. Log the debug endpoint response periodically, OR
2. Add a `session_drift_history` hypertable in a future phase (Phase 8 candidate)

#### Compute idle ratio manually across a session:

```sql
SELECT
  AVG((payload->>'idle_seconds')::float / 2.0) AS avg_idle_ratio,
  SUM((payload->>'idle_seconds')::float) AS total_idle_s,
  COUNT(*) AS total_batches,
  COUNT(*) * 2 AS total_duration_s
FROM activity_events
WHERE session_id = 'YOUR_SESSION_ID'
  AND event_type = 'telemetry_batch';
```

#### Compare multiple sessions for a user:

```sql
SELECT
  s.id AS session_id,
  s.started_at,
  sds.drift_ema,
  sds.disruption_score,
  sds.engagement_score,
  sds.beta_ema,
  sds.confidence
FROM sessions s
JOIN session_drift_states sds ON sds.session_id = s.id
WHERE s.user_id = 'YOUR_USER_ID'
ORDER BY s.started_at DESC;
```

### Option D: Python analysis script

```python
import asyncio
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://lockin:lockin@localhost:5432/lockin"

async def pull_session_data(session_id: str):
    engine = create_async_engine(DATABASE_URL)
    async with AsyncSession(engine) as db:
        rows = await db.execute(text("""
            SELECT created_at, payload
            FROM activity_events
            WHERE session_id = :sid
              AND event_type = 'telemetry_batch'
            ORDER BY created_at ASC
        """), {"sid": session_id})
        batches = rows.fetchall()

    for ts, payload in batches:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        print(f"{ts} | idle={data.get('idle_seconds', 0):.1f}s "
              f"| focus={data.get('window_focus_state', '?')} "
              f"| para={data.get('current_paragraph_id', '?')}")

asyncio.run(pull_session_data("YOUR_SESSION_ID"))
```

---

## 14. Data Quality Guardrails (v3)

Each telemetry batch ingested by `POST /activity/batch` is tagged at ingestion time with three quality flags stored inside the JSONB payload:

| Flag | Condition | Confidence penalty |
|---|---|---|
| `telemetry_fault` | `idle_seconds > 2.0` received (should never happen post-fix) | ×0.5 |
| `scroll_capture_fault` | `scroll_delta_abs_sum ≈ 0` AND `scroll_event_count == 0` | ×0.7 |
| `paragraph_missing_fault` | `current_paragraph_id` is `null` | ×0.7 |

These flags are used in `features.py` to compute `quality_confidence_mult ∈ [0,1]`, which then multiplies the `base_confidence` before the drift model uses it:

```
confidence = base_confidence × quality_confidence_mult
```

A session with heavily broken telemetry (e.g. old v2 sessions with cumulative idle) will have low confidence and thus low model aggressiveness — it will not produce spuriously high drift scores from garbage input.

The `/drift/debug` endpoint now returns all quality flags, and the `/state-packet` endpoint explicitly exposes `flags.baseline_valid`.

---

## 15. LLM-Ready State Packet (Phase 8 Preparation)

`GET /sessions/{id}/state-packet` returns the structured payload the future local LLM (Ollama) will consume. No LLM is called here; this endpoint is purely for preparation and inspection.

```json
{
  "session_id": 144,
  "user_id": 3,
  "computed_at": "2026-03-11T15:00:00Z",
  "baseline_snapshot": {
    "wpm_effective": 0,
    "idle_ratio_mean": 0.35,
    ...
  },
  "window_features": {
    "n_batches": 15,
    "idle_ratio_mean": 0.87,
    "scroll_velocity_norm_mean": 0.0,
    ...
  },
  "z_scores": {
    "z_idle": 2.8,
    "z_focus_loss": 0.0,
    ...
  },
  "drift": {
    "drift_ema": 0.67,
    "beta_ema": 0.52,
    "disruption_score": 0.81,
    "engagement_score": 0.09,
    "confidence": 0.45
  },
  "flags": {
    "baseline_valid": false,
    "baseline_wpm_valid": false,
    "telemetry_fault_rate": 0.95,
    "scroll_capture_fault_rate": 1.0,
    "paragraph_missing_fault_rate": 1.0,
    "quality_confidence_mult": 0.35
  },
  "context": {
    "progress_ratio": 0.58,
    "current_paragraph_id": null,
    "pace_ratio": null,
    "pace_available": false,
    "progress_velocity": 0.002
  }
}
```

The LLM will later map these signals to probabilistic state labels:
- **Focused**: low drift, low disruption, high engagement, good baseline match
- **Drifting**: rising drift, high idle/blur, poor pace alignment
- **Hyperfocused**: very fast pace, high scroll velocity, low idle (skimming risk)
- **Fatigued**: slow pace, increasing regress/jitter, high stagnation

---

## 16. Drift History Hypertable (v3)

`session_drift_history` is a Timescale hypertable that stores one row every ≈10 seconds per session (every 5th upsert of `session_drift_states`):

```sql
CREATE TABLE session_drift_history (
    id BIGSERIAL,
    session_id INT REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL,
    drift_ema FLOAT NOT NULL,
    beta_ema FLOAT NOT NULL,
    disruption_score FLOAT NOT NULL,
    engagement_score FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    pace_ratio FLOAT
);
-- Converted to hypertable on startup
```

Query full drift trajectory for a session:
```sql
SELECT created_at, drift_ema, disruption_score, engagement_score, confidence
FROM session_drift_history
WHERE session_id = 144
ORDER BY created_at ASC;
```

---

## 17. Limitations and Phase 8 Candidates

| Limitation | Planned Fix |
|---|---|
| ~~Drift history only has the latest snapshot~~ | ✅ Fixed: `session_drift_history` hypertable added |
| Baseline may have `wpm_effective = 0` if old calibration was run | Re-calibrate; `baseline_valid` flag now warns in all debug endpoints |
| No acceleration of drift under sustained idleness | Track `consecutive_high_idle_windows` across updates; apply geometric penalty |
| Baseline only from calibration text (static reading) | Expand calibration to include a scrolling task and a distraction task |
| Mouse efficiency has no baseline from calibration | Add mouse efficiency to calibration baseline computation |
| LLM classification of attentional state | Phase 8: Feed `/state-packet` into a local Ollama LLM for state labelling |

---

## 15. References

- Ebbinghaus, H. (1885). *Über das Gedächtnis* [Memory]. Duncker & Humblot.
- Just, M. A., & Carpenter, P. A. (1980). A theory of reading: from eye fixations to comprehension. *Psychological Review, 87*(4), 329–354.
- Rayner, K. (1998). Eye movements in reading and information processing: 20 years of research. *Psychological Bulletin, 124*(3), 372–422.
- Rayner, K., & Pollatsek, A. (1989). *The Psychology of Reading*. Prentice-Hall.
- Smallwood, J., & Schooler, J. W. (2006). The restless mind. *Psychological Bulletin, 132*(6), 946–958.
- Unsworth, N., & McMillan, B. D. (2013). Mind wandering and reading comprehension: Examining the roles of working memory capacity, interest, motivation, and topic experience. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 39*(3), 832–842.
