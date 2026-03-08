# Phase 7 — Personalised Mathematical Drift Modelling

> **Status:** Implemented — 147 backend tests, 52 frontend tests, all passing.
> No LLM, no interventions, no UI panels beyond the top-bar drift meter.

---

## What "Drift" Means

Drift is a continuous, dimensionless score in **[0, 1)** that estimates how
much a reader's attention has decayed relative to their baseline.

- **0** → fully attentive (no decay from starting state).
- **→ 1** → highly drifted (attention approaching zero).

Drift is **not a diagnosis**. It is a mathematical signal derived from
observable reading-behaviour signals: scroll patterns, idle time, focus
switches, and reading pace. It is personalised to each user's calibration
baseline so that the same behaviour means different things for different people.

---

## Architecture (SOLID)

```
services/api/app/services/drift/
  types.py       — WindowFeatures, ZScores, DriftResult (pure dataclasses)
  features.py    — pure feature extraction from batch payloads
  model.py       — pure maths: normalization, beta, attention, drift, EMA
  windowing.py   — async SQL query for the last 30 s of telemetry
  store.py       — DB upsert / fetch for session_drift_states
```

Routers hold no math. Services hold no HTTP. Everything is independently testable.

---

## Rolling Window

Every 2-second telemetry batch triggers a drift recompute using the **last
30 seconds of batches** for that session (up to ~15 batches).

```sql
SELECT payload FROM activity_events
WHERE session_id = :sid
  AND event_type = 'telemetry_batch'
  AND created_at >= NOW() - INTERVAL '30 seconds'
ORDER BY created_at ASC
```

---

## Feature Extraction (`features.py`)

The following features are computed per batch then aggregated over the window:

| Feature | Computation | Scale |
|---------|------------|-------|
| `scroll_velocity_norm_mean` | `scroll_delta_abs_sum / (viewport_height_px × 2.0)` | dimensionless/s |
| `scroll_jitter_mean` | `direction_changes / max(event_count, 1)` | [0, 1] |
| `regress_rate_mean` | `neg_sum / (neg_sum + pos_sum + ε)` | [0, 1] |
| `idle_ratio_mean` | `idle_seconds / 2.0`, clamped [0, 1] | [0, 1] |
| `scroll_pause_mean` | mean `scroll_pause_seconds`, capped 10 s | seconds |
| `focus_loss_rate` | proportion of batches with `focus_state != "focused"` | [0, 1] |
| `mouse_efficiency_mean` | `net_px / max(path_px, ε)`, clamped [0, 1] | [0, 1] |
| `paragraph_stagnation` | max single-paragraph dominance fraction (≥ 0.5 else 0) | [0, 1] |
| `pace_ratio` | `window_wpm_effective / baseline_wpm_effective` | ratio |
| `pace_dev` | `|log(pace_ratio)|` — symmetric, 0 when pace_ratio = 1 | nats |

### Pace Estimation

Pace detection addresses **both too fast (skimming) and too slow (stuck)**:

```
window_wpm_effective = (words_read_in_window / effective_seconds) × 60
pace_ratio           = window_wpm_effective / baseline_wpm_effective
pace_dev             = |log(pace_ratio)|
```

`log` is symmetric around 1: `log(0.5) = log(2)`, so a 2× overspeed and a
0.5× underspeed are equally deviant.

---

## Baseline Normalisation (`model.py`)

Each feature is z-scored against the user's calibration baseline:

```
z = (x - μ) / (σ + ε),   clamped to [-3, +3]
```

For signals where **high = bad** (idle, regress, jitter, focus loss, pause,
stagnation) we floor at zero: `z_pos = max(0, z)`.

For mouse efficiency, we invert: `z_mouse = z_pos(baseline_eff - actual_eff)`.

For pace: `z_pace = clamp(pace_dev / scale, 0, 3)` where `scale` is derived
from `scroll_velocity_norm_std` (a proxy for reading variability).

Users without a completed calibration fall back to sensible population defaults.

---

## Beta Effective

```
beta = beta0                          # 0.06 — baseline decay without any signals
     + w_idle       × z_idle          # 0.30
     + w_focus_loss × z_focus_loss    # 0.35
     + w_jitter     × z_jitter        # 0.18
     + w_regress    × z_regress       # 0.12
     + w_pause      × z_pause         # 0.10
     + w_stagnation × z_stagnation    # 0.18
     + w_mouse      × z_mouse         # 0.10
     + w_pace       × z_pace          # 0.25

beta = clamp(beta, 0.02, 1.50)
```

Weights are documented constants in `model.py`; they can be tuned once more
session data accumulates.

---

## Attention and Drift Score

```
A(t) = exp(-beta × t_minutes)      # [0, 1]; 1 at t=0
drift_score = 1 - A(t)             # [0, 1)
```

`t_minutes` is the wall-clock elapsed time from `session.started_at`.

**Drift EMA** smooths the score to reduce jitter:

```
drift_ema = alpha × drift_score + (1 - alpha) × prev_ema
alpha = 0.25
```

---

## Confidence Score

```
confidence = min(1.0, n_batches_in_window / 15)
```

15 batches = a full 30-second window at 2 s/batch.  In the first few seconds
of a session confidence is low and the drift display is less meaningful.

---

## DB Schema

```sql
CREATE TABLE session_drift_states (
    session_id       INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    beta_effective   FLOAT   NOT NULL DEFAULT 0.06,
    attention_score  FLOAT   NOT NULL DEFAULT 1.0,
    drift_score      FLOAT   NOT NULL DEFAULT 0.0,
    drift_ema        FLOAT   NOT NULL DEFAULT 0.0,
    confidence       FLOAT   NOT NULL DEFAULT 0.0,
    last_window_ends_at TIMESTAMPTZ,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/sessions/{id}/drift` | Current drift state; recomputes if stale (> 10 s) |
| `GET` | `/sessions/{id}/drift/debug` | Features, z-scores, baseline excerpt — only when `DEBUG=true` |

Both endpoints enforce session ownership (404 for wrong user).

### Automatic recompute on batch ingestion

`POST /activity/batch` now recomputes drift immediately after inserting the
telemetry row.  A failure in drift computation does **not** fail the batch
insert (wrapped in try/except).

---

## Frontend (`ReaderPage.tsx`)

The top bar shows a **Drift Meter** next to the timer:

```
Drift  [35%]          ← yellow pill for 30–60 %
```

Colour thresholds:
- **Green** `< 30 %` — on track
- **Yellow** `30–60 %` — mild drift
- **Red** `> 60 %` — significant drift

In dev mode, a confidence percentage is also shown.

The drift state is polled every **3 seconds** while the session is active
via `driftService.getDrift()`.

---

## Testing

### Backend (`tests/test_drift.py`) — 30 tests

| Class | Tests |
|-------|-------|
| `TestDriftModel` | 11 pure-math tests: symmetry, monotonicity, clamping, EMA, confidence |
| `TestDriftFeatures` | 10 pure-feature tests: velocity norm, jitter, regress, efficiency, stagnation |
| `TestDriftIntegration` | 9 integration tests: batch→state, endpoint, ownership, calibration session |

### Frontend (`src/test/drift.test.ts`) — 7 tests

| Suite | Tests |
|-------|-------|
| `driftColor` | 5 colour threshold tests |
| `driftPct` | 2 rounding tests |

---

## Manual Verification

Start a session and use the reader with these scenarios to observe drift changes:

| Scenario | Expected behaviour |
|----------|--------------------|
| Smooth continuous scrolling | Low drift (green), beta close to 0.06 |
| Stop scrolling / idle 20+ s | idle_ratio_mean rises → z_idle positive → beta increases → drift rises |
| Blur the window 5+ times | focus_loss_rate → 1.0 → large z_focus_loss → drift rises quickly |
| Rapid skimming (fast scroll) | pace_ratio >> 1 → pace_dev high → z_pace rises → drift rises |
| Read very slowly / stagnate | pace_ratio << 1 → pace_dev high → drift rises (same as skimming) |
| Frequently scroll back up | regress_rate_mean → 0.3–0.5 → z_regress positive → drift rises |

Debug endpoint (requires `DEBUG=true` in environment):
```
GET /sessions/{id}/drift/debug
```
Shows `features`, `z_scores`, `baseline_used`, `pace_ratio`, `window_wpm_effective`.
