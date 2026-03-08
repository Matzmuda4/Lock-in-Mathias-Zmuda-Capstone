# Phase 6 — Calibration (Baseline Collection)

> **Status:** Implemented — all tests passing (33 backend, 45 frontend).
> Phase 7 (drift modelling) will consume the baseline stored here.

---

## Purpose

Calibration establishes a **personalised reading profile** for each user by
having them read a short, plain-text passage at their natural pace.  Every
two-second telemetry batch logged during that session is later aggregated into
a set of baseline metrics stored in `user_baselines.baseline_json`.

The resulting profile answers the key Phase 7 question:
> *"What does this person look like when they are reading normally?"*

Anomaly detection in drift scoring then fires when live session metrics deviate
meaningfully from those personal baselines.

---

## Calibration Text

- **File:** `callibration/callibration.txt` (repo root, note spelling)
- **Format:** Plain UTF-8 text, paragraphs separated by blank lines
- **Word count:** ~450 words (verified from DB chunk sum at completion time)
- **Paragraph count:** ~26 (depends on blank-line splitting)

Using a plain text file (not a PDF) ensures instantaneous chunk creation —
no Docling parse job is needed.

---

## Flow

```
GET  /calibration/status      → { has_baseline, calib_available, parse_status }
POST /calibration/start       → { session_id, document_id }       [creates chunks]
  [user reads at natural pace; telemetry batches stream to /activity/batch]
POST /calibration/complete    → { baseline, completed_at, session_id }
GET  /calibration/session/{id}→ { session, paragraphs, total_words }   [reader page]
GET  /calibration/baseline    → stored baseline_json
```

### Gating

On `HomePage` mount the frontend calls `GET /calibration/status`.  If
`has_baseline = false` and `calib_available = true` the user is redirected to
`/calibration` and cannot use normal sessions until calibration is finished.

---

## Baseline Metrics — v2 Schema

```jsonc
{
  // ── WPM ────────────────────────────────────────────────────────────────
  "wpm_gross": 180.0,
  "wpm_effective": 210.0,
  "words_read_estimated": 450,
  "effective_reading_seconds": 128.0,

  // ── Scroll velocity ────────────────────────────────────────────────────
  "scroll_velocity_px_s_mean": 60.5,
  "scroll_velocity_px_s_std":  18.2,
  "scroll_velocity_norm_mean": 0.076,    // dimensionless (per-viewport/s)
  "scroll_velocity_norm_std":  0.022,

  // ── Jitter ─────────────────────────────────────────────────────────────
  "scroll_jitter_mean": 0.12,
  "scroll_jitter_std":  0.05,

  // ── Idle ───────────────────────────────────────────────────────────────
  "idle_ratio_mean": 0.08,
  "idle_ratio_std":  0.04,
  "idle_seconds_mean": 0.16,
  "idle_seconds_std":  0.08,

  // ── Regress rate ───────────────────────────────────────────────────────
  "regress_rate_mean": 0.04,
  "regress_rate_std":  0.02,

  // ── Paragraph dwell distribution ────────────────────────────────────────
  "para_dwell_mean_s":   6.4,
  "para_dwell_median_s": 5.8,
  "para_dwell_iqr_s":    3.2,
  "paragraph_count_observed": 24,

  // ── Presentation profile ────────────────────────────────────────────────
  "presentation_profile": {
    "viewport_height_px_mean": 812.0,
    "viewport_height_px_std":   4.5,
    "viewport_width_px_mean":  1440.0,
    "viewport_width_px_std":    2.1,
    "reader_container_height_px_mean": 770.0,
    "calibration_text_word_count": 450,
    "paragraph_count_total": 26
  },

  "calibration_duration_seconds": 148,

  // ── Legacy aliases (v1; kept for backwards compatibility) ────────────────
  "wpm_mean": 180.0,
  "wpm_std": 0.0,
  "scroll_velocity_mean": 60.5,
  "scroll_velocity_std": 18.2,
  "paragraph_dwell_mean": 6.4,
  "regress_rate_mean_legacy": 0.12
}
```

---

## Metric Definitions & Rationale

### WPM — gross vs effective

| Metric | Definition | Why it matters |
|--------|-----------|----------------|
| `wpm_gross` | `total_words / total_duration_min` | Conservative baseline; easy to compute; degrades when user pauses or tabs away |
| `wpm_effective` | `words_read_estimated / effective_reading_min` | Only counts windows where `focus_state = "focused"` AND `idle_seconds ≤ 1.5 s`; a cleaner signal of actual reading speed |
| `words_read_estimated` | Sum of `word_count` for paragraphs observed for ≥ 1 batch (≥ 2 s) | Paragraph coverage during calibration |
| `effective_reading_seconds` | Sum of qualifying 2-second window durations | Feeds Phase 7 expected-dwell calculations |

**Gross WPM is insufficient** because a user who pauses to think, switches tabs,
or takes a sip of water accumulates idle time in the denominator, producing an
artificially low speed.  `wpm_effective` is what the drift model should compare
against in real sessions.

### Scroll Velocity — raw vs normalised

Raw `scroll_velocity_px_s_mean` (px/s) is kept for debugging, but **it is not
directly comparable across machines** with different screen DPI or window sizes.
A user on a 1080p display will have the same viewport height in CSS pixels as one
on a 4K display, but a user with a maximised 27-inch window vs a half-screen
laptop window will scroll more frequently.

Normalised velocity solves this:

```
scroll_velocity_norm = scroll_delta_abs_sum / (viewport_height_px × window_seconds)
```

Unit: *fraction of viewport per second*.  A value of `0.08` means the user
scrolled ~8 % of the reader height per second on average.  This is
**window-size invariant** and directly comparable across sessions regardless of
screen resolution.

Phase 7 will flag drift when a live session's `scroll_velocity_norm` deviates
more than `k × scroll_velocity_norm_std` from the calibration baseline.

### Paragraph Dwell — median & IQR over mean

The paragraph dwell distribution is typically right-skewed: a few complex
paragraphs attract very long dwell times while most are read quickly.  The
**mean is pulled upward** by those outliers.

Using `para_dwell_median_s` and `para_dwell_iqr_s` (Q3 − Q1) gives a robust
centre and spread:

```
expected_dwell_s(chunk) = chunk.meta.word_count / (wpm_effective / 60)
anomaly_ratio = actual_dwell_s / expected_dwell_s
```

Phase 7 will flag paragraphs where `anomaly_ratio` falls outside a band defined
by `para_dwell_iqr_s`.

### Regress Rate — true scroll backtracking

`scroll_jitter_mean` (direction changes / total events) is a proxy for
instability but conflates genuine re-reading with micro-corrections.

`regress_rate_mean` gives a cleaner signal:

```
regress_rate_window = scroll_delta_neg_sum / (scroll_delta_neg_sum + scroll_delta_pos_sum + ε)
```

where `scroll_delta_pos_sum` = sum of downward deltas and
`scroll_delta_neg_sum` = sum of `|upward deltas|` in each 2-second window.

A high regress rate (e.g. > 0.15) indicates the user frequently scrolled back
— a strong predictor of confusion or attention loss.

### Idle — ratio and seconds

`idle_ratio_mean` (idle_s / 2.0, clamped [0,1]) is the primary signal. The
`idle_ratio_std` quantifies variability: a user with low mean but high std
alternates between focused reading and zoning out.

Phase 7 also needs the raw `idle_seconds_mean/std` to compute expected idle
budgets for content types:

```
figure_budget_seconds = 60 × (1 + idle_ratio_mean)
table_budget_seconds  = 90 × (1 + idle_ratio_mean)
```

The `+1` ensures the budget is never zero — a user with `idle_ratio_mean = 0`
still gets the full 60 s / 90 s default.  A user who habitually pauses gets a
proportionally larger budget, so the system does not fire false drift alerts
while they are studying a diagram.

### Presentation Profile

Stores the distribution of viewport dimensions observed during calibration.
Phase 7 uses this to:
1. Check if a live session runs at a **significantly different window size**
   and apply normalisation correction.
2. Warn the user if calibration was performed on a fundamentally different
   display configuration (e.g. phone vs desktop).

---

## Extrapolation to Longer Documents

Because the calibration text is plain prose with no images or tables, all
metrics represent the user's **baseline reading speed for text-only content**.
For longer, richer documents Phase 7 applies these extrapolation rules:

| Content type | Expected idle budget | Personalisation |
|--------------|---------------------|-----------------|
| Plain text paragraph | `word_count / (wpm_effective / 60)` seconds | Direct WPM scaling |
| Figure / diagram | `60 s × (1 + idle_ratio_mean)` | User's idle habit scales the fixed budget |
| Table | `90 s × (1 + idle_ratio_mean)` | Same |

Paragraph-level dwell anomaly detection in Phase 7 uses a **dimensionless
`dwell_ratio`** rather than bounds expressed in seconds, which is more robust
to documents with paragraphs of very different lengths:

```
expected_dwell_s = chunk.meta.word_count / (wpm_effective / 60)
dwell_ratio      = actual_dwell_s / expected_dwell_s
```

Phase 7 will compare `dwell_ratio` against the baseline distribution
(`para_dwell_median_s` / `para_dwell_iqr_s` converted to ratio space):

```
# k1 = 1.0, k2 = 2.0 — placeholder coefficients to be tuned in Phase 7
lower_bound = baseline_dwell_ratio_median - k1 × baseline_dwell_ratio_iqr
upper_bound = baseline_dwell_ratio_median + k2 × baseline_dwell_ratio_iqr
```

A `dwell_ratio` outside `[lower_bound, upper_bound]` triggers a soft drift
signal.  Because the ratio is dimensionless, the same thresholds apply
regardless of paragraph length or document type.

---

## Implementation

### Files modified

| File | Change |
|------|--------|
| `apps/desktop/src/hooks/useTelemetry.ts` | Added `scroll_delta_pos_sum`, `scroll_delta_neg_sum`, `viewport_height_px`, `viewport_width_px`, `reader_container_height_px` to each batch |
| `apps/desktop/src/services/activityService.ts` | Updated `TelemetryBatch` interface with new fields |
| `services/api/app/schemas/activity.py` | Added optional fields to `ActivityBatchCreate` |
| `services/api/app/schemas/calibration.py` | Rewrote `BaselineData` for v2 schema; legacy fields kept |
| `services/api/app/services/calibration/baseline.py` | Full rewrite with new pure functions |
| `services/api/app/routers/calibration.py` | `calibration_complete` now passes `paragraph_word_counts`, `calibration_text_word_count`, `paragraph_count_total` to `compute_baseline` |
| `services/api/app/routers/sessions.py` | CSV `_CSV_FIELDS` updated with new columns |

### New pure functions in `baseline.py`

| Function | Description |
|----------|-------------|
| `scroll_velocities_norm(batches, fallback_vp)` | Per-batch viewport-normalised velocity |
| `compute_effective_wpm(batches, para_wc, total_words_override)` | Computes gross + effective WPM |
| `compute_scroll_velocity_stats(batches)` | Mean/std for raw px/s and normalised velocity |
| `compute_paragraph_dwell_distribution(batches)` | Mean, median, IQR, count |
| `compute_regress_rate_stats(batches)` | Mean/std from signed scroll sums |
| `compute_idle_stats(batches)` | Mean/std for ratio and raw seconds |
| `compute_jitter_stats(batches)` | Mean/std for direction-change ratio |
| `compute_presentation_profile_stats(batches, word_count, para_count)` | Viewport dimension stats |

---

## Testing

### Backend (`services/api/tests/test_calibration.py`) — 33 tests

| Class | Tests |
|-------|-------|
| `TestBaselineComputation` | 8 unit tests for existing helper functions |
| `TestBaselineComputationV2` | 11 new unit tests covering all v2 metric functions |
| `TestCalibrationStatus` | 2 integration tests |
| `TestCalibrationStart` | 3 integration tests |
| `TestCalibrationComplete` | 4 integration tests |
| `TestExportCsvV2` | 1 integration test verifying new CSV columns |
| `TestExportCsv` | 3 integration tests for original export behaviour |

Run with:
```bash
cd services/api
.venv/bin/python -m pytest tests/test_calibration.py -v
```

### Frontend (`apps/desktop/src/test/telemetry.test.ts`) — 23 tests

Includes 6 new tests for `computeSignedScrollSums` and 4 for
`normalizedScrollVelocity` alongside the existing 13.

Run with:
```bash
cd apps/desktop
pnpm test
```

---

## Manual Verification

After starting the application:

1. Register a new user → redirected to `/calibration`.
2. Click **Start Calibration** → reader page loads immediately (no spinner).
3. Read the text at normal pace — scroll continuously.
4. Click **Done** → baseline summary appears.  Verify:
   - `wpm_effective > 0`
   - `scroll_velocity_norm_mean > 0`
   - `regress_rate_mean` between 0 and 0.3 for normal reading
   - `presentation_profile.calibration_text_word_count ≈ 450`
5. Navigate to dashboard and start a normal session.
6. Use **Export CSV** (dev tool on reader page) and verify the header row
   contains `scroll_delta_pos_sum`, `scroll_delta_neg_sum`,
   `viewport_height_px`, `viewport_width_px`, `reader_container_height_px`.
7. Call `GET /calibration/baseline` and confirm the full v2 JSON shape.
