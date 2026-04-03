#!/usr/bin/env python3
"""
generate_synthetic.py
─────────────────────
Append new synthetic training sessions to unlabelled.jsonl and labelled.jsonl.

Duplicate protection is session-level: if any packet with (user_id, session_id)
already exists in unlabelled.jsonl the entire session is skipped, so re-running
is always safe.

After running this script, run merge_supervised.py to update supervised.jsonl.

Usage (from TrainingData/ directory):
    python generate_synthetic.py

To add more sessions in the future, add entries to NEW_SESSIONS at the bottom.
"""

import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 42
_rng = random.Random(SEED)

BASE          = Path(__file__).parent
UNLABELLED    = BASE / "unlabelled.jsonl"
LABELLED      = BASE / "labelled.jsonl"
BASELINES_DIR = BASE / "baselines"

UTC = timezone.utc

# ── Constants (must match services/api/app/services/drift/model.py) ────────────
Z_POS_CAP       = 3.0
SKIM_THRESHOLD  = 1.6
SKIM_SCALE      = 0.5
_SKIM_IDLE_LO   = 0.40
_SKIM_IDLE_HI   = 0.80
_SKIM_STAG_LO   = 0.65
_SKIM_STAG_HI   = 0.85
W_D_IDLE        = 0.22
W_D_FOCUS       = 0.18
W_D_STAGNATION  = 0.12
W_D_SKIM        = 0.18
W_D_REGRESS     = 0.10
W_D_JITTER      = 0.08
W_D_BURSTINESS  = 0.05
DISRUPT_CENTER  = 0.40
DISRUPT_SCALE   = 0.25


# ── Core computation helpers ────────────────────────────────────────────────────

def _z_pos(v: float, mu: float, std: float, cap: float = Z_POS_CAP) -> float:
    return min(cap, max(0.0, (v - mu) / (std + 1e-5)))


def _compute_z_scores(f: dict, b: dict) -> dict:
    z_idle       = _z_pos(f["idle_ratio_mean"],
                          b["idle_ratio_mean"],
                          max(b.get("idle_ratio_std", 0.10), 0.10))
    z_focus_loss = _z_pos(f["focus_loss_rate"], 0.0, 0.08)
    z_jitter     = _z_pos(f["scroll_jitter_mean"],
                          b.get("scroll_jitter_mean", 0.10),
                          max(b.get("scroll_jitter_std", 0.10), 0.05))
    z_regress    = _z_pos(f["regress_rate_mean"],
                          max(b.get("regress_rate_mean", 0.05), 0.05),
                          max(b.get("regress_rate_std",  0.06), 0.06))
    z_pause      = _z_pos(f["scroll_pause_mean"],
                          b.get("idle_seconds_mean", 1.0),
                          max(b.get("idle_seconds_std", 1.0), 0.5))
    para_median  = b.get("para_dwell_median_s", 10.0)
    para_iqr     = b.get("para_dwell_iqr_s", 3.0)
    stag_mu      = min(max((para_median + 0.5 * para_iqr) / 30.0, 0.08), 0.80)
    z_stagnation = _z_pos(f["stagnation_ratio"], stag_mu, 0.15)
    z_mouse      = (0.0 if f["mouse_path_px_mean"] < 10.0
                    else _z_pos(0.70 - f["mouse_efficiency_mean"], 0.0, 0.15))
    if not f["pace_available"]:
        z_pace = 0.0
    else:
        pace_scale = min(max(para_iqr / max(para_median, 1e-5), 0.15), 0.60)
        z_pace = min(Z_POS_CAP, f.get("pace_dev", 0.0) / (pace_scale + 1e-5))
    idle_damp = 1.0 - min(1.0, max(0.0,
        (f["idle_ratio_mean"] - _SKIM_IDLE_LO) / (_SKIM_IDLE_HI - _SKIM_IDLE_LO)))
    stag_damp = 1.0 - min(1.0, max(0.0,
        (f["stagnation_ratio"] - _SKIM_STAG_LO) / (_SKIM_STAG_HI - _SKIM_STAG_LO)))
    if f["pace_available"] and f["pace_ratio"] > SKIM_THRESHOLD:
        z_skim_raw = min(Z_POS_CAP, (f["pace_ratio"] - 1.0) / SKIM_SCALE)
        z_skim = z_skim_raw * idle_damp * stag_damp
    else:
        z_skim = 0.0
    # Scroll-velocity fallback: catches excessive speed in any direction when
    # pace is unavailable.  Uses a gentler scale (2× SKIM_SCALE) so the signal
    # rises gradually; same idle + stagnation gates apply to exclude acute
    # single-paragraph thrashing (stag_damp→0).
    if not f["pace_available"]:
        bl_sv    = b.get("scroll_velocity_norm_mean", 0.010)
        sv_ratio = f["scroll_velocity_norm_mean"] / max(bl_sv, 1e-5)
        if sv_ratio > SKIM_THRESHOLD:
            sv_z_raw = min(Z_POS_CAP, (sv_ratio - 1.0) / (SKIM_SCALE * 2.0))
            z_skim   = max(z_skim, sv_z_raw * idle_damp * stag_damp)
    z_burstiness = _z_pos(f["scroll_burstiness"], 1.0, 0.5)
    return dict(
        z_idle=round(z_idle, 6), z_pace=round(z_pace, 6),
        z_skim=round(z_skim, 6), z_mouse=round(z_mouse, 6),
        z_pause=round(z_pause, 6), z_jitter=round(z_jitter, 6),
        z_regress=round(z_regress, 6), z_burstiness=round(z_burstiness, 6),
        z_focus_loss=round(z_focus_loss, 6), z_stagnation=round(z_stagnation, 6),
    )


def _compute_disruption(z: dict, b: dict) -> tuple:
    jm  = max(abs(b.get("scroll_jitter_mean",  0.01)), 0.01)
    js  = b.get("scroll_jitter_std", 0.10)
    rm  = max(abs(b.get("regress_rate_mean", 0.01)), 0.01)
    rs  = b.get("regress_rate_std", 0.06)
    w_j = W_D_JITTER  / (1.0 + 0.5 * min(js / jm, 2.0))
    w_r = W_D_REGRESS / (1.0 + 0.5 * min(rs / rm, 2.0))
    raw = (W_D_IDLE       * z["z_idle"]
         + W_D_FOCUS      * z["z_focus_loss"]
         + W_D_STAGNATION * z["z_stagnation"]
         + w_r            * z["z_regress"]
         + w_j            * z["z_jitter"]
         + W_D_SKIM       * z["z_skim"]
         + W_D_BURSTINESS * z["z_burstiness"])
    score = 1.0 / (1.0 + math.exp(-(raw - DISRUPT_CENTER) / DISRUPT_SCALE))
    return round(score, 6), round(raw, 6)


def _compute_engagement(z: dict, f: dict) -> float:
    calm = ((1.0 - min(z["z_idle"]       / Z_POS_CAP, 1.0))
          * (1.0 - min(z["z_focus_loss"] / Z_POS_CAP, 1.0)))
    if f["pace_available"]:
        pa = 1.0 - min(z["z_skim"] / Z_POS_CAP, 1.0)
        # Regression drag: subtle per-batch backward reading reduces engagement
        # even when the pace gate passes.
        regress_drag = min(z["z_regress"] / Z_POS_CAP, 1.0) * 0.25
        pa = max(0.0, pa - regress_drag)
    else:
        if f["scroll_velocity_norm_mean"] > 0.005:
            pa = 0.65 - min(z["z_regress"] / Z_POS_CAP, 1.0) * 0.30
        else:
            pa = 0.50
        # Velocity fallback drag: frantic scroll speed also suppresses engagement.
        if z["z_skim"] > 0.0:
            skim_drag = min(z["z_skim"] / Z_POS_CAP, 1.0) * 0.15
            pa = max(0.0, pa - skim_drag)
    pb = min(1.0, f.get("progress_markers_count", 0) / 2.0)
    return round(min(1.0, max(0.0, calm * (0.80 * pa + 0.20 * pb))), 6)


# ── Label + note generators ─────────────────────────────────────────────────────

def _labels_from_z(z: dict, target: str, confidence: float) -> tuple:
    """Derive soft labels from z-scores, guided by the target state."""
    zs = z["z_skim"];  zi = z["z_idle"];  zr = z["z_regress"]
    zst = z["z_stagnation"];  zf = z["z_focus_loss"];  zb = z["z_burstiness"]

    if target == "hyperfocused":
        h  = min(90, int(42 + zs * 20))
        c  = max(0, int(zst * 4))
        d  = max(0, int(zi * 3))
        f  = max(5, 100 - h - c - d)
        h  = 100 - f - d - c
    elif target == "cognitive_overload":
        sig = max(zr * 0.7, zst * 0.7)
        c  = min(88, int(30 + sig * 22))
        d  = max(5, int(8 + zi * 4 + zf * 4))
        f  = max(5, int(20 - sig * 3))
        h  = 0
        c  = 100 - f - d - h
    elif target == "drifting":
        ds = max(zi * 0.6, zf * 0.8, zb * 0.5)
        d  = min(85, int(35 + ds * 25))
        f  = max(8, int(22 - ds * 4))
        c  = max(5, int(5 + zst * 4 + zr * 3))
        h  = 0
        d  = 100 - f - c - h
    else:  # focused
        mix = max(zi, zf, zb * 0.3)
        f  = min(80, int(45 + (1.0 - min(mix, 2.0) / 2.0) * 25))
        d  = max(5, int(10 + zi * 7 + zf * 8))
        c  = max(3, int(5 + zst * 5 + zr * 4))
        h  = max(0, min(12, int(zs * 6)))
        f  = 100 - d - c - h

    labels = {k: max(0, v) for k, v in
              zip(["focused", "drifting", "hyperfocused", "cognitive_overload"],
                  [f, d, h, c])}

    # Widen distribution for early/incomplete windows
    if confidence < 0.8:
        unc = int((1.0 - confidence) * 18)
        pk  = max(labels, key=labels.get)
        for k in labels:
            labels[k] = (max(25, labels[k] - unc)
                         if k == pk else labels[k] + unc // 3)

    # Normalise to exactly 100
    total = sum(labels.values())
    if total > 0:
        labels = {k: max(0, round(v * 100 / total)) for k, v in labels.items()}
    diff = 100 - sum(labels.values())
    pk = max(labels, key=labels.get)
    labels[pk] += diff
    primary = max(labels, key=labels.get)
    return labels, primary


def _make_note(z: dict, f: dict, confidence: float) -> str:
    parts = []
    nb = f["n_batches"]
    if confidence < 0.7:
        parts.append(
            f"Early/incomplete window (n_batches={nb}, confidence={confidence:.2f}); "
            "labels conservative.")
    if f["pace_available"]:
        pr = f["pace_ratio"]; zs = z["z_skim"]
        if zs >= 1.2:
            if f["idle_ratio_mean"] >= _SKIM_IDLE_LO:
                parts.append(
                    f"pace_ratio={pr:.2f} fast but z_skim={zs:.2f} DAMPENED by "
                    f"idle_ratio_mean={f['idle_ratio_mean']:.2f} (threshold {_SKIM_IDLE_LO}).")
            elif f["stagnation_ratio"] >= _SKIM_STAG_LO:
                parts.append(
                    f"pace_ratio={pr:.2f} fast but z_skim={zs:.2f} DAMPENED by "
                    f"stagnation_ratio={f['stagnation_ratio']:.2f}.")
            else:
                parts.append(
                    f"pace_ratio={pr:.2f} (>{SKIM_THRESHOLD}×) triggers undampened "
                    f"z_skim={zs:.2f}; primary hyperfocus signal "
                    f"(idle={f['idle_ratio_mean']:.2f}, stag={f['stagnation_ratio']:.2f}).")
        else:
            parts.append(
                f"pace_available=True, pace_ratio={pr:.2f} — below skim threshold "
                f"({SKIM_THRESHOLD}×), z_skim=0.")
    else:
        if z["z_skim"] > 0.0:
            sv = f.get("scroll_velocity_norm_mean", 0.0)
            parts.append(
                f"pace_available=False; scroll_velocity_norm_mean={sv:.4f} exceeds "
                f"skim threshold → velocity-fallback z_skim={z['z_skim']:.2f} "
                "(frantic rapid non-linear scrolling detected).")
        else:
            parts.append("pace_available=False; z_skim=0 regardless of scroll speed.")
    if z["z_idle"] > 0.15:
        parts.append(
            f"z_idle={z['z_idle']:.2f} (idle_ratio_mean={f['idle_ratio_mean']:.2f} "
            "above user baseline).")
    if z["z_focus_loss"] > 0.3:
        parts.append(
            f"z_focus_loss={z['z_focus_loss']:.2f} (focus_loss_rate={f['focus_loss_rate']:.2f}): "
            "window-blur/tab-switch detected.")
    if z["z_stagnation"] > 0.5:
        parts.append(
            f"z_stagnation={z['z_stagnation']:.2f} (stagnation_ratio={f['stagnation_ratio']:.2f}): "
            "elevated dwell on single paragraph.")
    if z["z_regress"] > 0.5:
        parts.append(
            f"z_regress={z['z_regress']:.2f} (regress_rate_mean={f['regress_rate_mean']:.2f}): "
            "backward-scroll re-reading detected.")
    if z["z_burstiness"] > 1.0:
        parts.append(f"z_burstiness={z['z_burstiness']:.2f}: erratic scroll rhythm.")
    if z["z_pause"] > 1.0:
        parts.append(
            f"z_pause={z['z_pause']:.2f} (scroll_pause_mean={f['scroll_pause_mean']:.2f}s): "
            "long inter-scroll pauses.")
    return " ".join(parts) if parts else "Signals within normal bounds; no dominant attentional indicator."


# ── Packet builder ──────────────────────────────────────────────────────────────

def _build_packet(user_id, session_id, doc_id, seq,
                  session_start, baseline, baseline_ref, p):
    """Build a complete unlabelled packet from feature parameter dict `p`."""
    window_end   = session_start + timedelta(seconds=30 + seq * 10)
    window_start = window_end - timedelta(seconds=30)
    created_at   = window_end + timedelta(milliseconds=_rng.randint(50, 400))

    f = {
        "pace_dev":                     0.0,
        "n_batches":                    p["n_batches"],
        "pace_ratio":                   p["pace_ratio"],
        "pace_available":               p["pace_available"],
        "focus_loss_rate":              p["focus_loss_rate"],
        "idle_ratio_mean":              p["idle_ratio_mean"],
        "long_pause_share":             0.0,
        "stagnation_ratio":             p["stagnation_ratio"],
        "progress_velocity":            p["progress_velocity"],
        "regress_rate_mean":            p["regress_rate_mean"],
        "scroll_burstiness":            p["scroll_burstiness"],
        "scroll_pause_mean":            p["scroll_pause_mean"],
        "at_end_of_document":           False,
        "mouse_path_px_mean":           p.get("mouse_path_px_mean", 0.0),
        "scroll_jitter_mean":           p.get("scroll_jitter_mean",
                                            baseline.get("scroll_jitter_mean", 0.02)),
        "paragraphs_observed":          p["paragraphs_observed"],
        "paragraph_stagnation":         p["stagnation_ratio"],
        "telemetry_fault_rate":         0.0,
        "window_wpm_effective":         0.0,
        "mouse_efficiency_mean":        p.get("mouse_efficiency_mean", 0.97),
        "progress_markers_count":       p.get("progress_markers_count", 0),
        "panel_interaction_share":      p.get("panel_interaction_share", 0.0),
        "quality_confidence_mult":      1.0,
        "scroll_velocity_norm_std":     p.get("scroll_velocity_norm_std",
                                            baseline.get("scroll_velocity_norm_std", 0.015)),
        "scroll_capture_fault_rate":    0.0,
        "scroll_velocity_norm_mean":    p.get("scroll_velocity_norm_mean",
                                            baseline.get("scroll_velocity_norm_mean", 0.010)),
        "paragraph_missing_fault_rate": 0.0,
    }

    z     = _compute_z_scores(f, baseline)
    conf  = min(1.0, f["n_batches"] / 15.0)
    d_score, d_raw = _compute_disruption(z, baseline)
    e_score        = _compute_engagement(z, f)
    panel          = f["panel_interaction_share"]
    reader         = max(0.0, round(1.0 - panel, 4))

    beta_ema   = round(min(0.12, max(0.005, d_raw * 0.03)), 6)
    drift_ema  = round(d_raw * conf * 0.015, 6)
    drift_lvl  = round(drift_ema * 1.5, 6)
    beta_eff   = round(beta_ema * (1.0 + _rng.uniform(-0.02, 0.02)), 6)

    drift_block = {
        "beta_ema":         beta_ema,
        "drift_ema":        drift_ema,
        "confidence":       round(conf, 6),
        "pace_ratio":       f["pace_ratio"] if f["pace_available"] else None,
        "drift_level":      drift_lvl,
        "beta_effective":   beta_eff,
        "pace_available":   f["pace_available"],
        "disruption_score": d_score,
        "engagement_score": e_score,
    }
    ui_agg = {
        "panel_share_30s":                round(panel, 4),
        "reader_share_30s":               reader,
        "iz_other_share_30s":             0.0,
        "iz_panel_share_30s":             0.0,
        "iz_reader_share_30s":            reader,
        "ui_read_main_share_30s":         reader,
        "ui_panel_open_share_30s":        round(panel, 4),
        "ui_user_paused_share_30s":       0.0,
        "ui_panel_interacting_share_30s": round(panel * 0.8, 4),
    }
    bl_snap = {
        "baseline_json":      {k: v for k, v in baseline.items()
                               if not k.startswith("_")},
        "baseline_valid":     True,
        "baseline_updated_at": baseline.get("_updated_at",
                                "2026-03-20T08:30:00+00:00"),
    }
    debug = {
        "idle":           round(f["idle_ratio_mean"] - baseline["idle_ratio_mean"], 4),
        "pace":           0.0,
        "skim":           round(z["z_skim"], 4),
        "beta0":          0.01,
        "jitter":         round(z["z_jitter"], 4),
        "regress":        round(z["z_regress"], 4),
        "beta_ema":       beta_ema,
        "burstiness":     round(z["z_burstiness"], 4),
        "confidence":     round(conf, 4),
        "focus_loss":     round(z["z_focus_loss"], 4),
        "stagnation":     round(z["z_stagnation"], 4),
        "beta_effective": beta_eff,
        "disruption_raw": d_raw,
        "base_confidence": round(conf, 4),
        "disruption_score": d_score,
        "engagement_score": e_score,
    }
    packet = {
        "key":               f"u{user_id}_s{session_id}_p{seq}",
        "user_id":           user_id,
        "session_id":        session_id,
        "packet_seq":        seq,
        "created_at":        created_at.isoformat(),
        "window_start_at":   window_start.isoformat(),
        "window_end_at":     window_end.isoformat(),
        "drift":             drift_block,
        "features":          f,
        "z_scores":          z,
        "ui_aggregates":     ui_agg,
        "baseline_snapshot": bl_snap,
        "baseline_ref":      baseline_ref,
        "packet_raw": {
            "debug":             debug,
            "drift":             drift_block,
            "user_id":           user_id,
            "features":          f,
            "z_scores":          z,
            "session_id":        session_id,
            "document_id":       doc_id,
            "session_mode":      "adaptive",
            "ui_aggregates":     ui_agg,
            "baseline_snapshot": bl_snap,
        },
    }
    return packet, z, conf


# ── Session definitions ─────────────────────────────────────────────────────────
# Each entry in packet_rows is a dict of feature parameters + "target_state".
# Columns: n_batches, pace_ratio, pace_available, idle_ratio_mean, stagnation_ratio,
#          regress_rate_mean, focus_loss_rate, scroll_burstiness, scroll_pause_mean,
#          paragraphs_observed, progress_velocity, panel_interaction_share,
#          progress_markers_count, target_state
#
# Automatically derived (unless overridden):
#   scroll_jitter_mean      ← baseline value ± noise
#   scroll_velocity_norm_mean ← baseline * pace_ratio * 0.88 (hyperfocus) or * 0.75
#   mouse_path_px_mean      ← 0 for focused/hyp, small for others
#   mouse_efficiency_mean   ← 0.97
#
# stag_mu for user_248: (13 + 0.5*5)/30 = 0.517  →  z_stag = (stag - 0.517)/0.15
# stag_mu for user_249: (17 + 0.5*7)/30 = 0.683  →  z_stag = (stag - 0.683)/0.15

_S = True   # pace_available=True shorthand
_N = False  # pace_available=False shorthand


# ── SESSION 179  user_248  hyperfocused ────────────────────────────────────────
# User 248 is a fast reader (baseline WPM=415). This session shows a clear
# hyperfocused arc: early warm-up, sustained high pace_ratio 1.70–2.40, low idle,
# low stagnation → undampened z_skim ≥ 1.4 throughout.
_S179 = [
    # nb   pr    pa  idle  stag   regr  focl  burst  pause  para  pvel       panel  mrkr  target
    (  4, 1.00, _N, 0.190, 0.620, 0.002, 0.00, 1.22, 0.980,  1, 0.000600, 0.00, 0, "focused"),
    (  9, 1.00, _N, 0.160, 0.570, 0.001, 0.00, 1.15, 0.910,  1, 0.000900, 0.00, 0, "focused"),
    ( 13, 1.78, _S, 0.110, 0.360, 0.003, 0.00, 0.92, 0.840,  3, 0.005200, 0.00, 0, "hyperfocused"),
    ( 15, 1.82, _S, 0.090, 0.320, 0.002, 0.00, 0.88, 0.790,  4, 0.005800, 0.00, 1, "hyperfocused"),
    ( 16, 1.91, _S, 0.080, 0.300, 0.003, 0.00, 0.85, 0.760,  5, 0.006400, 0.00, 1, "hyperfocused"),
    ( 15, 2.03, _S, 0.100, 0.340, 0.002, 0.00, 0.91, 0.800,  5, 0.007200, 0.00, 1, "hyperfocused"),
    ( 16, 2.15, _S, 0.070, 0.280, 0.001, 0.00, 0.86, 0.740,  6, 0.008100, 0.00, 1, "hyperfocused"),
    ( 15, 2.22, _S, 0.090, 0.310, 0.004, 0.00, 0.89, 0.770,  5, 0.007900, 0.00, 1, "hyperfocused"),
    ( 16, 2.30, _S, 0.080, 0.270, 0.002, 0.00, 0.83, 0.720,  6, 0.008800, 0.00, 1, "hyperfocused"),
    ( 15, 2.40, _S, 0.070, 0.260, 0.001, 0.00, 0.80, 0.700,  6, 0.009400, 0.00, 1, "hyperfocused"),
    ( 16, 2.35, _S, 0.100, 0.290, 0.003, 0.00, 0.87, 0.750,  5, 0.009100, 0.00, 0, "hyperfocused"),
    ( 15, 2.28, _S, 0.120, 0.330, 0.002, 0.00, 0.92, 0.780,  5, 0.008600, 0.00, 1, "hyperfocused"),
    ( 16, 2.18, _S, 0.110, 0.300, 0.003, 0.00, 0.88, 0.770,  5, 0.008200, 0.00, 1, "hyperfocused"),
    ( 15, 2.08, _S, 0.140, 0.350, 0.002, 0.00, 0.91, 0.800,  4, 0.007600, 0.00, 0, "hyperfocused"),
    ( 16, 1.97, _S, 0.130, 0.370, 0.003, 0.00, 0.95, 0.820,  4, 0.006800, 0.00, 0, "hyperfocused"),
    ( 15, 1.88, _S, 0.150, 0.380, 0.002, 0.00, 0.97, 0.840,  4, 0.006200, 0.00, 1, "hyperfocused"),
    ( 16, 1.95, _S, 0.120, 0.350, 0.004, 0.00, 0.90, 0.810,  5, 0.007000, 0.00, 1, "hyperfocused"),
    ( 15, 2.05, _S, 0.100, 0.320, 0.003, 0.00, 0.88, 0.780,  5, 0.007500, 0.00, 1, "hyperfocused"),
    ( 16, 2.12, _S, 0.090, 0.310, 0.002, 0.00, 0.87, 0.760,  5, 0.007900, 0.00, 0, "hyperfocused"),
    ( 15, 2.20, _S, 0.080, 0.290, 0.001, 0.00, 0.84, 0.730,  6, 0.008500, 0.00, 1, "hyperfocused"),
    ( 16, 2.10, _S, 0.110, 0.330, 0.003, 0.00, 0.90, 0.790,  5, 0.007700, 0.00, 1, "hyperfocused"),
    ( 15, 2.00, _S, 0.130, 0.360, 0.002, 0.00, 0.93, 0.810,  4, 0.007200, 0.00, 0, "hyperfocused"),
    ( 16, 1.92, _S, 0.150, 0.380, 0.003, 0.00, 0.96, 0.830,  4, 0.006500, 0.00, 0, "hyperfocused"),
    ( 15, 1.71, _S, 0.200, 0.410, 0.004, 0.00, 1.02, 0.880,  3, 0.004200, 0.00, 0, "hyperfocused"),
    ( 15, 1.65, _S, 0.230, 0.430, 0.004, 0.00, 1.06, 0.910,  3, 0.003800, 0.00, 0, "hyperfocused"),
]

# ── SESSION 180  user_248  cognitive_overload ──────────────────────────────────
# Dense technical text: high regress_rate (0.13–0.22) + stagnation_ratio (0.70–1.00)
# → z_regress 1.33–2.83, z_stagnation 1.22–3.22.  Panel opened for AI help.
_S180 = [
    (  4, 1.00, _N, 0.240, 0.750, 0.030, 0.00, 1.45, 1.500,  1,  0.000200, 0.00, 0, "cognitive_overload"),
    (  9, 1.00, _N, 0.220, 0.800, 0.060, 0.00, 1.35, 1.620,  1,  0.000100, 0.00, 0, "cognitive_overload"),
    ( 13, 1.00, _N, 0.210, 0.850, 0.100, 0.00, 1.28, 1.780,  1,  0.000050, 0.00, 0, "cognitive_overload"),
    ( 15, 0.85, _S, 0.200, 0.880, 0.140, 0.00, 1.22, 1.950,  2, -0.000200, 0.10, 0, "cognitive_overload"),
    ( 15, 0.80, _S, 0.180, 0.920, 0.170, 0.00, 1.18, 2.100,  1, -0.000300, 0.15, 0, "cognitive_overload"),
    ( 16, 0.75, _S, 0.220, 0.880, 0.190, 0.00, 1.25, 2.050,  2, -0.000100, 0.12, 0, "cognitive_overload"),
    ( 15, 0.70, _S, 0.190, 0.950, 0.200, 0.00, 1.20, 2.200,  1, -0.000400, 0.18, 0, "cognitive_overload"),
    ( 16, 0.68, _S, 0.210, 1.000, 0.220, 0.00, 1.32, 2.350,  1, -0.000500, 0.20, 0, "cognitive_overload"),
    ( 15, 0.72, _S, 0.170, 0.980, 0.210, 0.00, 1.28, 2.280,  1, -0.000300, 0.22, 0, "cognitive_overload"),
    ( 16, 0.78, _S, 0.200, 0.930, 0.180, 0.00, 1.22, 2.120,  2, -0.000100, 0.15, 0, "cognitive_overload"),
    ( 15, 0.82, _S, 0.220, 0.880, 0.150, 0.00, 1.25, 1.980,  2,  0.000100, 0.10, 0, "cognitive_overload"),
    ( 16, 0.77, _S, 0.190, 0.900, 0.190, 0.00, 1.21, 2.080,  1, -0.000200, 0.18, 0, "cognitive_overload"),
    ( 15, 0.71, _S, 0.210, 0.950, 0.210, 0.00, 1.27, 2.220,  1, -0.000400, 0.20, 0, "cognitive_overload"),
    ( 16, 0.68, _S, 0.180, 0.980, 0.220, 0.00, 1.30, 2.380,  1, -0.000500, 0.22, 0, "cognitive_overload"),
    ( 15, 0.72, _S, 0.200, 0.920, 0.200, 0.00, 1.24, 2.150,  2, -0.000300, 0.18, 0, "cognitive_overload"),
    ( 16, 0.75, _S, 0.220, 0.850, 0.170, 0.00, 1.22, 2.020,  2, -0.000100, 0.12, 0, "cognitive_overload"),
    ( 15, 0.80, _S, 0.210, 0.820, 0.160, 0.00, 1.25, 1.920,  2,  0.000100, 0.10, 0, "cognitive_overload"),
    ( 16, 0.85, _S, 0.200, 0.800, 0.140, 0.00, 1.28, 1.850,  3,  0.000200, 0.08, 0, "cognitive_overload"),
    ( 15, 0.90, _S, 0.220, 0.770, 0.120, 0.00, 1.30, 1.750,  3,  0.000400, 0.05, 0, "cognitive_overload"),
    ( 16, 0.95, _S, 0.250, 0.720, 0.100, 0.00, 1.35, 1.650,  3,  0.000600, 0.05, 0, "cognitive_overload"),
    ( 15, 1.00, _N, 0.250, 0.730, 0.080, 0.00, 1.38, 1.620,  3,  0.000500, 0.03, 0, "cognitive_overload"),
    ( 16, 1.00, _N, 0.270, 0.750, 0.070, 0.00, 1.40, 1.680,  2,  0.000400, 0.03, 0, "cognitive_overload"),
    ( 15, 1.00, _N, 0.260, 0.780, 0.100, 0.00, 1.38, 1.720,  2,  0.000200, 0.05, 0, "cognitive_overload"),
    ( 16, 0.95, _S, 0.240, 0.800, 0.120, 0.00, 1.35, 1.800,  2,  0.000100, 0.08, 0, "cognitive_overload"),
    ( 15, 0.88, _S, 0.230, 0.820, 0.150, 0.00, 1.30, 1.880,  2, -0.000200, 0.10, 0, "cognitive_overload"),
]

# ── SESSION 181  user_248  drifting (tab-switching) ────────────────────────────
# Reader repeatedly switches to other tabs (focus_loss_rate 0.10–0.25).
# z_focus_loss 1.25–3.0 is the primary drifting signal.  Stagnation is mild
# (not advancing much while tabbed away).
_S181 = [
    (  4, 1.00, _N, 0.200, 0.580, 0.002, 0.000, 1.25, 0.980,  1, 0.000500, 0.00, 0, "focused"),
    (  9, 1.00, _N, 0.220, 0.540, 0.003, 0.000, 1.18, 0.920,  1, 0.000800, 0.00, 0, "focused"),
    ( 13, 1.05, _S, 0.250, 0.500, 0.004, 0.050, 1.22, 1.050,  2, 0.001200, 0.05, 0, "drifting"),
    ( 15, 1.00, _N, 0.450, 0.550, 0.005, 0.100, 1.52, 1.350,  2, 0.000500, 0.08, 0, "drifting"),
    ( 16, 1.00, _N, 0.480, 0.580, 0.004, 0.130, 1.65, 1.480,  2, 0.000300, 0.10, 0, "drifting"),
    ( 15, 0.95, _S, 0.420, 0.520, 0.003, 0.150, 1.72, 1.550,  2, 0.000400, 0.12, 0, "drifting"),
    ( 16, 1.00, _N, 0.500, 0.600, 0.004, 0.180, 1.85, 1.650,  1, 0.000200, 0.15, 0, "drifting"),
    ( 15, 1.00, _N, 0.520, 0.620, 0.005, 0.200, 1.95, 1.750,  1, 0.000100, 0.18, 0, "drifting"),
    ( 16, 1.00, _N, 0.480, 0.580, 0.004, 0.220, 2.05, 1.820,  1, 0.000200, 0.20, 0, "drifting"),
    ( 15, 0.98, _S, 0.450, 0.550, 0.003, 0.200, 1.98, 1.780,  2, 0.000300, 0.18, 0, "drifting"),
    ( 16, 1.00, _N, 0.500, 0.600, 0.004, 0.180, 1.88, 1.680,  1, 0.000200, 0.15, 0, "drifting"),
    ( 15, 1.00, _N, 0.520, 0.620, 0.005, 0.150, 1.80, 1.620,  1, 0.000100, 0.12, 0, "drifting"),
    ( 16, 1.00, _N, 0.480, 0.570, 0.004, 0.200, 1.92, 1.720,  1, 0.000200, 0.18, 0, "drifting"),
    ( 15, 1.00, _N, 0.500, 0.600, 0.004, 0.220, 2.02, 1.800,  1, 0.000100, 0.20, 0, "drifting"),
    ( 16, 1.00, _N, 0.520, 0.630, 0.005, 0.250, 2.15, 1.950,  1, 0.000100, 0.22, 0, "drifting"),
    ( 15, 0.95, _S, 0.480, 0.580, 0.004, 0.200, 1.95, 1.780,  2, 0.000200, 0.18, 0, "drifting"),
    ( 16, 1.00, _N, 0.500, 0.600, 0.004, 0.180, 1.85, 1.680,  1, 0.000200, 0.15, 0, "drifting"),
    ( 15, 1.00, _N, 0.520, 0.620, 0.005, 0.150, 1.78, 1.620,  1, 0.000100, 0.12, 0, "drifting"),
    ( 16, 1.00, _N, 0.480, 0.570, 0.003, 0.120, 1.68, 1.520,  2, 0.000300, 0.10, 0, "drifting"),
    ( 15, 1.00, _N, 0.450, 0.550, 0.003, 0.100, 1.60, 1.420,  2, 0.000400, 0.08, 0, "drifting"),
    ( 16, 1.00, _N, 0.420, 0.520, 0.004, 0.080, 1.52, 1.350,  2, 0.000500, 0.05, 0, "drifting"),
    ( 15, 1.00, _N, 0.380, 0.500, 0.004, 0.050, 1.42, 1.250,  2, 0.000800, 0.03, 0, "drifting"),
    ( 16, 1.05, _S, 0.350, 0.480, 0.003, 0.030, 1.35, 1.180,  3, 0.001200, 0.02, 0, "drifting"),
    ( 15, 1.10, _S, 0.300, 0.450, 0.002, 0.010, 1.28, 1.100,  3, 0.001800, 0.01, 0, "focused"),
    ( 15, 1.12, _S, 0.280, 0.430, 0.002, 0.000, 1.22, 1.050,  3, 0.002200, 0.00, 0, "focused"),
]

# ── SESSION 182  user_249  hyperfocused ────────────────────────────────────────
# User 249 has a tight idle baseline (0.22, std=0.18). During hyperfocus,
# idle drops to 0.07–0.19 and pace_ratio reaches 1.65–2.17 → undampened z_skim.
# stag_mu_249 = 0.683; stagnation_ratio stays below that during flow.
_S182 = [
    (  4, 1.00, _N, 0.150, 0.450, 0.008, 0.00, 1.18, 0.850,  1, 0.000800, 0.00, 0, "focused"),
    (  9, 1.00, _N, 0.120, 0.400, 0.006, 0.00, 1.10, 0.780,  2, 0.001500, 0.00, 0, "focused"),
    ( 13, 1.68, _S, 0.090, 0.320, 0.005, 0.00, 0.92, 0.720,  3, 0.004800, 0.00, 0, "hyperfocused"),
    ( 15, 1.73, _S, 0.080, 0.290, 0.004, 0.00, 0.88, 0.690,  4, 0.005500, 0.00, 1, "hyperfocused"),
    ( 16, 1.81, _S, 0.070, 0.270, 0.003, 0.00, 0.85, 0.660,  5, 0.006200, 0.00, 1, "hyperfocused"),
    ( 15, 1.90, _S, 0.080, 0.300, 0.004, 0.00, 0.88, 0.700,  5, 0.007000, 0.00, 1, "hyperfocused"),
    ( 16, 1.98, _S, 0.070, 0.270, 0.003, 0.00, 0.84, 0.670,  5, 0.007600, 0.00, 1, "hyperfocused"),
    ( 15, 2.05, _S, 0.080, 0.280, 0.004, 0.00, 0.86, 0.680,  6, 0.008200, 0.00, 1, "hyperfocused"),
    ( 16, 2.10, _S, 0.070, 0.260, 0.002, 0.00, 0.83, 0.650,  6, 0.008800, 0.00, 1, "hyperfocused"),
    ( 15, 2.17, _S, 0.090, 0.290, 0.003, 0.00, 0.87, 0.700,  5, 0.008300, 0.00, 1, "hyperfocused"),
    ( 16, 2.12, _S, 0.080, 0.270, 0.003, 0.00, 0.85, 0.680,  5, 0.008000, 0.00, 0, "hyperfocused"),
    ( 15, 2.05, _S, 0.100, 0.310, 0.004, 0.00, 0.89, 0.720,  5, 0.007600, 0.00, 1, "hyperfocused"),
    ( 16, 1.98, _S, 0.090, 0.290, 0.004, 0.00, 0.87, 0.700,  5, 0.007200, 0.00, 1, "hyperfocused"),
    ( 15, 1.92, _S, 0.110, 0.320, 0.005, 0.00, 0.90, 0.740,  4, 0.006800, 0.00, 0, "hyperfocused"),
    ( 16, 1.88, _S, 0.100, 0.300, 0.004, 0.00, 0.88, 0.720,  4, 0.006500, 0.00, 0, "hyperfocused"),
    ( 15, 1.95, _S, 0.090, 0.280, 0.003, 0.00, 0.86, 0.690,  5, 0.007000, 0.00, 1, "hyperfocused"),
    ( 16, 2.00, _S, 0.080, 0.270, 0.003, 0.00, 0.84, 0.670,  5, 0.007500, 0.00, 1, "hyperfocused"),
    ( 15, 2.05, _S, 0.090, 0.290, 0.004, 0.00, 0.87, 0.700,  5, 0.007800, 0.00, 1, "hyperfocused"),
    ( 16, 1.97, _S, 0.100, 0.310, 0.004, 0.00, 0.90, 0.730,  4, 0.007200, 0.00, 0, "hyperfocused"),
    ( 15, 1.91, _S, 0.110, 0.330, 0.005, 0.00, 0.93, 0.760,  4, 0.006800, 0.00, 0, "hyperfocused"),
    ( 16, 1.85, _S, 0.120, 0.350, 0.004, 0.00, 0.95, 0.780,  4, 0.006200, 0.00, 0, "hyperfocused"),
    ( 15, 1.79, _S, 0.140, 0.370, 0.005, 0.00, 0.98, 0.810,  4, 0.005800, 0.00, 0, "hyperfocused"),
    ( 16, 1.73, _S, 0.150, 0.380, 0.004, 0.00, 1.00, 0.830,  4, 0.005200, 0.00, 0, "hyperfocused"),
    ( 15, 1.68, _S, 0.170, 0.400, 0.005, 0.00, 1.03, 0.860,  3, 0.004700, 0.00, 0, "hyperfocused"),
    ( 15, 1.66, _S, 0.190, 0.420, 0.005, 0.00, 1.06, 0.890,  3, 0.004300, 0.00, 0, "hyperfocused"),
]

# ── SESSION 183  user_249  cognitive_overload ──────────────────────────────────
# Dense material: regress_rate 0.12–0.23 → z_regress 1.17–3.0.
# stagnation_ratio 0.80–0.95 → z_stagnation 0.78–1.78 (stag_mu_249=0.683).
# Panel opened frequently for AI help.
_S183 = [
    (  4, 1.00, _N, 0.220, 0.820, 0.025, 0.00, 1.42, 1.450,  1,  0.000300, 0.00, 0, "cognitive_overload"),
    (  9, 1.00, _N, 0.200, 0.850, 0.040, 0.00, 1.35, 1.620,  1,  0.000200, 0.00, 0, "cognitive_overload"),
    ( 13, 1.00, _N, 0.210, 0.880, 0.070, 0.00, 1.28, 1.780,  1,  0.000100, 0.00, 0, "cognitive_overload"),
    ( 15, 0.88, _S, 0.190, 0.900, 0.120, 0.00, 1.22, 1.950,  2, -0.000200, 0.12, 0, "cognitive_overload"),
    ( 16, 0.82, _S, 0.180, 0.920, 0.160, 0.00, 1.18, 2.120,  1, -0.000400, 0.18, 0, "cognitive_overload"),
    ( 15, 0.78, _S, 0.200, 0.880, 0.180, 0.00, 1.22, 2.050,  2, -0.000200, 0.15, 0, "cognitive_overload"),
    ( 16, 0.72, _S, 0.190, 0.920, 0.200, 0.00, 1.20, 2.220,  1, -0.000500, 0.20, 0, "cognitive_overload"),
    ( 15, 0.68, _S, 0.210, 0.950, 0.220, 0.00, 1.28, 2.380,  1, -0.000600, 0.22, 0, "cognitive_overload"),
    ( 16, 0.72, _S, 0.180, 0.920, 0.210, 0.00, 1.24, 2.280,  1, -0.000400, 0.20, 0, "cognitive_overload"),
    ( 15, 0.78, _S, 0.200, 0.900, 0.190, 0.00, 1.22, 2.150,  2, -0.000200, 0.17, 0, "cognitive_overload"),
    ( 16, 0.82, _S, 0.220, 0.870, 0.160, 0.00, 1.25, 2.020,  2, -0.000100, 0.12, 0, "cognitive_overload"),
    ( 15, 0.78, _S, 0.200, 0.900, 0.200, 0.00, 1.22, 2.120,  1, -0.000300, 0.18, 0, "cognitive_overload"),
    ( 16, 0.72, _S, 0.190, 0.930, 0.220, 0.00, 1.28, 2.280,  1, -0.000500, 0.20, 0, "cognitive_overload"),
    ( 15, 0.68, _S, 0.210, 0.950, 0.230, 0.00, 1.32, 2.420,  1, -0.000700, 0.22, 0, "cognitive_overload"),
    ( 16, 0.72, _S, 0.190, 0.920, 0.210, 0.00, 1.26, 2.250,  1, -0.000400, 0.20, 0, "cognitive_overload"),
    ( 15, 0.78, _S, 0.200, 0.880, 0.170, 0.00, 1.23, 2.080,  2, -0.000200, 0.15, 0, "cognitive_overload"),
    ( 16, 0.82, _S, 0.220, 0.850, 0.150, 0.00, 1.25, 1.950,  2, -0.000100, 0.10, 0, "cognitive_overload"),
    ( 15, 0.88, _S, 0.220, 0.830, 0.130, 0.00, 1.28, 1.850,  3,  0.000100, 0.08, 0, "cognitive_overload"),
    ( 16, 0.92, _S, 0.230, 0.800, 0.110, 0.00, 1.30, 1.750,  3,  0.000300, 0.05, 0, "cognitive_overload"),
    ( 15, 0.98, _S, 0.240, 0.780, 0.090, 0.00, 1.32, 1.680,  3,  0.000500, 0.05, 0, "cognitive_overload"),
    ( 16, 1.00, _N, 0.240, 0.770, 0.080, 0.00, 1.35, 1.620,  3,  0.000400, 0.03, 0, "cognitive_overload"),
    ( 15, 1.00, _N, 0.250, 0.780, 0.090, 0.00, 1.38, 1.680,  2,  0.000300, 0.03, 0, "cognitive_overload"),
    ( 16, 1.00, _N, 0.230, 0.800, 0.110, 0.00, 1.35, 1.750,  2,  0.000100, 0.05, 0, "cognitive_overload"),
    ( 15, 0.95, _S, 0.220, 0.820, 0.140, 0.00, 1.32, 1.850,  2, -0.000100, 0.08, 0, "cognitive_overload"),
    ( 15, 0.90, _S, 0.230, 0.850, 0.160, 0.00, 1.28, 1.920,  2, -0.000300, 0.10, 0, "cognitive_overload"),
]

# ── SESSION 184  user_249  drifting (idle + burstiness) ───────────────────────
# User 249 has a tight idle baseline (0.22, std=0.18).  When idle rises to
# 0.55–0.72, z_idle = 1.83–2.78 — strong mind-wandering signal.
# Combined with erratic scroll (z_burstiness 1.6–2.6), this is clear drifting.
_S184 = [
    (  4, 1.00, _N, 0.280, 0.450, 0.010, 0.00, 1.32, 1.150,  1, 0.000600, 0.00, 0, "drifting"),
    (  9, 1.00, _N, 0.320, 0.480, 0.012, 0.00, 1.42, 1.250,  1, 0.000500, 0.00, 0, "drifting"),
    ( 13, 1.00, _N, 0.400, 0.520, 0.010, 0.00, 1.55, 1.400,  1, 0.000400, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.550, 0.550, 0.012, 0.00, 1.85, 1.620,  2, 0.000300, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.600, 0.580, 0.011, 0.00, 2.00, 1.780,  1, 0.000200, 0.00, 0, "drifting"),
    ( 15, 0.95, _S, 0.620, 0.550, 0.010, 0.00, 2.05, 1.820,  1, 0.000100, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.650, 0.600, 0.012, 0.00, 2.10, 1.920,  1, 0.000100, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.680, 0.620, 0.013, 0.00, 2.18, 2.020,  1, 0.000050, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.700, 0.600, 0.012, 0.00, 2.22, 2.080,  1, 0.000050, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.720, 0.620, 0.011, 0.00, 2.28, 2.150,  1, 0.000050, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.680, 0.600, 0.012, 0.00, 2.20, 2.050,  1, 0.000100, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.650, 0.580, 0.011, 0.00, 2.12, 1.950,  1, 0.000100, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.700, 0.620, 0.012, 0.00, 2.25, 2.120,  1, 0.000050, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.720, 0.630, 0.013, 0.00, 2.30, 2.180,  1, 0.000050, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.680, 0.600, 0.012, 0.00, 2.18, 2.050,  1, 0.000100, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.650, 0.580, 0.011, 0.00, 2.10, 1.950,  1, 0.000200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.620, 0.570, 0.010, 0.00, 2.02, 1.880,  2, 0.000300, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.600, 0.550, 0.010, 0.00, 1.95, 1.820,  2, 0.000300, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.580, 0.530, 0.009, 0.00, 1.88, 1.750,  2, 0.000400, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.550, 0.520, 0.010, 0.00, 1.82, 1.680,  2, 0.000500, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.520, 0.500, 0.009, 0.00, 1.75, 1.600,  2, 0.000600, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.480, 0.480, 0.009, 0.00, 1.68, 1.520,  2, 0.000700, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.440, 0.470, 0.010, 0.00, 1.60, 1.450,  2, 0.000900, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.380, 0.450, 0.009, 0.00, 1.50, 1.350,  3, 0.001200, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.320, 0.430, 0.009, 0.00, 1.42, 1.250,  3, 0.001500, 0.00, 0, "focused"),
]

# ── SESSION 200  user_260  ADHD drifting showcase — all six phenotypes ─────────
#
# User 260 is a completely normal reader (WPM=310, idle_ratio_mean=0.28).
# For this session they are highly ADHD and cycle through every drifting
# phenotype naturally:
#
#   stag_mu_260 = (15 + 0.5*6) / 30 = 0.60
#   z_idle   = (idle  - 0.28) / 0.22
#   z_regress = z_pos(regr, max(0.015,0.05)=0.05, max(0.025,0.06)=0.06)
#   z_pause  = (pause - 1.05) / 0.85
#   z_burst  = (burst - 1.0)  / 0.50   (floored at 0)
#   z_stag   = (stag  - 0.60) / 0.15   (floored at 0)
#   z_focus  = focus_loss_rate / 0.08
#   skim damping: idle_damp = 1 - (idle-0.40)/(0.80-0.40) when idle > 0.40
#
# Arc: warm-up → focused → zombie-scroll → burst-freeze → tab-switching →
#      panel+focus recovery → aimless-regression → surface-reading →
#      brief recovery → frozen-gaze → dampened-flick → heavy tabs →
#      panel+focus → zombie+burst-freeze mix → aimless+surface →
#      final frozen-gaze

_S = True   # pace_available=True
_N = False  # pace_available=False

_S200 = [
    # ── PHASE 1: Warm-up / initial focus (p0-p4) ────────────────────────────
    # nb   pr    pa  idle  stag   regr  focl  burst  pause  para  pvel       panel  mrkr  target
    (  4, 1.00, _N, 0.250, 0.700, 0.010, 0.000, 0.900, 1.200,  1, 0.000400, 0.00, 0, "focused"),
    (  9, 1.00, _N, 0.220, 0.650, 0.008, 0.000, 0.950, 1.100,  2, 0.001200, 0.00, 0, "focused"),
    ( 14, 0.92, _S, 0.240, 0.580, 0.012, 0.000, 0.880, 1.050,  3, 0.001800, 0.00, 0, "focused"),
    ( 16, 0.88, _S, 0.260, 0.550, 0.018, 0.000, 0.920, 1.150,  3, 0.001600, 0.00, 0, "focused"),
    ( 16, 0.95, _S, 0.280, 0.530, 0.015, 0.000, 0.980, 1.080,  4, 0.002400, 0.00, 1, "focused"),

    # ── PHASE 2: Steady focused reading (p5-p8) ─────────────────────────────
    ( 16, 1.02, _S, 0.250, 0.500, 0.016, 0.000, 0.920, 0.950,  5, 0.003200, 0.00, 1, "focused"),
    ( 16, 1.05, _S, 0.270, 0.520, 0.014, 0.000, 0.950, 1.000,  5, 0.003500, 0.00, 0, "focused"),
    ( 16, 0.98, _S, 0.240, 0.550, 0.012, 0.000, 0.900, 1.020,  4, 0.002800, 0.00, 1, "focused"),
    ( 16, 1.00, _N, 0.260, 0.570, 0.014, 0.000, 0.930, 1.050,  4, 0.002600, 0.00, 0, "focused"),

    # ── PHASE 3: Zombie-scroll drift — looks fine, zoning out (p9-p14) ──────
    # Low idle, near-zero stag and pause, many paragraphs, high progress.
    # All z-scores near zero but the reader is not actually absorbing content.
    ( 16, 1.00, _N, 0.200, 0.380, 0.010, 0.000, 0.850, 0.520,  7, 0.006200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.180, 0.320, 0.008, 0.000, 0.800, 0.380,  9, 0.008100, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.150, 0.280, 0.006, 0.000, 0.780, 0.250, 10, 0.009500, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.170, 0.300, 0.008, 0.000, 0.820, 0.300,  9, 0.008800, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.190, 0.350, 0.009, 0.000, 0.840, 0.420,  8, 0.007500, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.220, 0.400, 0.010, 0.000, 0.880, 0.550,  7, 0.005800, 0.00, 0, "drifting"),

    # ── PHASE 4: Burst-freeze drift (p15-p19) ────────────────────────────────
    # z_burst and z_pause both elevated; stagnation LOW (not stuck — bouncing).
    # Reader makes short erratic scrolls then freezes; mind has wandered.
    ( 16, 1.00, _N, 0.350, 0.420, 0.012, 0.000, 1.650, 1.800,  5, 0.002800, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.380, 0.450, 0.010, 0.000, 1.950, 2.200,  4, 0.002000, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.400, 0.400, 0.008, 0.000, 2.150, 2.800,  4, 0.001500, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.420, 0.380, 0.009, 0.000, 2.450, 3.100,  3, 0.001200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.450, 0.400, 0.008, 0.000, 2.600, 3.500,  3, 0.001000, 0.00, 0, "drifting"),

    # ── PHASE 5: Tab-switching drift (p20-p24) ───────────────────────────────
    # z_focus_loss rising as reader escapes to other apps.
    ( 16, 1.00, _N, 0.420, 0.550, 0.010, 0.075, 1.800, 2.600,  3, 0.001000, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.480, 0.620, 0.008, 0.133, 1.550, 3.200,  2, 0.000600, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.520, 0.680, 0.006, 0.200, 1.350, 4.100,  2, 0.000400, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.550, 0.700, 0.005, 0.250, 1.280, 4.500,  2, 0.000300, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.500, 0.650, 0.008, 0.200, 1.400, 4.000,  2, 0.000500, 0.00, 0, "drifting"),

    # ── PHASE 6: Panel interaction → brief focus recovery (p25-p28) ──────────
    # Reader opens the AI panel; pace drops below baseline (careful reading).
    # Focus loss fades, engagement rises. Two clean focused windows.
    ( 16, 0.82, _S, 0.350, 0.620, 0.010, 0.062, 1.200, 2.800,  3, 0.001500, 0.25, 0, "focused"),
    ( 16, 0.75, _S, 0.300, 0.580, 0.012, 0.000, 1.100, 2.200,  3, 0.001800, 0.30, 0, "focused"),
    ( 16, 0.85, _S, 0.280, 0.550, 0.015, 0.000, 0.950, 1.850,  4, 0.002400, 0.20, 1, "focused"),
    ( 16, 0.92, _S, 0.260, 0.520, 0.016, 0.000, 0.900, 1.450,  4, 0.002800, 0.15, 0, "focused"),

    # ── PHASE 7: Aimless regression drift (p29-p34) ──────────────────────────
    # z_regress elevated but wide paragraph coverage + flat/negative progress.
    # Reader scrolls back without a clear target — unfocused checking vs
    # the purposeful overload rereading pattern (which narrows in scope).
    ( 16, 1.00, _N, 0.300, 0.420, 0.085, 0.000, 1.250, 1.600,  6, 0.000800, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.280, 0.380, 0.105, 0.000, 1.350, 1.400,  7, 0.000500, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.250, 0.350, 0.120, 0.000, 1.200, 1.200,  7, -0.000300, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.220, 0.320, 0.135, 0.000, 1.150, 0.980,  8, -0.000800, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.240, 0.350, 0.150, 0.000, 1.300, 1.100,  7, -0.001200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.280, 0.400, 0.125, 0.000, 1.220, 1.250,  6, -0.000600, 0.00, 0, "drifting"),

    # ── PHASE 8: Surface reading drift (p35-p38) ─────────────────────────────
    # All z-scores are near-zero (looks normal) but progress_velocity is
    # near-zero and only 1-2 paragraphs are covered. Reader is re-reading
    # the same passage mentally absent — the subtlest drift phenotype.
    ( 16, 1.00, _N, 0.300, 0.580, 0.018, 0.000, 0.980, 1.250,  2, 0.000280, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.280, 0.600, 0.015, 0.000, 0.950, 1.200,  2, 0.000200, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.320, 0.620, 0.014, 0.000, 1.020, 1.300,  2, 0.000150, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.350, 0.650, 0.015, 0.000, 1.050, 1.400,  2, 0.000180, 0.00, 0, "drifting"),

    # ── PHASE 9: Brief focused recovery (p39-p41) ────────────────────────────
    ( 16, 0.95, _S, 0.250, 0.550, 0.016, 0.000, 0.900, 1.050,  4, 0.002800, 0.10, 0, "focused"),
    ( 16, 1.05, _S, 0.240, 0.520, 0.015, 0.000, 0.880, 0.980,  5, 0.003500, 0.05, 1, "focused"),
    ( 16, 1.08, _S, 0.260, 0.530, 0.014, 0.000, 0.920, 1.020,  5, 0.003200, 0.00, 0, "focused"),

    # ── PHASE 10: Frozen-gaze drift (p42-p47) ────────────────────────────────
    # High stagnation + near-zero burstiness + long pause + elevated idle.
    # No thrashing (unlike overload): the reader simply stopped and stares.
    ( 16, 1.00, _N, 0.450, 0.780, 0.008, 0.000, 0.400, 2.800,  2, 0.000500, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.480, 0.820, 0.006, 0.000, 0.300, 3.200,  1, 0.000300, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.520, 0.880, 0.005, 0.000, 0.250, 3.800,  1, 0.000200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.550, 0.900, 0.004, 0.000, 0.200, 4.200,  1, 0.000100, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.500, 0.850, 0.006, 0.000, 0.280, 3.600,  1, 0.000200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.450, 0.800, 0.008, 0.000, 0.350, 3.100,  2, 0.000400, 0.00, 0, "drifting"),

    # ── PHASE 11: Dampened-flick drift (p48-p52) ─────────────────────────────
    # pace_ratio > 1.6 → z_skim_raw > 0 BUT idle > 0.40 triggers idle_damp.
    # Reader skims at surface speed, mind is absent — z_skim is suppressed
    # (0.3-1.1) by the high idle fraction, so the model sees partial skim,
    # high idle, and no genuine engagement.
    ( 16, 1.70, _S, 0.500, 0.420, 0.008, 0.000, 1.550, 1.200,  6, 0.004200, 0.00, 0, "drifting"),
    ( 16, 1.85, _S, 0.550, 0.380, 0.006, 0.000, 1.650, 1.400,  7, 0.005200, 0.00, 0, "drifting"),
    ( 16, 1.95, _S, 0.620, 0.350, 0.005, 0.000, 1.800, 1.600,  8, 0.005800, 0.00, 0, "drifting"),
    ( 15, 2.00, _S, 0.680, 0.320, 0.005, 0.000, 1.850, 1.800,  9, 0.006200, 0.00, 0, "drifting"),
    ( 16, 1.80, _S, 0.720, 0.350, 0.008, 0.000, 1.700, 2.000,  8, 0.005400, 0.00, 0, "drifting"),

    # ── PHASE 12: Heavy tab-switching (p53-p56) ──────────────────────────────
    ( 16, 1.00, _N, 0.580, 0.650, 0.006, 0.125, 1.450, 3.500,  3, 0.000800, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.620, 0.700, 0.005, 0.188, 1.380, 4.200,  2, 0.000500, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.650, 0.720, 0.004, 0.200, 1.320, 4.800,  2, 0.000300, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.600, 0.680, 0.005, 0.250, 1.400, 5.000,  2, 0.000200, 0.00, 0, "drifting"),

    # ── PHASE 13: Panel → brief focus again (p57-p60) ────────────────────────
    # Reader opens AI panel again; focus_loss fades, pace slows, focus returns.
    ( 16, 0.75, _S, 0.380, 0.620, 0.012, 0.062, 1.150, 2.800,  3, 0.001600, 0.28, 0, "focused"),
    ( 16, 0.80, _S, 0.300, 0.580, 0.015, 0.000, 1.050, 2.200,  3, 0.002000, 0.22, 1, "focused"),
    ( 16, 0.90, _S, 0.260, 0.550, 0.016, 0.000, 0.950, 1.800,  4, 0.002600, 0.12, 0, "focused"),
    ( 16, 0.95, _S, 0.250, 0.520, 0.015, 0.000, 0.920, 1.500,  4, 0.002900, 0.05, 1, "focused"),

    # ── PHASE 14: Mixed zombie + burst-freeze drift returns (p61-p67) ─────────
    # After the panel closes, drift resurfaces — first zombie-scroll phenotype
    # then transitioning into burst-freeze as restlessness grows.
    ( 16, 1.00, _N, 0.200, 0.350, 0.009, 0.000, 0.820, 0.450,  8, 0.007200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.180, 0.300, 0.008, 0.000, 0.780, 0.320, 10, 0.009500, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.250, 0.380, 0.010, 0.000, 1.450, 1.600,  6, 0.004200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.380, 0.420, 0.008, 0.000, 2.050, 2.400,  4, 0.001800, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.420, 0.400, 0.006, 0.000, 2.350, 2.950,  3, 0.001200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.450, 0.380, 0.007, 0.000, 2.500, 3.300,  3, 0.000900, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.400, 0.400, 0.009, 0.000, 2.200, 2.750,  4, 0.001400, 0.00, 0, "drifting"),

    # ── PHASE 15: Aimless regression + surface reading mix (p68-p74) ──────────
    # Regression returns without narrowing scope (aimless), then dissolves
    # into surface reading where everything looks normal but nothing moves.
    ( 16, 1.00, _N, 0.320, 0.450, 0.095, 0.000, 1.250, 1.600,  6, -0.000400, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.280, 0.420, 0.115, 0.000, 1.200, 1.400,  7, -0.000800, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.300, 0.380, 0.130, 0.000, 1.150, 1.300,  8, -0.001200, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.320, 0.550, 0.018, 0.000, 0.950, 1.280,  2, 0.000180, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.300, 0.580, 0.016, 0.000, 0.980, 1.180,  2, 0.000210, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.280, 0.600, 0.015, 0.000, 1.000, 1.220,  2, 0.000160, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.300, 0.620, 0.015, 0.000, 1.020, 1.250,  2, 0.000140, 0.00, 0, "drifting"),

    # ── PHASE 16: Final frozen-gaze drift (p75-p79) ──────────────────────────
    # Session ends in the quietest, most invisible drift: staring at the page,
    # barely scrolling, very long pauses, near-zero progress.
    ( 16, 1.00, _N, 0.480, 0.800, 0.005, 0.000, 0.300, 3.500,  1, 0.000200, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.520, 0.850, 0.004, 0.000, 0.220, 3.900,  1, 0.000100, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.550, 0.900, 0.003, 0.000, 0.180, 4.200,  1, 0.000100, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.500, 0.850, 0.005, 0.000, 0.250, 3.800,  1, 0.000100, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.450, 0.780, 0.008, 0.000, 0.320, 3.200,  2, 0.000300, 0.00, 0, "drifting"),
]


# ── SESSION 210  user_261  cognitive_overload — 100-packet major overload arc ──
#
# User 261 baseline: WPM=195, idle_ratio_mean=0.30, idle_ratio_std=0.18,
# regress_rate_mean=0.0 (calibration forward-only), scroll_velocity_norm_mean=0.0075
#
# Key z-score thresholds for user_261:
#   z_regress = (regr - 0.05) / 0.06       → 3.0 at regr ≥ 0.23
#   z_stag    = (stag - 0.55) / 0.15       → 3.0 at stag ≥ 1.00
#     stag_mu = (14 + 0.5×5) / 30 = 0.55
#   z_idle    = (idle - 0.30) / 0.18       → 3.0 at idle ≥ 0.84
#   z_pause   = (pause - 1.10) / 0.85      → 3.0 at pause ≥ 3.65 s
#   z_burst   = (burst - 1.0)  / 0.50      → 3.0 at burst ≥ 2.5
#   velocity fallback:  fires when sv > 0.0075×1.6=0.012 (pace_available=False)
#                       15th tuple element = sv_override; 0.0 = use default
#
# Session arc:
#   p0–p7:   warm-up → focused reading
#   p8–p14:  overload building (regression + stagnation rising)
#   p15–p24: full rereading overload (z_regress→3.0)
#   p25–p33: acute episode 1 — thrashing on 1-2 paragraphs (triple-max)
#   p34–p41: wide-scope frantic rereading — HIGH sv triggers velocity fallback
#   p42–p47: partial panel recovery
#   p48–p57: second overload wave
#   p58–p65: second frantic high-speed regression (velocity fallback)
#   p66–p73: freeze-thrash alternation (max stag + max burst cycling)
#   p74–p82: exhausted rereading (high idle + moderate regression)
#   p83–p91: panel-assisted partial recovery
#   p92–p99: final acute episode

# nb  pr    pa  idle  stag   regr  focl  burst  pause  para  pvel     panel mrkr  target          sv_override
_S210 = [
    # ── PHASE 1: Warm-up / initial focused reading (p0–p7) ──────────────────
    (  4, 1.00, _N, 0.270, 0.600, 0.008, 0.000, 0.940, 1.100,  1,  0.0010, 0.00, 0, "focused",          0.0),
    (  9, 1.00, _N, 0.280, 0.570, 0.010, 0.000, 0.920, 1.080,  2,  0.0016, 0.00, 0, "focused",          0.0),
    ( 13, 0.92, _S, 0.300, 0.540, 0.012, 0.000, 0.900, 1.050,  3,  0.0021, 0.00, 0, "focused",          0.0),
    ( 15, 0.90, _S, 0.280, 0.520, 0.014, 0.000, 0.880, 1.020,  3,  0.0024, 0.00, 0, "focused",          0.0),
    ( 16, 0.95, _S, 0.290, 0.510, 0.013, 0.000, 0.900, 1.000,  4,  0.0027, 0.00, 1, "focused",          0.0),
    ( 16, 0.98, _S, 0.300, 0.500, 0.014, 0.000, 0.880, 0.980,  4,  0.0030, 0.00, 0, "focused",          0.0),
    ( 16, 1.00, _S, 0.300, 0.520, 0.016, 0.000, 0.900, 1.020,  4,  0.0028, 0.00, 0, "focused",          0.0),
    ( 16, 1.02, _S, 0.280, 0.500, 0.015, 0.000, 0.880, 0.980,  4,  0.0030, 0.00, 1, "focused",          0.0),

    # ── PHASE 2: Overload building (p8–p14) — regression + stagnation rising ──
    # pace_available=True for p8–p10 (gate not yet tripped); _N from p11 onward.
    # Regression drag fix applies to p8–p10 even though gate hasn't fired yet.
    ( 16, 0.90, _S, 0.320, 0.620, 0.030, 0.000, 1.050, 1.350,  3,  0.0018, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 0.85, _S, 0.340, 0.660, 0.060, 0.000, 1.100, 1.480,  3,  0.0012, 0.05, 0, "cognitive_overload", 0.0),
    ( 16, 0.82, _S, 0.360, 0.700, 0.080, 0.000, 1.150, 1.620,  2,  0.0008, 0.08, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.380, 0.720, 0.100, 0.000, 1.180, 1.750,  2,  0.0005, 0.10, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.400, 0.750, 0.120, 0.000, 1.220, 1.900,  2,  0.0002, 0.12, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.420, 0.780, 0.140, 0.000, 1.280, 2.050,  2, -0.0002, 0.14, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.440, 0.800, 0.160, 0.000, 1.320, 2.200,  1, -0.0004, 0.16, 0, "cognitive_overload", 0.0),

    # ── PHASE 3: Full rereading overload (p15–p24) — z_regress → 3.0 ─────────
    # Pace unavailable; actively rereading with low-to-moderate idle.
    ( 16, 1.00, _N, 0.380, 0.800, 0.190, 0.000, 1.350, 2.400,  2, -0.0006, 0.18, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.360, 0.820, 0.210, 0.000, 1.320, 2.600,  2, -0.0008, 0.20, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.350, 0.820, 0.230, 0.000, 1.280, 2.780,  2, -0.0010, 0.22, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.320, 0.780, 0.250, 0.000, 1.420, 2.500,  3, -0.0012, 0.20, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.280, 0.750, 0.270, 0.000, 1.500, 2.200,  3, -0.0015, 0.18, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.250, 0.720, 0.280, 0.000, 1.550, 2.000,  4, -0.0018, 0.15, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.280, 0.760, 0.250, 0.000, 1.480, 2.350,  3, -0.0012, 0.18, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.320, 0.800, 0.220, 0.000, 1.380, 2.550,  2, -0.0008, 0.20, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.280, 0.780, 0.250, 0.000, 1.420, 2.400,  3, -0.0010, 0.20, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.300, 0.800, 0.230, 0.000, 1.420, 2.400,  2, -0.0010, 0.20, 0, "cognitive_overload", 0.0),

    # ── PHASE 4: Acute episode 1 — thrashing on 1–2 paragraphs (p25–p33) ────
    # Triple-max: z_stag=2.2–3.0, z_burst=2.0–5.0(cap), z_idle=2.5–3.0+(cap),
    # z_pause=3.0(cap).  stag_damp suppresses velocity fallback here (correct).
    ( 16, 1.00, _N, 0.720, 0.880, 0.060, 0.000, 2.000, 4.500,  1,  0.0002, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.800, 0.920, 0.040, 0.000, 2.500, 5.500,  1,  0.0001, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.860, 0.950, 0.030, 0.000, 3.000, 6.500,  1,  0.0001, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.900, 1.000, 0.020, 0.000, 3.500, 7.500,  1,  0.0000, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.920, 1.000, 0.015, 0.000, 3.200, 8.000,  1,  0.0000, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.850, 0.950, 0.025, 0.000, 2.800, 6.500,  1,  0.0001, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.780, 0.900, 0.035, 0.000, 2.500, 5.500,  1,  0.0002, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.720, 0.850, 0.045, 0.000, 2.000, 4.500,  2,  0.0003, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.650, 0.800, 0.055, 0.000, 1.800, 3.800,  2,  0.0004, 0.00, 0, "cognitive_overload", 0.0),

    # ── PHASE 5: Wide-scope frantic rereading (p34–p41) — velocity fallback ──
    # Low idle → idle_damp=1.0; low stagnation → stag_damp=1.0 → z_skim fires.
    # sv_override > 0.0075×1.6=0.012: velocity fallback triggers z_skim 2.8–3.0.
    ( 16, 1.00, _N, 0.220, 0.480, 0.320, 0.000, 1.600, 1.200,  6, -0.0025, 0.00, 0, "cognitive_overload", 0.018),
    ( 16, 1.00, _N, 0.200, 0.450, 0.350, 0.000, 1.800, 1.050,  7, -0.0032, 0.00, 0, "cognitive_overload", 0.020),
    ( 16, 1.00, _N, 0.180, 0.420, 0.380, 0.000, 1.750, 0.950,  7, -0.0038, 0.00, 0, "cognitive_overload", 0.022),
    ( 16, 1.00, _N, 0.160, 0.400, 0.420, 0.000, 1.900, 0.850,  8, -0.0045, 0.00, 0, "cognitive_overload", 0.024),
    ( 16, 1.00, _N, 0.180, 0.420, 0.400, 0.000, 1.850, 0.900,  7, -0.0040, 0.00, 0, "cognitive_overload", 0.023),
    ( 16, 1.00, _N, 0.200, 0.450, 0.350, 0.000, 1.700, 1.050,  6, -0.0030, 0.00, 0, "cognitive_overload", 0.020),
    ( 16, 1.00, _N, 0.220, 0.470, 0.300, 0.000, 1.600, 1.180,  6, -0.0022, 0.00, 0, "cognitive_overload", 0.017),
    ( 16, 1.00, _N, 0.250, 0.500, 0.280, 0.000, 1.500, 1.350,  5, -0.0018, 0.00, 0, "cognitive_overload", 0.016),

    # ── PHASE 6: Partial panel recovery (p42–p47) ────────────────────────────
    ( 16, 0.82, _S, 0.380, 0.750, 0.180, 0.000, 1.280, 2.200,  2, -0.0005, 0.20, 0, "cognitive_overload", 0.0),
    ( 16, 0.78, _S, 0.350, 0.720, 0.140, 0.000, 1.200, 2.000,  3,  0.0003, 0.25, 0, "cognitive_overload", 0.0),
    ( 16, 0.80, _S, 0.320, 0.680, 0.100, 0.000, 1.150, 1.750,  3,  0.0008, 0.30, 0, "focused",          0.0),
    ( 16, 0.82, _S, 0.300, 0.650, 0.080, 0.000, 1.100, 1.600,  4,  0.0012, 0.32, 1, "focused",          0.0),
    ( 16, 0.85, _S, 0.310, 0.680, 0.100, 0.000, 1.120, 1.700,  3,  0.0010, 0.28, 0, "focused",          0.0),
    ( 16, 0.88, _S, 0.330, 0.720, 0.120, 0.000, 1.150, 1.850,  3,  0.0008, 0.22, 0, "cognitive_overload", 0.0),

    # ── PHASE 7: Second overload wave (p48–p57) ──────────────────────────────
    ( 16, 0.85, _S, 0.340, 0.700, 0.090, 0.000, 1.100, 1.600,  3,  0.0005, 0.10, 0, "cognitive_overload", 0.0),
    ( 16, 0.82, _S, 0.380, 0.740, 0.120, 0.000, 1.180, 1.800,  2,  0.0002, 0.12, 0, "cognitive_overload", 0.0),
    ( 16, 0.78, _S, 0.420, 0.780, 0.150, 0.000, 1.220, 2.000,  2, -0.0003, 0.15, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.450, 0.820, 0.180, 0.000, 1.280, 2.200,  2, -0.0006, 0.18, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.480, 0.840, 0.210, 0.000, 1.320, 2.450,  1, -0.0008, 0.20, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.500, 0.860, 0.230, 0.000, 1.380, 2.600,  1, -0.0010, 0.22, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.520, 0.880, 0.250, 0.000, 1.420, 2.800,  1, -0.0012, 0.22, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.480, 0.850, 0.220, 0.000, 1.380, 2.500,  1, -0.0009, 0.20, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.440, 0.820, 0.200, 0.000, 1.320, 2.300,  2, -0.0007, 0.18, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.400, 0.800, 0.180, 0.000, 1.280, 2.150,  2, -0.0005, 0.16, 0, "cognitive_overload", 0.0),

    # ── PHASE 8: Second frantic high-speed regression (p58–p65) ─────────────
    # Faster than phase 5 — racing backward over 7–9 paragraphs.
    # Low idle + low stag → both dampeners = 1.0 → velocity fallback fires fully.
    ( 16, 1.00, _N, 0.180, 0.420, 0.380, 0.000, 1.900, 0.950,  7, -0.0038, 0.00, 0, "cognitive_overload", 0.020),
    ( 16, 1.00, _N, 0.160, 0.400, 0.420, 0.000, 2.000, 0.850,  8, -0.0048, 0.00, 0, "cognitive_overload", 0.023),
    ( 16, 1.00, _N, 0.150, 0.380, 0.450, 0.000, 2.100, 0.780,  8, -0.0055, 0.00, 0, "cognitive_overload", 0.025),
    ( 16, 1.00, _N, 0.140, 0.360, 0.480, 0.000, 2.200, 0.700,  9, -0.0062, 0.00, 0, "cognitive_overload", 0.027),
    ( 15, 1.00, _N, 0.140, 0.340, 0.500, 0.000, 2.150, 0.680,  9, -0.0068, 0.00, 0, "cognitive_overload", 0.028),
    ( 16, 1.00, _N, 0.150, 0.360, 0.460, 0.000, 2.050, 0.720,  8, -0.0058, 0.00, 0, "cognitive_overload", 0.026),
    ( 16, 1.00, _N, 0.170, 0.390, 0.420, 0.000, 1.950, 0.850,  7, -0.0048, 0.00, 0, "cognitive_overload", 0.023),
    ( 16, 1.00, _N, 0.200, 0.420, 0.380, 0.000, 1.850, 0.980,  7, -0.0038, 0.00, 0, "cognitive_overload", 0.020),

    # ── PHASE 9: Freeze-thrash alternation (p66–p73) ─────────────────────────
    # Odd packets: max burst + high stag (active thrash).
    # Even packets: max idle + max pause + max stag (complete freeze).
    # stag_damp suppresses velocity fallback on both sub-types (stag ≥ 0.88).
    ( 16, 1.00, _N, 0.580, 0.880, 0.060, 0.000, 2.800, 3.500,  2,  0.0003, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.880, 0.980, 0.010, 0.000, 0.200, 7.000,  1,  0.0001, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.620, 0.900, 0.040, 0.000, 3.000, 4.000,  1,  0.0002, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.920, 1.000, 0.008, 0.000, 0.150, 8.000,  1,  0.0000, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.640, 0.920, 0.035, 0.000, 3.200, 4.200,  1,  0.0001, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.950, 1.000, 0.005, 0.000, 0.100, 9.000,  1,  0.0000, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.600, 0.880, 0.045, 0.000, 2.500, 3.600,  2,  0.0002, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.850, 0.950, 0.015, 0.000, 0.250, 7.500,  1,  0.0001, 0.00, 0, "cognitive_overload", 0.0),

    # ── PHASE 10: Exhausted rereading (p74–p82) ──────────────────────────────
    # Elevated idle (tired), moderate regression, 2–3 paragraphs covered.
    ( 16, 1.00, _N, 0.600, 0.750, 0.150, 0.000, 1.380, 3.200,  3, -0.0008, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.650, 0.720, 0.180, 0.000, 1.420, 3.500,  3, -0.0012, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.680, 0.740, 0.200, 0.000, 1.380, 3.800,  2, -0.0010, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.700, 0.760, 0.180, 0.000, 1.350, 4.000,  2, -0.0008, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.680, 0.780, 0.160, 0.000, 1.320, 3.700,  2, -0.0006, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.650, 0.760, 0.140, 0.000, 1.280, 3.400,  3, -0.0004, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.620, 0.740, 0.120, 0.000, 1.250, 3.100,  3, -0.0002, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.580, 0.720, 0.100, 0.000, 1.220, 2.800,  3,  0.0002, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.550, 0.700, 0.080, 0.000, 1.200, 2.600,  3,  0.0005, 0.00, 0, "cognitive_overload", 0.0),

    # ── PHASE 11: Panel-assisted partial recovery (p83–p91) ──────────────────
    # Panel open for AI clarification; regression drops, idle retreats.
    # Packets p85–p89 reach "focused" as signals normalise under panel guidance.
    ( 16, 0.82, _S, 0.450, 0.720, 0.140, 0.000, 1.220, 2.200,  3,  0.0003, 0.22, 0, "cognitive_overload", 0.0),
    ( 16, 0.80, _S, 0.400, 0.680, 0.100, 0.000, 1.180, 2.000,  3,  0.0008, 0.28, 0, "cognitive_overload", 0.0),
    ( 16, 0.82, _S, 0.350, 0.650, 0.080, 0.000, 1.150, 1.800,  4,  0.0012, 0.32, 0, "focused",          0.0),
    ( 16, 0.80, _S, 0.320, 0.620, 0.060, 0.000, 1.120, 1.650,  4,  0.0016, 0.35, 1, "focused",          0.0),
    ( 16, 0.85, _S, 0.300, 0.600, 0.050, 0.000, 1.100, 1.550,  4,  0.0018, 0.30, 1, "focused",          0.0),
    ( 16, 0.88, _S, 0.320, 0.620, 0.060, 0.000, 1.100, 1.600,  4,  0.0015, 0.25, 0, "focused",          0.0),
    ( 16, 0.88, _S, 0.340, 0.650, 0.080, 0.000, 1.120, 1.750,  3,  0.0012, 0.20, 0, "focused",          0.0),
    ( 16, 0.85, _S, 0.360, 0.680, 0.100, 0.000, 1.150, 1.900,  3,  0.0008, 0.15, 0, "cognitive_overload", 0.0),
    ( 16, 0.82, _S, 0.380, 0.720, 0.120, 0.000, 1.200, 2.100,  3,  0.0005, 0.10, 0, "cognitive_overload", 0.0),

    # ── PHASE 12: Final acute episode (p92–p99) ──────────────────────────────
    # After the panel closes, overload surges again into a triple-max finish.
    ( 16, 1.00, _N, 0.450, 0.780, 0.160, 0.000, 1.350, 2.400,  2, -0.0005, 0.12, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.550, 0.850, 0.200, 0.000, 1.480, 2.800,  2, -0.0008, 0.10, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.650, 0.900, 0.220, 0.000, 2.000, 3.800,  1, -0.0008, 0.08, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.760, 0.950, 0.080, 0.000, 2.800, 5.000,  1,  0.0001, 0.05, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.820, 0.980, 0.050, 0.000, 3.000, 6.000,  1,  0.0001, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.880, 1.000, 0.030, 0.000, 3.200, 7.000,  1,  0.0000, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.800, 0.950, 0.040, 0.000, 2.500, 5.800,  1,  0.0001, 0.00, 0, "cognitive_overload", 0.0),
    ( 16, 1.00, _N, 0.720, 0.900, 0.055, 0.000, 2.200, 4.800,  2,  0.0002, 0.00, 0, "cognitive_overload", 0.0),
]


# ── Master sessions list ────────────────────────────────────────────────────────
# Add new entries here for future synthetic sessions.
# ── SESSION 211  user_300  focused / drifting with panel-assisted recovery ─────
# User 300: average reader (baseline 238 WPM, idle_ratio=0.35, sv_norm=0.008).
# Realistic reading session with three drift episodes triggered by tab-switching
# / attentional lapses (focus_loss_rate 0.05–0.35, high idle, high stagnation).
# In every episode the AI panel opens and helps the user re-engage:
#   panel_interaction_share rises during drift → focus_loss drops → pace returns.
# pace_ratio never exceeds 1.35 (<1.6 skim threshold) → z_skim=0 throughout.
# z_regress stays near 0 → zero cognitive_overload signal.
# Result: clear focused↔drifting cycling with panel recovery arc for the LLM.
#
# stag_mu for user_300: (16 + 0.5×5)/30 = 0.617  →  z_stag = (stag-0.617)/0.15
# z_idle mu=0.35 std=0.22; z_fl scale=0.05; z_burst mu=1.0 std=0.5
# z_pause mu=1.15 std=max(0.90,0.50)=0.90
#
# nb   pr     pa    idle   stag   regr   focl   burst  pause  para  pvel       panel  mrkr  target
_S211 = [
    # ── Phase 1: initial focused reading (p0–p11) ─────────────────────────────
    (  4, 1.10, _S, 0.190, 0.375, 0.022, 0.00, 0.82, 0.550,  4, 0.003800, 0.00, 0, "focused"),
    (  7, 1.18, _S, 0.220, 0.438, 0.025, 0.00, 0.79, 0.600,  4, 0.004100, 0.00, 0, "focused"),
    ( 10, 1.25, _S, 0.200, 0.375, 0.028, 0.00, 0.84, 0.555,  5, 0.004800, 0.00, 0, "focused"),
    ( 13, 1.28, _S, 0.180, 0.438, 0.025, 0.00, 0.80, 0.580,  4, 0.004500, 0.00, 1, "focused"),
    ( 15, 1.30, _S, 0.200, 0.500, 0.025, 0.00, 0.78, 0.650,  4, 0.004200, 0.00, 0, "focused"),
    ( 16, 1.22, _S, 0.240, 0.500, 0.028, 0.00, 0.83, 0.700,  3, 0.003800, 0.00, 0, "focused"),
    ( 15, 1.32, _S, 0.220, 0.438, 0.025, 0.00, 0.87, 0.650,  4, 0.004400, 0.00, 0, "focused"),
    ( 16, 1.35, _S, 0.200, 0.375, 0.022, 0.00, 0.80, 0.600,  5, 0.004900, 0.00, 1, "focused"),
    ( 15, 1.28, _S, 0.180, 0.438, 0.025, 0.00, 0.82, 0.580,  4, 0.004600, 0.00, 0, "focused"),
    ( 16, 1.20, _S, 0.210, 0.500, 0.028, 0.00, 0.85, 0.680,  3, 0.003900, 0.00, 0, "focused"),
    ( 15, 1.25, _S, 0.230, 0.438, 0.025, 0.00, 0.83, 0.650,  4, 0.004300, 0.00, 0, "focused"),
    ( 16, 1.30, _S, 0.220, 0.500, 0.025, 0.00, 0.86, 0.680,  4, 0.004200, 0.00, 1, "focused"),
    # ── Phase 2: first drift onset — tab-switching / attentional lapse (p12–p21)
    # p12: subtle pace slow-down, still below baseline idle — final focused packet
    ( 15, 1.10, _S, 0.300, 0.500, 0.028, 0.00, 0.92, 0.850,  3, 0.003100, 0.00, 0, "focused"),
    # p13–p21: focus_loss rises, idle climbs above baseline, stagnation up
    ( 16, 1.00, _N, 0.380, 0.562, 0.025, 0.05, 1.02, 1.150,  3, 0.002400, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.460, 0.625, 0.028, 0.10, 1.18, 1.520,  2, 0.001700, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.540, 0.688, 0.025, 0.15, 1.32, 1.880,  2, 0.001100, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.600, 0.625, 0.025, 0.20, 1.38, 2.150,  2, 0.000800, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.650, 0.750, 0.028, 0.25, 1.48, 2.450,  2, 0.000580, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.680, 0.688, 0.025, 0.25, 1.42, 2.620,  2, 0.000420, 0.06, 0, "drifting"),
    ( 16, 1.00, _N, 0.720, 0.812, 0.025, 0.30, 1.52, 2.850,  1, 0.000280, 0.13, 0, "drifting"),
    ( 15, 1.00, _N, 0.700, 0.750, 0.025, 0.28, 1.45, 2.700,  2, 0.000350, 0.20, 0, "drifting"),
    ( 16, 1.00, _N, 0.650, 0.688, 0.025, 0.22, 1.35, 2.450,  2, 0.000480, 0.26, 0, "drifting"),
    # ── Phase 3: panel-assisted recovery (p22–p32) ─────────────────────────────
    # panel peaks, focus_loss fades, idle drops, pace returns
    ( 15, 1.00, _N, 0.580, 0.625, 0.025, 0.15, 1.22, 2.150,  2, 0.000800, 0.32, 0, "drifting"),
    ( 16, 1.00, _N, 0.500, 0.562, 0.022, 0.10, 1.12, 1.880,  3, 0.001300, 0.32, 0, "drifting"),
    ( 15, 1.15, _S, 0.420, 0.500, 0.022, 0.05, 1.05, 1.550,  3, 0.002100, 0.26, 0, "focused"),
    ( 16, 1.20, _S, 0.350, 0.438, 0.025, 0.00, 0.95, 1.200,  3, 0.002900, 0.20, 0, "focused"),
    ( 15, 1.25, _S, 0.300, 0.438, 0.025, 0.00, 0.90, 1.000,  4, 0.003500, 0.13, 0, "focused"),
    ( 16, 1.28, _S, 0.260, 0.375, 0.025, 0.00, 0.88, 0.820,  4, 0.003900, 0.06, 0, "focused"),
    ( 15, 1.30, _S, 0.240, 0.375, 0.025, 0.00, 0.85, 0.750,  4, 0.004100, 0.00, 0, "focused"),
    ( 16, 1.32, _S, 0.220, 0.438, 0.025, 0.00, 0.82, 0.680,  4, 0.004300, 0.00, 1, "focused"),
    ( 15, 1.35, _S, 0.200, 0.312, 0.022, 0.00, 0.79, 0.600,  5, 0.004800, 0.00, 1, "focused"),
    ( 16, 1.30, _S, 0.200, 0.375, 0.025, 0.00, 0.81, 0.620,  4, 0.004500, 0.00, 0, "focused"),
    ( 15, 1.28, _S, 0.220, 0.438, 0.028, 0.00, 0.83, 0.650,  4, 0.004400, 0.00, 0, "focused"),
    # ── Phase 4: sustained focused reading (p33–p46) ───────────────────────────
    ( 16, 1.25, _S, 0.240, 0.500, 0.028, 0.00, 0.86, 0.700,  4, 0.004100, 0.00, 0, "focused"),
    ( 15, 1.30, _S, 0.220, 0.438, 0.025, 0.00, 0.83, 0.650,  4, 0.004300, 0.00, 0, "focused"),
    ( 16, 1.35, _S, 0.200, 0.375, 0.022, 0.00, 0.80, 0.600,  5, 0.004700, 0.00, 0, "focused"),
    ( 15, 1.32, _S, 0.190, 0.438, 0.025, 0.00, 0.81, 0.620,  4, 0.004500, 0.00, 1, "focused"),
    ( 16, 1.28, _S, 0.210, 0.500, 0.025, 0.00, 0.84, 0.670,  4, 0.004200, 0.00, 0, "focused"),
    ( 15, 1.22, _S, 0.270, 0.562, 0.028, 0.00, 0.88, 0.770,  3, 0.003600, 0.00, 0, "focused"),
    ( 16, 1.18, _S, 0.300, 0.625, 0.028, 0.00, 0.91, 0.850,  3, 0.003200, 0.00, 0, "focused"),
    ( 15, 1.25, _S, 0.260, 0.500, 0.025, 0.00, 0.86, 0.700,  4, 0.004000, 0.00, 0, "focused"),
    ( 16, 1.28, _S, 0.230, 0.438, 0.025, 0.00, 0.83, 0.650,  4, 0.004300, 0.00, 0, "focused"),
    ( 15, 1.30, _S, 0.210, 0.375, 0.022, 0.00, 0.80, 0.600,  4, 0.004500, 0.00, 0, "focused"),
    ( 16, 1.25, _S, 0.240, 0.500, 0.025, 0.00, 0.85, 0.680,  3, 0.004000, 0.00, 0, "focused"),
    ( 15, 1.28, _S, 0.260, 0.438, 0.028, 0.00, 0.87, 0.720,  4, 0.004200, 0.00, 0, "focused"),
    ( 16, 1.30, _S, 0.220, 0.375, 0.025, 0.00, 0.82, 0.620,  4, 0.004400, 0.00, 1, "focused"),
    ( 15, 1.32, _S, 0.200, 0.312, 0.022, 0.00, 0.79, 0.580,  5, 0.004700, 0.00, 0, "focused"),
    # ── Phase 5: second drift — heavier episode (p47–p58) ─────────────────────
    # Deeper lapse: idle peaks 0.78, focl peaks 0.35, stagnation rises sharply
    ( 16, 1.00, _N, 0.360, 0.562, 0.028, 0.05, 1.05, 1.150,  3, 0.002500, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.460, 0.625, 0.025, 0.10, 1.20, 1.550,  2, 0.001600, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.570, 0.688, 0.028, 0.20, 1.32, 2.000,  2, 0.000980, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.650, 0.750, 0.025, 0.25, 1.48, 2.300,  2, 0.000620, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.720, 0.812, 0.025, 0.30, 1.58, 2.650,  1, 0.000310, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.780, 0.875, 0.028, 0.35, 1.72, 2.950,  1, 0.000180, 0.00, 0, "drifting"),
    # panel opens — user reaches out for AI help mid-drift
    ( 16, 1.00, _N, 0.750, 0.750, 0.025, 0.28, 1.58, 2.700,  2, 0.000380, 0.06, 0, "drifting"),
    ( 15, 1.00, _N, 0.700, 0.688, 0.025, 0.22, 1.45, 2.420,  2, 0.000600, 0.19, 0, "drifting"),
    ( 16, 1.00, _N, 0.640, 0.625, 0.022, 0.16, 1.32, 2.180,  2, 0.000900, 0.26, 0, "drifting"),
    ( 15, 1.00, _N, 0.560, 0.562, 0.022, 0.10, 1.18, 1.900,  3, 0.001300, 0.32, 0, "drifting"),
    # pace returns as user re-engages; panel still present
    ( 16, 1.12, _S, 0.480, 0.500, 0.022, 0.05, 1.08, 1.520,  3, 0.002000, 0.26, 0, "drifting"),
    ( 15, 1.18, _S, 0.400, 0.438, 0.025, 0.00, 0.98, 1.250,  3, 0.002600, 0.19, 0, "focused"),
    # ── Phase 6: recovery from second drift (p59–p68) ─────────────────────────
    ( 16, 1.22, _S, 0.350, 0.438, 0.025, 0.00, 0.93, 1.050,  3, 0.003200, 0.13, 0, "focused"),
    ( 15, 1.25, _S, 0.300, 0.375, 0.025, 0.00, 0.88, 0.920,  4, 0.003700, 0.06, 0, "focused"),
    ( 16, 1.28, _S, 0.260, 0.375, 0.025, 0.00, 0.85, 0.820,  4, 0.004100, 0.00, 1, "focused"),
    ( 15, 1.30, _S, 0.240, 0.438, 0.025, 0.00, 0.82, 0.720,  4, 0.004300, 0.00, 0, "focused"),
    ( 16, 1.32, _S, 0.220, 0.375, 0.022, 0.00, 0.80, 0.650,  5, 0.004600, 0.00, 0, "focused"),
    ( 15, 1.30, _S, 0.210, 0.375, 0.025, 0.00, 0.79, 0.620,  4, 0.004500, 0.00, 0, "focused"),
    ( 16, 1.28, _S, 0.220, 0.438, 0.025, 0.00, 0.82, 0.650,  4, 0.004300, 0.00, 0, "focused"),
    ( 15, 1.30, _S, 0.240, 0.500, 0.025, 0.00, 0.85, 0.700,  3, 0.004000, 0.00, 0, "focused"),
    ( 16, 1.28, _S, 0.230, 0.438, 0.025, 0.00, 0.83, 0.680,  4, 0.004200, 0.00, 0, "focused"),
    ( 15, 1.25, _S, 0.250, 0.500, 0.028, 0.00, 0.86, 0.720,  3, 0.003900, 0.00, 0, "focused"),
    # ── Phase 7: third drift — milder, faster panel response (p69–p80) ────────
    # Lighter episode; panel opens at second packet → shorter recovery arc
    ( 16, 1.10, _S, 0.320, 0.562, 0.025, 0.00, 0.92, 0.880,  3, 0.003000, 0.00, 0, "focused"),
    ( 15, 1.00, _N, 0.400, 0.625, 0.025, 0.05, 1.05, 1.250,  3, 0.002100, 0.00, 0, "drifting"),
    ( 16, 1.00, _N, 0.520, 0.688, 0.025, 0.12, 1.22, 1.680,  2, 0.001350, 0.00, 0, "drifting"),
    ( 15, 1.00, _N, 0.600, 0.625, 0.028, 0.18, 1.35, 2.050,  2, 0.000870, 0.06, 0, "drifting"),
    ( 16, 1.00, _N, 0.640, 0.688, 0.025, 0.20, 1.42, 2.250,  2, 0.000680, 0.20, 0, "drifting"),
    ( 15, 1.00, _N, 0.600, 0.625, 0.025, 0.15, 1.28, 2.000,  2, 0.000900, 0.26, 0, "drifting"),
    ( 16, 1.00, _N, 0.520, 0.562, 0.025, 0.10, 1.15, 1.720,  3, 0.001400, 0.32, 0, "drifting"),
    # recovery — pace re-establishes, panel tapers
    ( 15, 1.12, _S, 0.420, 0.500, 0.025, 0.05, 1.05, 1.350,  3, 0.002100, 0.26, 0, "focused"),
    ( 16, 1.18, _S, 0.350, 0.438, 0.025, 0.00, 0.95, 1.050,  3, 0.003000, 0.13, 0, "focused"),
    ( 15, 1.25, _S, 0.280, 0.375, 0.025, 0.00, 0.88, 0.800,  4, 0.003800, 0.06, 0, "focused"),
    ( 16, 1.28, _S, 0.250, 0.375, 0.022, 0.00, 0.85, 0.720,  4, 0.004100, 0.00, 0, "focused"),
    ( 15, 1.30, _S, 0.220, 0.438, 0.025, 0.00, 0.82, 0.650,  4, 0.004400, 0.00, 0, "focused"),
    # ── Phase 8: strong focused finish (p81–p99) ───────────────────────────────
    # Consistent focus, minor panel check at p85, overall improving engagement
    ( 16, 1.32, _S, 0.210, 0.375, 0.022, 0.00, 0.80, 0.600,  5, 0.004700, 0.00, 0, "focused"),
    ( 15, 1.35, _S, 0.190, 0.312, 0.022, 0.00, 0.78, 0.550,  5, 0.004900, 0.00, 0, "focused"),
    ( 16, 1.30, _S, 0.220, 0.375, 0.025, 0.00, 0.82, 0.620,  4, 0.004500, 0.00, 0, "focused"),
    ( 15, 1.28, _S, 0.240, 0.438, 0.025, 0.00, 0.85, 0.680,  4, 0.004200, 0.00, 0, "focused"),
    ( 16, 1.25, _S, 0.260, 0.500, 0.025, 0.00, 0.88, 0.720,  3, 0.003900, 0.06, 0, "focused"),
    ( 15, 1.28, _S, 0.230, 0.438, 0.025, 0.00, 0.84, 0.650,  4, 0.004300, 0.00, 0, "focused"),
    ( 16, 1.30, _S, 0.210, 0.375, 0.022, 0.00, 0.80, 0.600,  5, 0.004700, 0.00, 0, "focused"),
    ( 15, 1.32, _S, 0.190, 0.312, 0.020, 0.00, 0.77, 0.550,  5, 0.005000, 0.00, 1, "focused"),
    ( 16, 1.28, _S, 0.210, 0.375, 0.025, 0.00, 0.81, 0.600,  4, 0.004500, 0.00, 0, "focused"),
    ( 15, 1.25, _S, 0.230, 0.438, 0.025, 0.00, 0.84, 0.650,  4, 0.004200, 0.00, 0, "focused"),
    ( 16, 1.22, _S, 0.260, 0.500, 0.025, 0.00, 0.87, 0.720,  3, 0.003800, 0.00, 0, "focused"),
    ( 15, 1.25, _S, 0.230, 0.438, 0.025, 0.00, 0.83, 0.650,  4, 0.004100, 0.00, 0, "focused"),
    ( 16, 1.28, _S, 0.210, 0.375, 0.022, 0.00, 0.80, 0.600,  5, 0.004500, 0.00, 0, "focused"),
    ( 15, 1.30, _S, 0.200, 0.312, 0.022, 0.00, 0.78, 0.550,  5, 0.004800, 0.00, 1, "focused"),
    ( 16, 1.28, _S, 0.220, 0.375, 0.025, 0.00, 0.82, 0.620,  4, 0.004400, 0.00, 0, "focused"),
    ( 15, 1.25, _S, 0.240, 0.438, 0.025, 0.00, 0.85, 0.680,  4, 0.004100, 0.00, 0, "focused"),
    ( 16, 1.28, _S, 0.210, 0.375, 0.022, 0.00, 0.80, 0.600,  5, 0.004600, 0.00, 0, "focused"),
    ( 15, 1.30, _S, 0.190, 0.312, 0.020, 0.00, 0.78, 0.550,  5, 0.004900, 0.00, 1, "focused"),
    ( 16, 1.28, _S, 0.220, 0.438, 0.025, 0.00, 0.83, 0.650,  4, 0.004300, 0.00, 0, "focused"),
]


NEW_SESSIONS = [
    {
        "user_id":      248,
        "session_id":   179,
        "document_id":  190,
        "baseline_key": "user_248",
        "started_at":   "2026-03-20T09:00:00+00:00",
        "rows":         _S179,
    },
    {
        "user_id":      248,
        "session_id":   180,
        "document_id":  191,
        "baseline_key": "user_248",
        "started_at":   "2026-03-20T10:00:00+00:00",
        "rows":         _S180,
    },
    {
        "user_id":      248,
        "session_id":   181,
        "document_id":  192,
        "baseline_key": "user_248",
        "started_at":   "2026-03-20T11:00:00+00:00",
        "rows":         _S181,
    },
    {
        "user_id":      249,
        "session_id":   182,
        "document_id":  193,
        "baseline_key": "user_249",
        "started_at":   "2026-03-20T13:00:00+00:00",
        "rows":         _S182,
    },
    {
        "user_id":      249,
        "session_id":   183,
        "document_id":  194,
        "baseline_key": "user_249",
        "started_at":   "2026-03-20T14:00:00+00:00",
        "rows":         _S183,
    },
    {
        "user_id":      249,
        "session_id":   184,
        "document_id":  195,
        "baseline_key": "user_249",
        "started_at":   "2026-03-20T15:00:00+00:00",
        "rows":         _S184,
    },
    {
        "user_id":      260,
        "session_id":   200,
        "document_id":  200,
        "baseline_key": "user_260",
        "started_at":   "2026-04-01T09:20:00+00:00",
        "rows":         _S200,
    },
    {
        "user_id":      261,
        "session_id":   210,
        "document_id":  210,
        "baseline_key": "user_261",
        "started_at":   "2026-04-02T10:15:00+00:00",
        "rows":         _S210,
    },
    {
        "user_id":      300,
        "session_id":   211,
        "document_id":  220,
        "baseline_key": "user_300",
        "started_at":   "2026-04-03T09:30:00+00:00",
        "rows":         _S211,
    },
]


# ── File I/O helpers ────────────────────────────────────────────────────────────

def _existing_session_ids() -> set:
    """Return set of (user_id, session_id) already in unlabelled.jsonl."""
    seen = set()
    if not UNLABELLED.exists():
        return seen
    with open(UNLABELLED) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
                seen.add((row["user_id"], row["session_id"]))
            except Exception:
                pass
    return seen


def _load_baseline(key: str) -> dict:
    path = BASELINES_DIR / f"{key}_baseline.json"
    with open(path) as fh:
        data = json.load(fh)
    bl = data.get("baseline", data)
    bl["_updated_at"] = data.get("baseline_updated_at", "2026-03-20T08:00:00+00:00")
    return bl


def _row_to_params(row: tuple) -> dict:
    """Unpack a 14- or 15-element compact row tuple into a feature param dict.

    The optional 15th element ``sv_override`` sets scroll_velocity_norm_mean
    directly, overriding the default derivation in main().  Use it for phases
    where frantic backward scrolling should trigger the velocity-fallback z_skim
    signal (sv_override > SKIM_THRESHOLD × baseline.scroll_velocity_norm_mean).
    Pass 0.0 to use the default derivation.
    """
    if len(row) == 15:
        (nb, pr, pa, idle, stag, regr, focl, burst, pause,
         para, pvel, panel, markers, target, sv_override) = row
    else:
        (nb, pr, pa, idle, stag, regr, focl, burst, pause,
         para, pvel, panel, markers, target) = row
        sv_override = 0.0
    params: dict = {
        "n_batches":               nb,
        "pace_ratio":              pr,
        "pace_available":          pa,
        "idle_ratio_mean":         idle,
        "stagnation_ratio":        stag,
        "regress_rate_mean":       regr,
        "focus_loss_rate":         focl,
        "scroll_burstiness":       burst,
        "scroll_pause_mean":       pause,
        "paragraphs_observed":     para,
        "progress_velocity":       pvel,
        "panel_interaction_share": panel,
        "progress_markers_count":  markers,
        "_target":                 target,
    }
    if sv_override > 0.0:
        params["scroll_velocity_norm_mean"] = sv_override
    return params


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    existing = _existing_session_ids()
    baselines_cache: dict[str, dict] = {}

    new_unlabelled: list[str] = []
    new_labelled:   list[str] = []
    skipped = generated = 0

    for sess in NEW_SESSIONS:
        uid = sess["user_id"]
        sid = sess["session_id"]
        if (uid, sid) in existing:
            print(f"  SKIP  user={uid} session={sid} (already in unlabelled.jsonl)")
            skipped += 1
            continue

        bl_key = sess["baseline_key"]
        if bl_key not in baselines_cache:
            baselines_cache[bl_key] = _load_baseline(bl_key)
        baseline = baselines_cache[bl_key]
        baseline_ref = f"TrainingData/baselines/{bl_key}_baseline.json"

        doc_id      = sess["document_id"]
        started_at  = datetime.fromisoformat(sess["started_at"])

        # SESSION_START header line for unlabelled.jsonl
        header_meta = {
            "user_id":        uid,
            "session_id":     sid,
            "document_id":    doc_id,
            "session_mode":   "adaptive",
            "started_at":     sess["started_at"],
            "ended_at":       None,
            "protocol_tag":   None,
            "baseline_valid": True,
            "baseline_updated_at": baseline.get("_updated_at"),
            "baseline_ref":   baseline_ref,
            "append_started_at": sess["started_at"],
        }
        new_unlabelled.append(f"# SESSION_START {json.dumps(header_meta)}")

        for seq, raw_row in enumerate(sess["rows"]):
            p = _row_to_params(raw_row)
            target = p.pop("_target")

            # Derive scroll_velocity_norm_mean from pace_ratio + baseline
            bl_snorm = baseline.get("scroll_velocity_norm_mean", 0.010)
            if p["pace_available"] and p["pace_ratio"] > 1.0:
                p.setdefault("scroll_velocity_norm_mean",
                             round(bl_snorm * p["pace_ratio"] * 0.88, 4))
            else:
                p.setdefault("scroll_velocity_norm_mean",
                             round(bl_snorm * 0.75, 4))
            p.setdefault("scroll_velocity_norm_std",
                         round(baseline.get("scroll_velocity_norm_std", 0.015) * 0.9, 4))
            p.setdefault("scroll_jitter_mean",
                         round(baseline.get("scroll_jitter_mean", 0.02) *
                               (1.0 + _rng.uniform(-0.15, 0.15)), 5))
            p.setdefault("mouse_path_px_mean", 0.0)
            p.setdefault("mouse_efficiency_mean", 0.97)

            packet, z, conf = _build_packet(
                uid, sid, doc_id, seq, started_at, baseline, baseline_ref, p)
            labels, primary = _labels_from_z(z, target, conf)
            note = _make_note(z, packet["features"], conf)

            new_unlabelled.append(json.dumps(packet, ensure_ascii=False))
            label_row = {
                "session_id":    sid,
                "packet_seq":    seq,
                "window_end_at": packet["window_end_at"],
                "labels":        labels,
                "primary_state": primary,
                "notes":         note,
            }
            new_labelled.append(json.dumps(label_row, ensure_ascii=False))

        print(f"  GEN   user={uid} session={sid}  ({len(sess['rows'])} packets)")
        generated += 1

    if new_unlabelled:
        with open(UNLABELLED, "a") as fh:
            fh.write("\n" + "\n".join(new_unlabelled) + "\n")
        with open(LABELLED, "a") as fh:
            fh.write("\n".join(new_labelled) + "\n")
        print(f"\nAppended {len([l for l in new_labelled])} new packets "
              f"({generated} sessions) to unlabelled.jsonl + labelled.jsonl")
    else:
        print("\nNothing to append — all sessions already present.")

    if skipped:
        print(f"Skipped {skipped} session(s) already in unlabelled.jsonl.")


if __name__ == "__main__":
    main()
