/**
 * BreakSuggestionOverlay
 *
 * Full-screen nuclear intervention.  When the LLM (or the dev menu) fires a
 * break_suggestion the overlay appears.
 *
 * Phase flow:
 *  "prompt"   → user sees suggestion, can confirm or skip
 *  "breaking" → 5-minute countdown, "I'm ready" button available at any time
 *  "resuming" → 10-second auto-resume warning
 *
 * IMPORTANT — the confirm action only pauses the session.  Acknowledging
 * the intervention (freeing the backend slot / starting the post-break
 * cooldown) happens in onAutoResume, so the overlay stays mounted for the
 * full duration of the break.
 *
 * Colour palette: calming green — deliberately low-stress.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import type { ActiveIntervention } from "../../services/interventionService";

// ─── Constants ────────────────────────────────────────────────────────────────

const BREAK_SECONDS    = 5 * 60;
const RESUME_COUNTDOWN = 10;

// ─── Props ────────────────────────────────────────────────────────────────────

export interface BreakSuggestionOverlayProps {
  intervention: ActiveIntervention;
  /** User confirmed the break — caller should pause the session only. */
  onConfirm:    (interventionId: number) => Promise<void>;
  /** User dismissed without taking a break — caller should acknowledge. */
  onDismiss:    (interventionId: number) => void;
  /** Break timer (or "I'm ready") finished — caller should acknowledge + resume. */
  onAutoResume: (interventionId: number) => void;
}

// ─── Helper ───────────────────────────────────────────────────────────────────

function fmt(s: number): string {
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

// ─── Colour tokens ────────────────────────────────────────────────────────────

const G600  = "#16a34a";   // green-600 — confirm / active
const G700  = "#15803d";   // green-700 — timer text
const G300  = "#86efac";   // green-300 — border
const G50   = "#f0fdf4";   // green-50  — card bg
const BLUE  = "#0284c7";   // sky-600   — "I'm ready" button
const MUTED = "#64748b";

// ─── Component ────────────────────────────────────────────────────────────────

export default function BreakSuggestionOverlay({
  intervention,
  onConfirm,
  onDismiss,
  onAutoResume,
}: BreakSuggestionOverlayProps) {
  const id = intervention.intervention_id!;

  const [phase,     setPhase]     = useState<"prompt" | "breaking" | "resuming">("prompt");
  const [remaining, setRemaining] = useState(BREAK_SECONDS);
  const [countdown, setCountdown] = useState(RESUME_COUNTDOWN);
  const [confirming, setConfirming] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
  }, []);

  // ── Start 5-minute break ──────────────────────────────────────────────────
  const handleConfirm = useCallback(async () => {
    setConfirming(true);
    await onConfirm(id);
    setConfirming(false);
    setPhase("breaking");
    intervalRef.current = setInterval(() => {
      setRemaining((r) => {
        if (r <= 1) { clearTimer(); setPhase("resuming"); return 0; }
        return r - 1;
      });
    }, 1000);
  }, [id, onConfirm, clearTimer]);

  // ── "I'm ready" — early manual resume ────────────────────────────────────
  const handleReady = useCallback(() => {
    clearTimer();
    onAutoResume(id);
  }, [id, onAutoResume, clearTimer]);

  // ── Auto-resume countdown ─────────────────────────────────────────────────
  useEffect(() => {
    if (phase !== "resuming") return;
    intervalRef.current = setInterval(() => {
      setCountdown((c) => {
        if (c <= 1) { clearTimer(); onAutoResume(id); return 0; }
        return c - 1;
      });
    }, 1000);
    return clearTimer;
  }, [phase, id, onAutoResume, clearTimer]);

  useEffect(() => () => clearTimer(), [clearTimer]);

  // Backdrop only dismisses during prompt phase
  const handleBackdrop = useCallback(() => {
    if (phase === "prompt") onDismiss(id);
  }, [phase, id, onDismiss]);

  const content = intervention.content as Record<string, string> | undefined;
  const headline = content?.headline ?? "You deserve a rest";
  const body     = content?.body     ?? "Step away from the screen. Close your eyes, stretch, or grab a glass of water.";

  return (
    <div style={styles.backdrop} onClick={handleBackdrop} role="dialog" aria-modal>
        <div style={styles.modal} onClick={(e) => e.stopPropagation()}>

          {/* Close — prompt only */}
          {phase === "prompt" && (
            <button
              className="break-close"
              style={styles.closeBtn}
              onClick={() => onDismiss(id)}
              type="button"
              aria-label="Dismiss break suggestion"
            >✕</button>
          )}

          {/* Icon */}
          <img src="/GamifiedIcons/BreakIcon.png" alt="Break" style={styles.icon} />

          {/* ── Prompt phase ── */}
          {phase === "prompt" && (
            <>
              <p style={styles.label}>BREAK SUGGESTED</p>
              <h2 style={styles.headline}>{headline}</h2>
              <p style={styles.body}>{body}</p>
              <div style={styles.btnCol}>
                <button
                  className="break-btn-confirm"
                  style={{ ...styles.btnBase, background: G600, color: "#fff" }}
                  type="button"
                  onClick={handleConfirm}
                  disabled={confirming}
                >
                  {confirming ? "Starting break…" : "Take a 5-Minute Break"}
                </button>
                <button
                  className="break-btn-skip"
                  style={{ ...styles.btnBase, background: "transparent", color: MUTED, border: `1.5px solid #e2e8f0` }}
                  type="button"
                  onClick={() => onDismiss(id)}
                >
                  Keep Reading
                </button>
              </div>
            </>
          )}

          {/* ── Breaking phase ── */}
          {phase === "breaking" && (
            <>
              <p style={styles.label}>BREAK IN PROGRESS</p>
              <div style={styles.timer}>{fmt(remaining)}</div>
              <p style={styles.body}>
                Rest up — step away from the screen.
              </p>
              <button
                className="break-btn-ready"
                style={{ ...styles.btnBase, background: BLUE, color: "#fff", marginTop: "4px" }}
                type="button"
                onClick={handleReady}
              >
                I'm ready to get back
              </button>
            </>
          )}

          {/* ── Resuming phase ── */}
          {phase === "resuming" && (
            <>
              <p style={styles.label}>BREAK COMPLETE</p>
              <div style={{ ...styles.timer, fontSize: "60px" }}>{countdown}</div>
              <p style={styles.body}>
                Resuming your session in {countdown} second{countdown !== 1 ? "s" : ""}…
              </p>
              <button
                className="break-btn-ready"
                style={{ ...styles.btnBase, background: BLUE, color: "#fff", marginTop: "4px" }}
                type="button"
                onClick={handleReady}
              >
                Resume now
              </button>
            </>
          )}

        </div>
      </div>
  );
}

// ─── Hover styles (injected once at module load, no re-render cost) ───────────

if (typeof document !== "undefined") {
  const ID = "break-overlay-styles";
  if (!document.getElementById(ID)) {
    const tag = document.createElement("style");
    tag.id = ID;
    tag.textContent = `
      .break-btn-confirm:hover { filter: brightness(1.08); }
      .break-btn-skip:hover    { background: #e9f5eb !important; }
      .break-btn-ready:hover   { filter: brightness(1.1); }
      .break-close:hover       { color: #475569 !important; }
    `;
    document.head.appendChild(tag);
  }
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles: Record<string, React.CSSProperties> = {
  backdrop: {
    position: "fixed",
    inset: 0,
    zIndex: 9999,
    background: "rgba(15, 23, 42, 0.6)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    // No backdropFilter — blur causes full GPU composite on every re-render
    // (the countdown ticks every second), making the whole UI stutter.
  },
  modal: {
    position: "relative",
    background: G50,
    borderRadius: "20px",
    padding: "40px 36px 32px",
    maxWidth: "420px",
    width: "90%",
    textAlign: "center",
    boxShadow: "0 24px 64px rgba(0,0,0,0.18)",
    border: `2px solid ${G300}`,
  },
  closeBtn: {
    position: "absolute",
    top: "14px",
    right: "16px",
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "18px",
    color: "#94a3b8",
    lineHeight: 1,
    padding: "4px 6px",
    borderRadius: "6px",
    transition: "color 0.15s",
  },
  icon: {
    width: "68px",
    height: "68px",
    objectFit: "contain",
    marginBottom: "14px",
    opacity: 0.8,
    filter: "hue-rotate(100deg) saturate(0.7)",  // tint to green
  },
  label: {
    margin: "0 0 6px",
    fontSize: "11px",
    fontWeight: 700,
    color: G600,
    textTransform: "uppercase",
    letterSpacing: "0.08em",
  },
  headline: {
    margin: "0 0 10px",
    fontSize: "20px",
    fontWeight: 700,
    color: "#1e293b",
    lineHeight: 1.3,
  },
  body: {
    margin: "0 0 20px",
    fontSize: "13px",
    color: MUTED,
    lineHeight: 1.6,
  },
  timer: {
    fontSize: "76px",
    fontWeight: 800,
    color: G700,
    lineHeight: 1,
    letterSpacing: "-2px",
    margin: "6px 0 14px",
    fontVariantNumeric: "tabular-nums",
  },
  btnCol: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },
  btnBase: {
    display: "block",
    width: "100%",
    padding: "12px 20px",
    border: "none",
    borderRadius: "10px",
    fontSize: "14px",
    fontWeight: 700,
    cursor: "pointer",
    transition: "filter 0.15s, background 0.15s",
    letterSpacing: "0.01em",
  },
};
