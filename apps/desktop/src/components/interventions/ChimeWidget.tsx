/**
 * ChimeWidget
 *
 * Instant, slotless intervention.  Appears in the reader viewport as a small
 * floating toast — not in the panel.  Plays Chime.mp3 on mount, then
 * auto-dismisses after 5 seconds (or after the audio ends, whichever is first).
 *
 * A brief fade-out animation precedes the actual unmount so the exit is smooth.
 */

import React, { useEffect, useRef, useState } from "react";
import type { ActiveIntervention } from "../../services/interventionService";

// ─── Constants ────────────────────────────────────────────────────────────────

const DISPLAY_MS  = 5_000;
const FADE_OUT_MS = 600;

// ─── Keyframes (once at module load) ──────────────────────────────────────────

if (typeof document !== "undefined") {
  const ID = "chime-widget-kf";
  if (!document.getElementById(ID)) {
    const t = document.createElement("style");
    t.id = ID;
    t.textContent = `
      @keyframes chime-in  { from { opacity:0; transform:translateY(-8px) scale(0.95); }
                              to   { opacity:1; transform:translateY(0)    scale(1);    } }
      @keyframes chime-out { from { opacity:1; transform:translateY(0)    scale(1);    }
                              to   { opacity:0; transform:translateY(-8px) scale(0.95); } }
    `;
    document.head.appendChild(t);
  }
}

// ─── Props ────────────────────────────────────────────────────────────────────

export interface ChimeWidgetProps {
  intervention: ActiveIntervention;
  onDismiss:    (id: number) => void;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ChimeWidget({ intervention, onDismiss }: ChimeWidgetProps) {
  const id = intervention.intervention_id!;
  const [fadingOut, setFadingOut] = useState(false);
  const dismissedRef = useRef(false);

  const dismiss = () => {
    if (dismissedRef.current) return;
    dismissedRef.current = true;
    setFadingOut(true);
    setTimeout(() => onDismiss(id), FADE_OUT_MS);
  };

  // Play chime audio immediately on mount
  useEffect(() => {
    const audio = new Audio("/GamifiedIcons/Chime.mp3");
    audio.volume = 0.7;
    audio.play().catch(() => {/* autoplay blocked — still show the toast */});

    // Dismiss after DISPLAY_MS or when the audio ends
    const timeoutId = setTimeout(dismiss, DISPLAY_MS);
    audio.addEventListener("ended", () => {
      clearTimeout(timeoutId);
      setTimeout(dismiss, 1_500); // brief pause after the chime ends
    });

    return () => {
      clearTimeout(timeoutId);
      audio.pause();
      audio.src = "";
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div
      style={{
        ...styles.toast,
        animation: fadingOut
          ? `chime-out ${FADE_OUT_MS}ms ease forwards`
          : "chime-in 300ms ease",
      }}
      role="status"
      aria-live="polite"
    >
      <span style={styles.bell}>🔔</span>
      <div>
        <p style={styles.label}>FOCUS CHIME</p>
        <p style={styles.sub}>Stay with it — you've got this.</p>
      </div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles: Record<string, React.CSSProperties> = {
  toast: {
    // Anchored to the top-right of the reading content area.
    // The assistant panel is 490px wide on the right; we sit just inside
    // the content area's right edge, 16px clear of the panel border.
    position: "fixed",
    top: "72px",
    right: "506px",
    zIndex: 3000,
    display: "flex",
    alignItems: "center",
    gap: "10px",
    background: "#fffbeb",
    border: "1.5px solid #fcd34d",
    borderRadius: "12px",
    padding: "10px 16px 10px 12px",
    boxShadow: "0 4px 18px rgba(0,0,0,0.14)",
    pointerEvents: "none",
    userSelect: "none",
    minWidth: "190px",
    maxWidth: "260px",
  },
  bell: {
    fontSize: "22px",
    lineHeight: 1,
    flexShrink: 0,
  },
  label: {
    margin: 0,
    fontSize: "10px",
    fontWeight: 800,
    color: "#b45309",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
  },
  sub: {
    margin: "1px 0 0",
    fontSize: "11.5px",
    color: "#78350f",
    fontWeight: 500,
  },
};
