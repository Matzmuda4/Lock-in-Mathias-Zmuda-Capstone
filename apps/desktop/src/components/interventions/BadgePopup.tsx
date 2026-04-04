/**
 * BadgePopup
 *
 * Full-screen (blurred backdrop) modal that appears when the user earns a badge.
 * Shows the badge icon large, the badge name, description, and XP reward.
 * A single "Okay!" button dismisses the popup — the badge is then miniaturised
 * and pinned to the Journey widget header.
 *
 * The popup never dismisses on backdrop click; the user must hit "Okay!"
 * (prevents accidental dismissal mid-reading).
 */

import React, { useEffect, useRef } from "react";
import type { BadgeDef } from "../../types/badges";

// ─── Keyframes (once at module load) ──────────────────────────────────────────

if (typeof document !== "undefined") {
  const ID = "badge-popup-kf";
  if (!document.getElementById(ID)) {
    const t = document.createElement("style");
    t.id = ID;
    t.textContent = `
      @keyframes bp-in  { from{opacity:0;transform:scale(0.85)} to{opacity:1;transform:scale(1)} }
      @keyframes bp-icon { 0%,100%{transform:scale(1) rotate(0deg)}
                           30%{transform:scale(1.18) rotate(-4deg)}
                           60%{transform:scale(1.12) rotate(3deg)} }
      .badge-ok-btn:hover { filter: brightness(1.08); }
    `;
    document.head.appendChild(t);
  }
}

// ─── Props ────────────────────────────────────────────────────────────────────

export interface BadgePopupProps {
  badge:       BadgeDef;
  onDismiss:   () => void;
}

// ─── Colours ──────────────────────────────────────────────────────────────────

const GOLD  = "#f59e0b";
const GOLD2 = "#fef3c7";
const GOLD3 = "#fcd34d";

// ─── Component ────────────────────────────────────────────────────────────────

export default function BadgePopup({ badge, onDismiss }: BadgePopupProps) {
  const btnRef = useRef<HTMLButtonElement>(null);

  // Auto-focus the Okay button so Enter/Space can dismiss
  useEffect(() => { btnRef.current?.focus(); }, []);

  return (
    <div style={styles.backdrop} role="dialog" aria-modal aria-label={`Badge earned: ${badge.label}`}>
      <div style={styles.modal}>

        {/* Badge icon — wobble animation */}
        <div style={styles.iconWrap}>
          <img
            src={badge.icon}
            alt={badge.label}
            style={styles.icon}
          />
        </div>

        <p style={styles.earned}>BADGE EARNED</p>
        <h2 style={styles.name}>{badge.label}</h2>
        <p style={styles.desc}>{badge.description}</p>

        <div style={styles.xpPill}>
          <span style={styles.xpStar}>⭐</span>
          <span style={styles.xpText}>+{badge.xp} XP</span>
        </div>

        <button
          ref={btnRef}
          className="badge-ok-btn"
          style={styles.okBtn}
          type="button"
          onClick={onDismiss}
        >
          Okay!
        </button>
      </div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles: Record<string, React.CSSProperties> = {
  backdrop: {
    position:       "fixed",
    inset:          0,
    zIndex:         9998,
    background:     "rgba(15, 23, 42, 0.55)",
    display:        "flex",
    alignItems:     "center",
    justifyContent: "center",
  },
  modal: {
    background:   "#fffbeb",
    border:       `2.5px solid ${GOLD3}`,
    borderRadius: "22px",
    padding:      "36px 32px 28px",
    maxWidth:     "380px",
    width:        "90%",
    textAlign:    "center",
    boxShadow:    "0 20px 60px rgba(0,0,0,0.22)",
    animation:    "bp-in 350ms cubic-bezier(0.34,1.56,0.64,1)",
  },
  iconWrap: {
    width:          "108px",
    height:         "108px",
    margin:         "0 auto 18px",
    borderRadius:   "50%",
    background:     GOLD2,
    border:         `3px solid ${GOLD3}`,
    display:        "flex",
    alignItems:     "center",
    justifyContent: "center",
    animation:      "bp-icon 0.8s ease 350ms both",
    overflow:       "hidden",
  },
  icon: {
    width:      "88px",
    height:     "88px",
    objectFit:  "contain",
  },
  earned: {
    margin:          "0 0 4px",
    fontSize:        "10.5px",
    fontWeight:      800,
    color:           GOLD,
    textTransform:   "uppercase",
    letterSpacing:   "0.1em",
  },
  name: {
    margin:     "0 0 8px",
    fontSize:   "22px",
    fontWeight: 800,
    color:      "#1c1917",
    lineHeight: 1.2,
  },
  desc: {
    margin:     "0 0 16px",
    fontSize:   "13px",
    color:      "#57534e",
    lineHeight: 1.6,
  },
  xpPill: {
    display:        "inline-flex",
    alignItems:     "center",
    gap:            "5px",
    background:     GOLD2,
    border:         `1.5px solid ${GOLD3}`,
    borderRadius:   "20px",
    padding:        "4px 14px",
    marginBottom:   "20px",
  },
  xpStar: {
    fontSize: "14px",
  },
  xpText: {
    fontSize:   "13px",
    fontWeight: 700,
    color:      "#92400e",
  },
  okBtn: {
    display:      "block",
    width:        "100%",
    padding:      "12px 20px",
    background:   GOLD,
    border:       "none",
    borderRadius: "10px",
    fontSize:     "15px",
    fontWeight:   800,
    color:        "#fff",
    cursor:       "pointer",
    letterSpacing:"0.02em",
    transition:   "filter 0.15s",
  },
};
