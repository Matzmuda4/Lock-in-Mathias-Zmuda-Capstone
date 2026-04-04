/**
 * JourneyWidget — "The Walk of Words"
 *
 * Appears only when a `gamification` intervention fires (not by default).
 * Stays pinned in the panel for the rest of the session — no dismiss button.
 *
 * Layout: horizontal track at top; checkpoint icons hang below on stems.
 * The yellow user-dot and the fill line are always in the same coordinate
 * space, so they move in perfect sync.
 *
 * XP thresholds: 0 / 25 / 50 / 75 / 100
 * Per-window XP: focused ≈ 3–5 · hyperfocused ≈ 5–8 · badges add bonus
 */

import React from "react";
import type { ActiveIntervention, GamificationContent } from "../../services/interventionService";
import type { BadgeDef } from "../../types/badges";

// ─── Checkpoints ──────────────────────────────────────────────────────────────

const CHECKPOINTS = [
  { icon: "/GamifiedIcons/WakeupIcon.png",   label: "Start",          xp: 0   },
  { icon: "/GamifiedIcons/HikeIcon.png",     label: "Walking Along",  xp: 25  },
  { icon: "/GamifiedIcons/TreeIcon.webp",    label: "Tree of Life",   xp: 50  },
  { icon: "/GamifiedIcons/LakeIcon.png",     label: "Lake of Wisdom", xp: 75  },
  { icon: "/GamifiedIcons/MountainIcon.jpg", label: "Mountain Peak",  xp: 100 },
] as const;

const TOTAL     = CHECKPOINTS.length;
const MAX_XP    = CHECKPOINTS[TOTAL - 1].xp;

// ─── Props ────────────────────────────────────────────────────────────────────

export interface JourneyWidgetProps {
  intervention:  ActiveIntervention;
  xp:            number;
  sessionEnded?: boolean;
  /** Badges already earned this session — shown as mini-icons beside the title. */
  earnedBadges?: BadgeDef[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Index of the highest checkpoint the user has reached. */
function reachedIndex(xp: number, ended: boolean): number {
  if (ended) return TOTAL - 1;
  let idx = 0;
  for (let i = 0; i < TOTAL; i++) {
    if (xp >= CHECKPOINTS[i].xp) idx = i; else break;
  }
  return idx;
}

/** Fractional progress (0–1) within the current segment. */
function segFrac(xp: number, idx: number): number {
  if (idx >= TOTAL - 1) return 1;
  const lo = CHECKPOINTS[idx].xp;
  const hi = CHECKPOINTS[idx + 1].xp;
  return Math.min(1, Math.max(0, (xp - lo) / (hi - lo)));
}

// ─── Colour tokens ────────────────────────────────────────────────────────────

const ACCENT   = "#3b82f6";
const TRACK_BG = "#dbeafe";
const CARD_BG  = "#eff6ff";
const BORDER   = "#bfdbfe";
const CIRCLE_INACTIVE = "#e0f2fe";

// ─── Layout constants ─────────────────────────────────────────────────────────

// Half the icon circle diameter — used to pad the track container so the
// first and last circles never clip against the card edge.
const CIRCLE_R    = 21;   // px  (circle diameter = 42 px)
const TRACK_H     = 5;    // px
const STEM_H      = 12;   // px
const LABEL_H     = 20;   // px (approx, for total height calc)
const TRACK_Y     = 8;    // px from top of inner container to track centre

// Total height of the inner positioned container:
// track-top + track-height/2 + track-height/2 + stem + circle-diam + label
const INNER_H = TRACK_Y + TRACK_H + STEM_H + CIRCLE_R * 2 + LABEL_H + 4;

// ─── Component ────────────────────────────────────────────────────────────────

export default function JourneyWidget({
  intervention,
  xp,
  sessionEnded = false,
  earnedBadges = [],
}: JourneyWidgetProps) {
  const idx      = reachedIndex(xp, sessionEnded);
  const frac     = segFrac(xp, idx);
  const dispXP   = Math.min(xp, MAX_XP);
  const content  = intervention.content as GamificationContent | null;
  const badgeMsg = content?.message;

  // Track progress: 0 → 1 across all checkpoints.
  // All positions (fill, dot, checkpoint nodes) reference this single value.
  const trackProg = sessionEnded ? 1 : (idx + frac) / (TOTAL - 1);

  return (
    <div style={s.card}>

      {/* Header */}
      <div style={s.header}>
        <div>
          <span style={s.overline}>SESSION JOURNEY</span>
          <div style={s.titleRow}>
            <p style={s.title}>The Walk of Words</p>
            {/* Mini-badges pinned after title — up to 6 visible */}
            {earnedBadges.length > 0 && (
              <div style={s.miniBadgeRow} aria-label="Earned badges">
                {earnedBadges.slice(0, 6).map((b) => (
                  <img
                    key={b.id}
                    src={b.icon}
                    alt={b.label}
                    title={b.label}
                    style={s.miniBadge}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
        <span style={s.xpBadge}>{dispXP} XP</span>
      </div>

      {badgeMsg && <p style={s.badge}>🏅 {badgeMsg}</p>}

      <p style={s.sub}>Focused reading earns XP and moves you forward on the path.</p>

      {/* ── Trail ──
          Outer div has horizontal padding = CIRCLE_R so the first/last circles
          never overflow the card.  The inner div is position:relative so ALL
          child percentages share the same coordinate space.  The yellow dot
          and the fill bar are both expressed as `trackProg * 100%` of this
          inner div, so they always stay in sync.
      */}
      <div style={{ paddingLeft: CIRCLE_R, paddingRight: CIRCLE_R }}>
        <div style={{ position: "relative", height: INNER_H }}>

          {/* Track background */}
          <div style={s.trackBg} />

          {/* Fill — same coordinate space as dot */}
          <div style={{
            ...s.trackFill,
            width: `${trackProg * 100}%`,
          }} />

          {/* User dot */}
          <div style={{
            ...s.dot,
            left: `calc(${trackProg * 100}% - ${CIRCLE_R / 3}px)`,
          }} />

          {/* Checkpoint nodes */}
          {CHECKPOINTS.map((cp, i) => {
            const cpPct    = (i / (TOTAL - 1)) * 100;
            const reached  = i <= idx || sessionEnded;
            const active   = i === idx && !sessionEnded;

            return (
              <div
                key={cp.label}
                style={{ ...s.node, left: `${cpPct}%` }}
              >
                {/* Small pin on the track */}
                <div style={{
                  ...s.pin,
                  background: reached ? ACCENT : TRACK_BG,
                  border: `2px solid ${reached ? ACCENT : BORDER}`,
                }} />
                {/* Stem */}
                <div style={{ ...s.stem, background: reached ? ACCENT : BORDER }} />
                {/* Icon circle */}
                <div style={{
                  ...s.circle,
                  background:  reached ? ACCENT : CIRCLE_INACTIVE,
                  border:      `2.5px solid ${reached ? ACCENT : BORDER}`,
                  animation:   active ? "jp 1.6s ease-in-out infinite" : "none",
                }}>
                  <img
                    src={cp.icon}
                    alt={cp.label}
                    style={{
                      ...s.img,
                      filter: reached ? "brightness(0) invert(1)" : "opacity(0.4)",
                    }}
                  />
                </div>
                {/* Label */}
                <span style={s.label}>{cp.label}</span>
              </div>
            );
          })}

        </div>
      </div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const TRACK_TOP_PX = `${TRACK_Y}px`;
const TRACK_H_PX   = `${TRACK_H}px`;

const s: Record<string, React.CSSProperties> = {
  card: {
    background: CARD_BG,
    border: `1.5px solid ${BORDER}`,
    borderRadius: "14px",
    padding: "13px 12px 16px",
    marginBottom: "12px",
    userSelect: "none",
  },
  header: {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    marginBottom: "4px",
    gap: "8px",
  },
  overline: {
    display: "block",
    fontSize: "9.5px",
    fontWeight: 700,
    color: ACCENT,
    textTransform: "uppercase",
    letterSpacing: "0.08em",
  },
  titleRow: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    flexWrap: "wrap",
  },
  title: {
    margin: 0,
    fontSize: "14px",
    fontWeight: 700,
    color: "#1e3a5f",
    lineHeight: 1.2,
    flexShrink: 0,
  },
  miniBadgeRow: {
    display: "flex",
    alignItems: "center",
    gap: "4px",
    flexWrap: "nowrap",
  },
  miniBadge: {
    width: "26px",
    height: "26px",
    objectFit: "contain",
    flexShrink: 0,
  },
  xpBadge: {
    fontSize: "11px",
    fontWeight: 700,
    color: ACCENT,
    background: "#dbeafe",
    padding: "2px 8px",
    borderRadius: "20px",
    whiteSpace: "nowrap",
    flexShrink: 0,
  },
  badge: {
    margin: "0 0 4px",
    padding: "5px 9px",
    background: "#dbeafe",
    borderRadius: "7px",
    fontSize: "11px",
    color: "#1e40af",
    fontWeight: 600,
    lineHeight: 1.4,
  },
  sub: {
    margin: "0 0 12px",
    fontSize: "10.5px",
    color: "#64748b",
    lineHeight: 1.4,
  },

  // ── Track ──
  trackBg: {
    position: "absolute",
    top: TRACK_TOP_PX,
    left: 0,
    right: 0,
    height: TRACK_H_PX,
    background: TRACK_BG,
    borderRadius: "3px",
  },
  trackFill: {
    position: "absolute",
    top: TRACK_TOP_PX,
    left: 0,
    height: TRACK_H_PX,
    background: ACCENT,
    borderRadius: "3px",
    transition: "width 0.7s ease",
  },

  // ── User dot ──
  dot: {
    position: "absolute",
    top: `${TRACK_Y - 4}px`,
    width: "14px",
    height: "14px",
    background: "#f59e0b",
    border: "2.5px solid #fff",
    borderRadius: "50%",
    boxShadow: "0 0 0 2px #fbbf24",
    zIndex: 10,
    transition: "left 0.7s ease",
    pointerEvents: "none",
    willChange: "left",
  },

  // ── Checkpoint node ──
  node: {
    position: "absolute",
    top: `${TRACK_Y + TRACK_H}px`,
    transform: "translateX(-50%)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    zIndex: 5,
  },
  pin: {
    width: "10px",
    height: "10px",
    borderRadius: "50%",
    flexShrink: 0,
    marginTop: `-${Math.round(TRACK_H / 2) + 5}px`,
    position: "relative",
    zIndex: 6,
  },
  stem: {
    width: "2px",
    height: `${STEM_H}px`,
    flexShrink: 0,
  },
  circle: {
    width:  `${CIRCLE_R * 2}px`,
    height: `${CIRCLE_R * 2}px`,
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "background 0.4s ease",
    flexShrink: 0,
    willChange: "transform",
  },
  img: {
    width: "23px",
    height: "23px",
    objectFit: "contain",
  },
  label: {
    marginTop: "4px",
    fontSize: "9px",
    fontWeight: 600,
    color: "#475569",
    textAlign: "center",
    whiteSpace: "nowrap",
    lineHeight: 1.2,
  },
};

// ─── Keyframes (injected once) ────────────────────────────────────────────────

if (typeof document !== "undefined") {
  const ID = "journey-kf";
  if (!document.getElementById(ID)) {
    const t = document.createElement("style");
    t.id = ID;
    t.textContent = `@keyframes jp { 0%,100%{transform:scale(1)} 50%{transform:scale(1.13)} }`;
    document.head.appendChild(t);
  }
}
