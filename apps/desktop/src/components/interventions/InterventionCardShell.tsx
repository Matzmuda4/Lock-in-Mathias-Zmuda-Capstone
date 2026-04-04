/**
 * InterventionCardShell.tsx
 * ─────────────────────────
 * Base wrapper used by every intervention card.
 *
 * Responsibilities:
 *  • Renders the card chrome: rounded border, tier-coloured accent, header row,
 *    dismiss button, and a content slot.
 *  • Owns ALL tier → colour mappings so individual cards never need to know
 *    about tiers.
 *
 * ── COLOUR RATIONALE (thesis-defensible) ────────────────────────────────────
 * Traffic-light metaphors in educational HCI (Hattie & Timperley, 2007) and
 * affective computing (Picard, 1997) support differentiated visual salience
 * based on urgency/severity.  Our four tiers map to:
 *
 *   subtle   → indigo  (same as the app accent — low urgency, reinforcing)
 *   moderate → amber   (caution — attention warranted, not critical)
 *   strong   → red     (re-engagement needed — high urgency)
 *   special  → purple  (hyperfocus / comprehension check — distinct context)
 *
 * The border is the primary signal; the header background is a 6% tint of the
 * same hue so the card is visually cohesive without being visually loud.
 * ─────────────────────────────────────────────────────────────────────────────
 */

import type { CSSProperties } from "react";
import type { InterventionTier } from "../../services/interventionService";

// ── Tier → visual tokens ──────────────────────────────────────────────────────

interface TierTokens {
  accent:     string;   // border & icon colour
  headerBg:   string;   // very light tint for the card header
  badgeBg:    string;   // pill background
  badgeColor: string;   // pill text colour
  label:      string;   // human-readable tier name
}

export const TIER_TOKENS: Record<string, TierTokens> = {
  subtle: {
    accent:     "#4f6ef7",
    headerBg:   "rgba(79,110,247,0.07)",
    badgeBg:    "rgba(79,110,247,0.13)",
    badgeColor: "#4f6ef7",
    label:      "Subtle",
  },
  moderate: {
    accent:     "#f59e0b",
    headerBg:   "rgba(245,158,11,0.07)",
    badgeBg:    "rgba(245,158,11,0.14)",
    badgeColor: "#b45309",
    label:      "Moderate",
  },
  strong: {
    accent:     "#ef4444",
    headerBg:   "rgba(239,68,68,0.07)",
    badgeBg:    "rgba(239,68,68,0.13)",
    badgeColor: "#dc2626",
    label:      "Strong",
  },
  special: {
    accent:     "#a855f7",
    headerBg:   "rgba(168,85,247,0.07)",
    badgeBg:    "rgba(168,85,247,0.13)",
    badgeColor: "#9333ea",
    label:      "Special",
  },
};

const defaultTokens: TierTokens = TIER_TOKENS.subtle;

// ── Prop types ────────────────────────────────────────────────────────────────

export interface InterventionCardShellProps {
  tier:       InterventionTier;
  icon:       string;             // emoji or small icon string
  title:      string;
  onDismiss:  () => void;
  children:   React.ReactNode;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function InterventionCardShell({
  tier,
  icon,
  title,
  onDismiss,
  children,
}: InterventionCardShellProps) {
  const tokens = TIER_TOKENS[tier] ?? defaultTokens;

  const card: CSSProperties = {
    background:   "var(--bg-surface, #ffffff)",
    border:       `1px solid var(--border, #e5e7eb)`,
    borderLeft:   `4px solid ${tokens.accent}`,
    borderRadius: "12px",
    overflow:     "hidden",
    marginBottom: "12px",
    boxShadow:    "0 1px 4px rgba(0,0,0,0.06)",
  };

  const header: CSSProperties = {
    display:        "flex",
    alignItems:     "center",
    justifyContent: "space-between",
    padding:        "10px 14px 10px 12px",
    background:     tokens.headerBg,
    borderBottom:   "1px solid var(--border, #e5e7eb)",
    gap:            "8px",
  };

  const headerLeft: CSSProperties = {
    display:    "flex",
    alignItems: "center",
    gap:        "8px",
    flex:       1,
    minWidth:   0,
  };

  const iconStyle: CSSProperties = {
    fontSize:   "15px",
    lineHeight: 1,
    flexShrink: 0,
  };

  const titleStyle: CSSProperties = {
    fontSize:     "12px",
    fontWeight:   700,
    color:        "var(--text, #111827)",
    whiteSpace:   "nowrap",
    overflow:     "hidden",
    textOverflow: "ellipsis",
  };

  const badge: CSSProperties = {
    display:      "inline-flex",
    alignItems:   "center",
    padding:      "1px 7px",
    background:   tokens.badgeBg,
    color:        tokens.badgeColor,
    borderRadius: "999px",
    fontSize:     "10px",
    fontWeight:   700,
    letterSpacing:"0.04em",
    textTransform:"uppercase" as const,
    flexShrink:   0,
  };

  const dismissBtn: CSSProperties = {
    background:   "none",
    border:       "none",
    cursor:       "pointer",
    color:        "var(--text-muted, #9ca3af)",
    fontSize:     "16px",
    lineHeight:   1,
    padding:      "0 2px",
    display:      "flex",
    alignItems:   "center",
    flexShrink:   0,
    opacity:      0.7,
  };

  const body: CSSProperties = {
    padding: "14px",
  };

  return (
    <div style={card} role="region" aria-label={`${title} intervention`}>
      <div style={header}>
        <div style={headerLeft}>
          <span style={iconStyle}>{icon}</span>
          <span style={titleStyle}>{title}</span>
          <span style={badge}>{tokens.label}</span>
        </div>
        <button
          type="button"
          style={dismissBtn}
          onClick={onDismiss}
          aria-label="Dismiss intervention"
          title="Dismiss"
        >
          ✕
        </button>
      </div>
      <div style={body}>{children}</div>
    </div>
  );
}
