/**
 * ReEngagementCard.tsx
 * ─────────────────────
 * ╔══════════════════════════════════════════╗
 * ║  ✨  RE-ENGAGEMENT      [Moderate]  [✕] ║
 * ╠══════════════════════════════════════════╣
 * ║  Still with us?                         ║
 * ║                                         ║
 * ║  You've been on this section for a      ║
 * ║  while. Take a breath, re-read the      ║
 * ║  last sentence, and carry on.           ║
 * ║                                         ║
 * ║  ┌──────────────────────────────────┐   ║
 * ║  │           Got it  →              │   ║
 * ║  └──────────────────────────────────┘   ║
 * ╚══════════════════════════════════════════╝
 *
 * A direct prompt to pull the user back to the text.  The LLM generates
 * the headline, body, and CTA based on the current attentional signals.
 *
 * To edit the visual appearance of Re-engagement cards, change this file only.
 */

import type { CSSProperties } from "react";
import type {
  ActiveIntervention,
  ReEngagementContent,
} from "../../services/interventionService";
import { TIER_TOKENS } from "./InterventionCardShell";
import { InterventionCardShell } from "./InterventionCardShell";

interface ReEngagementCardProps {
  intervention: ActiveIntervention;
  onDismiss:    (id: number) => void;
}

export function ReEngagementCard({ intervention, onDismiss }: ReEngagementCardProps) {
  const content  = intervention.content as ReEngagementContent | null;
  const headline = content?.headline ?? "Take a moment";
  const body     = content?.body     ?? "Re-read the last paragraph and continue when you're ready.";
  const cta      = content?.cta      ?? "Got it";

  const tokens = TIER_TOKENS[intervention.tier] ?? TIER_TOKENS.moderate;

  const headlineStyle: CSSProperties = {
    fontSize:   "15px",
    fontWeight: 700,
    color:      "var(--text, #111827)",
    margin:     "0 0 8px",
  };

  const bodyStyle: CSSProperties = {
    fontSize:   "13px",
    lineHeight: 1.6,
    color:      "var(--text-muted, #6b7280)",
    margin:     "0 0 14px",
  };

  const ctaBtn: CSSProperties = {
    width:          "100%",
    padding:        "9px 14px",
    background:     tokens.badgeBg,
    border:         `1.5px solid ${tokens.accent}44`,
    borderRadius:   "8px",
    color:          tokens.accent,
    fontSize:       "13px",
    fontWeight:     600,
    cursor:         "pointer",
    display:        "flex",
    alignItems:     "center",
    justifyContent: "center",
    gap:            "6px",
    transition:     "background 0.15s",
  };

  return (
    <InterventionCardShell
      tier={intervention.tier}
      icon="✨"
      title="Re-engagement"
      onDismiss={() => onDismiss(intervention.intervention_id!)}
    >
      <p style={headlineStyle}>{headline}</p>
      <p style={bodyStyle}>{body}</p>
      <button
        type="button"
        style={ctaBtn}
        onClick={() => onDismiss(intervention.intervention_id!)}
      >
        {cta} →
      </button>
    </InterventionCardShell>
  );
}
