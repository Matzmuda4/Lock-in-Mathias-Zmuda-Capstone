/**
 * ComprehensionCheckCard.tsx
 * ───────────────────────────
 * ╔══════════════════════════════════════════╗
 * ║  ✓  CHECK-IN              [Subtle]  [✕] ║
 * ╠══════════════════════════════════════════╣
 * ║  True or False:                         ║
 * ║  "The author argues that attention is   ║
 * ║   a single unified faculty."            ║
 * ║                                         ║
 * ║  ┌──────────┐   ┌──────────┐            ║
 * ║  │  TRUE    │   │  FALSE   │            ║
 * ║  └──────────┘   └──────────┘            ║
 * ╚══════════════════════════════════════════╝
 *
 * After the user answers:
 * ╔══════════════════════════════════════════╗
 * ║  ✓  CHECK-IN              [Subtle]  [✕] ║
 * ╠══════════════════════════════════════════╣
 * ║  ✅ Correct!  — OR — ❌ Not quite       ║
 * ║                                         ║
 * ║  "The author explicitly states that     ║
 * ║   attention is composed of distinct     ║
 * ║   sub-processes."                       ║
 * ║                                         ║
 * ║  ┌──────────────────────────────────┐   ║
 * ║  │           Continue  →            │   ║
 * ║  └──────────────────────────────────┘   ║
 * ╚══════════════════════════════════════════╝
 *
 * Interaction is click-only (TRUE / FALSE buttons).  No typing required.
 *
 * To edit the visual appearance of Comprehension Check cards, change this file only.
 */

import { useState, type CSSProperties } from "react";
import type {
  ActiveIntervention,
  ComprehensionCheckContent,
} from "../../services/interventionService";
import { TIER_TOKENS } from "./InterventionCardShell";
import { InterventionCardShell } from "./InterventionCardShell";

interface ComprehensionCheckCardProps {
  intervention: ActiveIntervention;
  onDismiss:    (id: number) => void;
}

export function ComprehensionCheckCard({ intervention, onDismiss }: ComprehensionCheckCardProps) {
  const content     = intervention.content as ComprehensionCheckContent | null;
  const question    = content?.question    ?? "Does the passage you just read support or challenge the main argument?";
  const correctAns  = content?.answer      ?? true;
  const explanation = content?.explanation;

  const tokens = TIER_TOKENS[intervention.tier] ?? TIER_TOKENS.subtle;

  const [selected, setSelected] = useState<boolean | null>(null);
  const answered  = selected !== null;
  const isCorrect = answered && selected === correctAns;

  // ── Styles ─────────────────────────────────────────────────────────────────

  const questionLabel: CSSProperties = {
    fontSize:     "11px",
    fontWeight:   700,
    color:        "var(--text-muted, #6b7280)",
    textTransform:"uppercase" as const,
    letterSpacing:"0.05em",
    margin:       "0 0 6px",
  };

  const questionText: CSSProperties = {
    fontSize:   "13px",
    lineHeight: 1.6,
    color:      "var(--text, #111827)",
    margin:     "0 0 16px",
  };

  const answerRow: CSSProperties = {
    display: "flex",
    gap:     "10px",
  };

  const makeTFBtn = (value: boolean): CSSProperties => {
    const isSelected = answered && selected === value;
    const isWrong    = isSelected && value !== correctAns;
    const isRight    = isSelected && value === correctAns;

    let bg     = "var(--bg, #f9fafb)";
    let border = "1.5px solid var(--border, #e5e7eb)";
    let color  = "var(--text, #111827)";

    if (isRight) { bg = "rgba(34,197,94,0.12)"; border = "1.5px solid #22c55e"; color = "#15803d"; }
    if (isWrong) { bg = "rgba(239,68,68,0.10)"; border = "1.5px solid #ef4444"; color = "#dc2626"; }

    return {
      flex:           1,
      padding:        "10px 0",
      background:     bg,
      border,
      borderRadius:   "8px",
      color,
      fontSize:       "13px",
      fontWeight:     700,
      cursor:         answered ? "default" : "pointer",
      transition:     "background 0.15s, border-color 0.15s",
    };
  };

  const feedbackRow: CSSProperties = {
    marginTop:  "14px",
    padding:    "10px 12px",
    background: isCorrect ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.07)",
    border:     `1px solid ${isCorrect ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.25)"}`,
    borderRadius: "8px",
  };

  const feedbackHeadline: CSSProperties = {
    fontSize:   "13px",
    fontWeight: 700,
    color:      isCorrect ? "#15803d" : "#dc2626",
    margin:     "0 0 (explanation ? 6px : 0)",
  };

  const feedbackExplain: CSSProperties = {
    fontSize:   "12px",
    lineHeight: 1.55,
    color:      "var(--text-muted, #6b7280)",
    margin:     "6px 0 0",
  };

  const continueBtn: CSSProperties = {
    width:          "100%",
    marginTop:      "14px",
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
  };

  return (
    <InterventionCardShell
      tier={intervention.tier}
      icon="✓"
      title="Check-In"
      onDismiss={() => onDismiss(intervention.intervention_id!)}
    >
      <p style={questionLabel}>True or False</p>
      <p style={questionText}>"{question}"</p>

      <div style={answerRow}>
        <button
          type="button"
          style={makeTFBtn(true)}
          disabled={answered}
          onClick={() => !answered && setSelected(true)}
        >
          TRUE
        </button>
        <button
          type="button"
          style={makeTFBtn(false)}
          disabled={answered}
          onClick={() => !answered && setSelected(false)}
        >
          FALSE
        </button>
      </div>

      {answered && (
        <div style={feedbackRow}>
          <p style={feedbackHeadline}>
            {isCorrect ? "✅ Correct!" : "❌ Not quite"}
          </p>
          {explanation && (
            <p style={feedbackExplain}>{explanation}</p>
          )}
        </div>
      )}

      {answered && (
        <button
          type="button"
          style={continueBtn}
          onClick={() => onDismiss(intervention.intervention_id!)}
        >
          Continue →
        </button>
      )}
    </InterventionCardShell>
  );
}
