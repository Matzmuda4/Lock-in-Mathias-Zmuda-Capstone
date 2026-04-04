/**
 * SectionSummaryCard.tsx
 * ───────────────────────
 * Inline section-summary widget that appears INSIDE the reading content area
 * (not in the assistant panel) so it reads as part of the document flow.
 *
 * Layout:
 * ┌──────────────────────────────────────────────────────────────┐
 * │  📋 Section Recap         [Moderate]  [▼ collapse]  [✕ dismiss] │
 * ├──────────────────────────────────────────────────────────────┤
 * │  Quick Recap                                                  │
 * │                                                               │
 * │  Adaptive learning systems optimise content delivery based... │
 * │                                                               │
 * │  💡 Key Point                                                 │
 * │  Attention is composed of distinct sub-processes, each        │
 * │  subserved by partially separable neural networks.            │
 * └──────────────────────────────────────────────────────────────┘
 *
 * The card is collapsible — the header is always visible; the body
 * can be toggled with the ▼/▶ button so the user never has to
 * permanently dismiss it just to read past it.
 *
 * To edit the visual appearance of Section Summary cards, change this file only.
 *
 * Classification note: while this card is visible, the backend boosts P(focused)
 * via the section_summary_active flag injected into the classification packet.
 */

import { useState, type CSSProperties } from "react";
import type {
  ActiveIntervention,
  SectionSummaryContent,
} from "../../services/interventionService";
import { TIER_TOKENS } from "./InterventionCardShell";

interface SectionSummaryCardProps {
  intervention: ActiveIntervention;
  onDismiss:    (id: number) => void;
}

export function SectionSummaryCard({ intervention, onDismiss }: SectionSummaryCardProps) {
  const [collapsed, setCollapsed] = useState(false);

  const content   = intervention.content as SectionSummaryContent | null;
  const title     = content?.title     ?? "Section Recap";
  const summary   = content?.summary   ?? "";
  const keyPoint  = content?.key_point ?? "";
  const tokens    = TIER_TOKENS[intervention.tier] ?? TIER_TOKENS.moderate;

  // ── Styles ──────────────────────────────────────────────────────────────────

  const outer: CSSProperties = {
    // Width and horizontal margins are controlled by the
    // .section-summary-card CSS rule in ReaderPage so the card
    // matches .chunk exactly in both standard and adaptive layouts.
    border:       `1px solid ${tokens.accent}33`,
    borderLeft:   `4px solid ${tokens.accent}`,
    borderRadius: "10px",
    overflow:     "hidden",
    background:   tokens.headerBg,
    boxShadow:    "0 1px 6px rgba(0,0,0,0.06)",
    marginBottom: "28px",
  };

  const header: CSSProperties = {
    display:        "flex",
    alignItems:     "center",
    padding:        "10px 14px",
    gap:            "8px",
    cursor:         "pointer",
    userSelect:     "none" as const,
  };

  const icon: CSSProperties = {
    fontSize:   "14px",
    flexShrink: 0,
  };

  const headerLabel: CSSProperties = {
    fontSize:   "11px",
    fontWeight: 700,
    color:      tokens.accent,
    textTransform: "uppercase" as const,
    letterSpacing: "0.06em",
    flex:       1,
    whiteSpace: "nowrap",
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

  const iconBtn: CSSProperties = {
    background:  "none",
    border:      "none",
    cursor:      "pointer",
    color:       "var(--text-muted, #9ca3af)",
    fontSize:    "14px",
    padding:     "0 4px",
    lineHeight:  1,
    display:     "flex",
    alignItems:  "center",
    opacity:     0.75,
    flexShrink:  0,
  };

  const body: CSSProperties = {
    padding:    "0 16px 16px",
    borderTop:  `1px solid ${tokens.accent}22`,
  };

  const summaryTitle: CSSProperties = {
    fontSize:     "13px",
    fontWeight:   700,
    color:        "var(--text, #111827)",
    margin:       "14px 0 8px",
  };

  const summaryText: CSSProperties = {
    fontSize:   "13px",
    lineHeight: 1.7,
    color:      "var(--text-muted, #374151)",
    margin:     0,
  };

  const keyPointBox: CSSProperties = {
    marginTop:    "14px",
    padding:      "10px 14px",
    background:   `${tokens.accent}0f`,
    border:       `1px solid ${tokens.accent}2a`,
    borderRadius: "7px",
  };

  const keyPointLabel: CSSProperties = {
    fontSize:     "10px",
    fontWeight:   700,
    color:        tokens.accent,
    textTransform:"uppercase" as const,
    letterSpacing:"0.05em",
    margin:       "0 0 5px",
  };

  const keyPointText: CSSProperties = {
    fontSize:   "13px",
    lineHeight: 1.6,
    color:      "var(--text, #111827)",
    margin:     0,
  };

  return (
    <div style={outer} className="section-summary-card" role="region" aria-label="Section summary">
      {/* ── Header (always visible) ── */}
      <div style={header} onClick={() => setCollapsed((c) => !c)}>
        <span style={icon}>📋</span>
        <span style={headerLabel}>Section Recap</span>
        <span style={badge}>{tokens.label}</span>
        <button
          type="button"
          style={iconBtn}
          aria-label={collapsed ? "Expand summary" : "Collapse summary"}
          onClick={(e) => { e.stopPropagation(); setCollapsed((c) => !c); }}
        >
          {collapsed ? "▶" : "▼"}
        </button>
        <button
          type="button"
          style={iconBtn}
          aria-label="Dismiss summary"
          onClick={(e) => { e.stopPropagation(); onDismiss(intervention.intervention_id!); }}
        >
          ✕
        </button>
      </div>

      {/* ── Expandable body ── */}
      {!collapsed && (
        <div style={body}>
          <p style={summaryTitle}>{title}</p>
          <p style={summaryText}>{summary}</p>
          {keyPoint && (
            <div style={keyPointBox}>
              <p style={keyPointLabel}>💡 Key Point</p>
              <p style={keyPointText}>{keyPoint}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
