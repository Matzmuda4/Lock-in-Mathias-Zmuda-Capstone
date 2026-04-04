/**
 * TextReformatBanner.tsx
 * ───────────────────────
 * Panel indicator shown when the LLM has activated adaptive text chunking.
 *
 * ┌─────────────────────────────────────────────────────────┐
 * │  📐  Adaptive Chunking Active          [Stop chunking]  │
 * │      Increased spacing to reduce visual density.        │
 * └─────────────────────────────────────────────────────────┘
 *
 * When the user clicks "Stop chunking", the banner dismisses (calls onStop)
 * which triggers acknowledge on the backend, removing the slot from the
 * ActiveInterventionTracker.  The reader-content CSS class is derived from
 * whether this intervention is present in activeInterventions, so removing it
 * automatically reverts the text layout to normal.
 *
 * To edit the visual appearance, change this file only.
 */

import type { CSSProperties } from "react";
import type {
  ActiveIntervention,
  TextReformatContent,
} from "../../services/interventionService";

interface TextReformatBannerProps {
  intervention: ActiveIntervention;
  onStop:       (id: number) => void;
}

export function TextReformatBanner({ intervention, onStop }: TextReformatBannerProps) {
  const content = intervention.content as TextReformatContent | null;
  const note    = content?.note ?? "Paragraph spacing increased to reduce visual density.";
  const mode    = content?.mode ?? "spaced";

  const modeLabel = mode === "chunked" ? "Chunked Reading" : "Adaptive Spacing";

  const banner: CSSProperties = {
    display:        "flex",
    alignItems:     "flex-start",
    gap:            "10px",
    padding:        "10px 12px",
    background:     "rgba(79,110,247,0.07)",
    border:         "1px solid rgba(79,110,247,0.2)",
    borderLeft:     "4px solid #4f6ef7",
    borderRadius:   "8px",
    marginBottom:   "8px",
  };

  const iconStyle: CSSProperties = {
    fontSize:   "16px",
    lineHeight: 1,
    flexShrink: 0,
    marginTop:  "1px",
  };

  const textCol: CSSProperties = {
    flex:    1,
    minWidth: 0,
  };

  const titleStyle: CSSProperties = {
    fontSize:   "12px",
    fontWeight: 700,
    color:      "#4f6ef7",
    margin:     "0 0 3px",
  };

  const noteStyle: CSSProperties = {
    fontSize:   "11px",
    color:      "var(--text-muted, #6b7280)",
    margin:     0,
    lineHeight: 1.4,
  };

  const stopBtn: CSSProperties = {
    flexShrink:   0,
    padding:      "5px 10px",
    background:   "none",
    border:       "1px solid rgba(79,110,247,0.35)",
    borderRadius: "6px",
    color:        "#4f6ef7",
    fontSize:     "11px",
    fontWeight:   600,
    cursor:       "pointer",
    whiteSpace:   "nowrap" as const,
  };

  return (
    <div style={banner} role="status" aria-label="Text reformat active">
      <span style={iconStyle}>📐</span>
      <div style={textCol}>
        <p style={titleStyle}>{modeLabel} Active</p>
        <p style={noteStyle}>{note}</p>
      </div>
      <button
        type="button"
        style={stopBtn}
        onClick={() => onStop(intervention.intervention_id!)}
        aria-label="Stop adaptive text chunking"
      >
        Stop
      </button>
    </div>
  );
}
