import { ApiError, apiRequest } from "./apiClient";

/**
 * Calibrated probability distribution over the four attentional states.
 * Values are floats in [0, 1] summing to 1.0, produced by the RF classifier.
 */
export interface AttentionalDistribution {
  focused: number;
  drifting: number;
  hyperfocused: number;
  cognitive_overload: number;
}

/**
 * Full attentional-state response from GET /sessions/{id}/attentional-state.
 *
 * intervention_context is the structured dict the future intervention LLM
 * will receive as part of its prompt.  It is available in every response so
 * the frontend can show ambiguity flags without extra round trips.
 */
export interface AttentionalState {
  session_id: number;
  packet_seq: number;
  classified_at: string;
  distribution: AttentionalDistribution;
  primary_state: "focused" | "drifting" | "hyperfocused" | "cognitive_overload";
  confidence: number;
  rationale: string;
  latency_ms: number;
  parse_ok: boolean;
  intervention_context: {
    primary_state: string;
    confidence: number;
    distribution: AttentionalDistribution;
    ambiguous: boolean;
  } | null;
}

export const classificationService = {
  /**
   * Fetch the latest attentional-state classification for a session.
   *
   * Returns null when:
   *  - The session is < 30 s old (no full window yet) → 404
   *  - The classifier is disabled → 503
   *  - Any network error
   *
   * These are all expected states during normal use, not errors.
   */
  async getAttentionalState(
    token: string,
    sessionId: number,
  ): Promise<AttentionalState | null> {
    try {
      return await apiRequest<AttentionalState>(
        `/sessions/${sessionId}/attentional-state`,
        { token },
      );
    } catch (err) {
      // 404 = no classification yet (window not full) — silent, not an error
      // 503 = classifier disabled — silent, not an error
      if (err instanceof ApiError && (err.status === 404 || err.status === 503)) {
        return null;
      }
      // Re-throw unexpected errors so callers can handle them
      throw err;
    }
  },
};

/** Human-readable label for each state. */
export const STATE_LABELS: Record<string, string> = {
  focused:            "Focused",
  drifting:           "Drifting",
  hyperfocused:       "Hyperfocused",
  cognitive_overload: "Overload",
};

/** Accent colour for each state (matches the design system palette). */
export const STATE_COLORS: Record<string, string> = {
  focused:            "#22c55e",   // green
  drifting:           "#eab308",   // yellow
  hyperfocused:       "#a855f7",   // purple
  cognitive_overload: "#ef4444",   // red
};
