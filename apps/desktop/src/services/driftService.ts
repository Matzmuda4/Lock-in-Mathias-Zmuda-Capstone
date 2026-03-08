import { apiRequest } from "./apiClient";

export interface DriftState {
  session_id: number;
  // Primary bidirectional state
  drift_level: number;
  drift_ema: number;
  disruption_score: number;
  engagement_score: number;
  confidence: number;
  // Legacy / compat
  beta_effective: number;
  attention_score: number;
  drift_score: number;
  pace_ratio: number | null;
  pace_available: boolean;
  baseline_used: boolean;
  updated_at: string;
}

export interface DriftDebug {
  session_id: number;
  user_id: number;
  baseline_used: boolean;
  // Primary state
  drift_level: number;
  drift_ema: number;
  disruption_score: number;
  engagement_score: number;
  confidence: number;
  // Legacy compat
  beta_effective: number;
  beta_ema: number;
  attention_score: number;
  drift_score: number;
  // Window metadata
  n_batches_in_window: number;
  elapsed_minutes: number;
  // Component breakdown
  beta_components: Record<string, number>;
  z_scores: Record<string, number>;
  features: Record<string, unknown>;
  baseline_snapshot: Record<string, unknown>;
  // Pace details
  pace_ratio: number | null;
  pace_dev: number;
  pace_available: boolean;
  window_wpm_effective: number;
}

export const driftService = {
  async getDrift(token: string, sessionId: number): Promise<DriftState | null> {
    try {
      return await apiRequest<DriftState>(`/sessions/${sessionId}/drift`, { token });
    } catch {
      return null;
    }
  },

  async getDebug(token: string, sessionId: number): Promise<DriftDebug | null> {
    try {
      return await apiRequest<DriftDebug>(`/sessions/${sessionId}/drift/debug`, { token });
    } catch {
      return null;
    }
  },
};

/**
 * Map drift_ema to a colour category.
 * green  < 0.30
 * yellow 0.30–0.60
 * red    > 0.60
 */
export function driftColor(driftEma: number): "green" | "yellow" | "red" {
  if (driftEma < 0.30) return "green";
  if (driftEma < 0.60) return "yellow";
  return "red";
}
