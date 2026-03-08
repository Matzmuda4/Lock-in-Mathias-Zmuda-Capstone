import { apiRequest } from "./apiClient";

export interface DriftState {
  session_id: number;
  beta_effective: number;
  attention_score: number;
  drift_score: number;
  drift_ema: number;
  confidence: number;
  baseline_used: boolean;
  updated_at: string;
}

export interface DriftDebug {
  session_id: number;
  user_id: number;
  baseline_used: boolean;
  beta_effective: number;
  beta_ema: number;
  beta_raw: number;
  attention_score: number;
  drift_score: number;
  drift_ema: number;
  confidence: number;
  n_batches_in_window: number;
  elapsed_minutes: number;
  beta_components: Record<string, number>;
  z_scores: Record<string, number>;
  features: Record<string, unknown>;
  baseline_snapshot: Record<string, unknown>;
  pace_ratio: number;
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
