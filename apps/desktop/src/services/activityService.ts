import { apiRequest } from "./apiClient";

export interface TelemetryBatch {
  session_id: number;
  // ── Scroll ────────────────────────────────────────────────────────────────
  scroll_delta_sum: number;
  scroll_delta_abs_sum: number;
  scroll_event_count: number;
  scroll_direction_changes: number;
  scroll_pause_seconds: number;
  /** Sum of positive (downward) scroll deltas only — for regress_rate. */
  scroll_delta_pos_sum: number;
  /** Sum of absolute values of negative (upward) scroll deltas — for regress_rate. */
  scroll_delta_neg_sum: number;
  // ── Engagement ────────────────────────────────────────────────────────────
  /** Seconds idle IN THIS 2s window (0..2). Fix: no longer cumulative. */
  idle_seconds: number;
  /** Diagnostic: total seconds since last interaction (not used by model). */
  idle_since_interaction_seconds?: number;
  // ── Mouse ─────────────────────────────────────────────────────────────────
  mouse_path_px: number;
  mouse_net_px: number;
  // ── Focus ─────────────────────────────────────────────────────────────────
  window_focus_state: "focused" | "blurred";
  // ── Reading position ──────────────────────────────────────────────────────
  current_paragraph_id: string | null;
  current_chunk_index: number | null;
  viewport_progress_ratio: number;
  // ── Presentation profile (viewport dimensions for normalisation) ───────────
  viewport_height_px: number;
  viewport_width_px: number;
  reader_container_height_px: number;
  // ── Timestamp ─────────────────────────────────────────────────────────────
  client_timestamp: string;
}

export interface BatchResponse {
  id: number;
  session_id: number;
  event_type: string;
  created_at: string;
}

export const activityService = {
  /**
   * POST /activity/batch
   * Sends one aggregated 2-second telemetry batch for an active session.
   * Silently swallows network errors — telemetry must never crash the reader.
   */
  async postBatch(token: string, batch: TelemetryBatch): Promise<BatchResponse | null> {
    try {
      return await apiRequest<BatchResponse>("/activity/batch", {
        method: "POST",
        token,
        body: batch,
      });
    } catch {
      return null;
    }
  },
};
