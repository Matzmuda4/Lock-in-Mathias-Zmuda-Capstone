import { apiRequest } from "./apiClient";

export interface TelemetryBatch {
  session_id: number;
  scroll_delta_sum: number;
  scroll_delta_abs_sum: number;
  scroll_event_count: number;
  scroll_direction_changes: number;
  scroll_pause_seconds: number;
  idle_seconds: number;
  mouse_path_px: number;
  mouse_net_px: number;
  window_focus_state: "focused" | "blurred";
  current_paragraph_id: string | null;
  current_chunk_index: number | null;
  viewport_progress_ratio: number;
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
