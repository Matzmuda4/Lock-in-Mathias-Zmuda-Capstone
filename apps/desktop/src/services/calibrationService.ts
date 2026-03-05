import { apiRequest } from "./apiClient";

export interface CalibrationStatus {
  has_baseline: boolean;
  calib_available: boolean;
  /** "none" | "pending" | "running" | "succeeded" | "failed" */
  parse_status: string;
}

export interface CalibrationStartResponse {
  session_id: number;
  document_id: number;
}

export interface BaselineData {
  wpm_mean: number;
  wpm_std: number;
  scroll_velocity_mean: number;
  scroll_velocity_std: number;
  scroll_jitter_mean: number;
  idle_ratio_mean: number;
  regress_rate_mean: number;
  paragraph_dwell_mean: number;
  calibration_duration_seconds: number;
}

export interface CalibrationCompleteResponse {
  baseline: BaselineData;
  completed_at: string;
  session_id: number;
}

export interface UserBaselineResponse {
  user_id: number;
  baseline_json: BaselineData;
  completed_at: string;
  updated_at: string;
}

export const calibrationService = {
  async getStatus(token: string): Promise<CalibrationStatus> {
    return apiRequest<CalibrationStatus>("/calibration/status", { token });
  },

  async start(token: string): Promise<CalibrationStartResponse> {
    return apiRequest<CalibrationStartResponse>("/calibration/start", {
      method: "POST",
      token,
    });
  },

  async complete(
    token: string,
    sessionId: number,
  ): Promise<CalibrationCompleteResponse> {
    return apiRequest<CalibrationCompleteResponse>("/calibration/complete", {
      method: "POST",
      token,
      body: { session_id: sessionId },
    });
  },

  async getBaseline(token: string): Promise<UserBaselineResponse> {
    return apiRequest<UserBaselineResponse>("/calibration/baseline", { token });
  },
};
