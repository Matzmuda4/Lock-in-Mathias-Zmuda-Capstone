/**
 * interventionService.ts
 *
 * Typed API client for the intervention engine endpoints.
 *
 * Endpoints consumed:
 *   GET  /sessions/{id}/interventions/active   — all currently active interventions
 *   POST /sessions/{id}/interventions/manual   — dev/test: bypass LLM, fire any type
 *   POST /sessions/{id}/interventions/{id}/acknowledge — user dismissed an intervention
 *
 * The /trigger endpoint (full LLM pipeline) is called from the backend polling
 * loop — not directly from the frontend.  The frontend only polls /active to
 * discover what has been fired.
 */

import { apiRequest } from "./apiClient";

// ── Shared types ──────────────────────────────────────────────────────────────

export type InterventionTier = "subtle" | "moderate" | "strong" | "special";

export type InterventionType =
  | "re_engagement"
  | "focus_point"
  | "section_summary"
  | "comprehension_check"
  | "break_suggestion"
  | "gamification"
  | "chime"
  | "ambient_sound"
  | "text_reformat";

// ── Content shapes per type ───────────────────────────────────────────────────

export interface FocusPointContent {
  prompt: string;
}

export interface ReEngagementContent {
  headline: string;
  body:     string;
  cta:      string;
}

export interface ComprehensionCheckContent {
  type:         "true_false";
  question:     string;
  answer:       boolean;
  explanation?: string;
}

export interface SectionSummaryContent {
  title:     string;
  summary:   string;
  key_point: string;
}

export interface BreakSuggestionContent {
  headline:   string;
  body:       string;
  cta_take:   string;
  cta_skip:   string;
  auto_pause: boolean;
}

export interface GamificationContent {
  event_type:  string;
  badge_id:    string;
  xp_awarded:  number;
  message:     string;
}

export interface ChimeContent {
  sound: string;
  note?: string;
}

export interface AmbientSoundContent {
  sound:    string;
  profile?: string;
  note?:    string;
}

export interface TextReformatContent {
  mode:             string;
  note?:            string;
  revert_after_s?:  number;
}

/** Union of all possible content shapes. */
export type AnyInterventionContent =
  | FocusPointContent
  | ReEngagementContent
  | ComprehensionCheckContent
  | SectionSummaryContent
  | BreakSuggestionContent
  | GamificationContent
  | ChimeContent
  | AmbientSoundContent
  | TextReformatContent
  | Record<string, unknown>;

// ── API response types ────────────────────────────────────────────────────────

export interface ActiveIntervention {
  intervention_id: number;
  session_id:      number;
  intervene:       boolean;
  tier:            InterventionTier;
  type:            InterventionType | null;
  content:         AnyInterventionContent | null;
  latency_ms:      number;
  cooldown_status: string;
  fired_at:        string | null;
}

// ── Service ───────────────────────────────────────────────────────────────────

export const interventionService = {
  /**
   * Fetch all currently active (unacknowledged, non-expired) interventions.
   *
   * Returns an empty array when nothing is active — never throws on 404.
   * Silently swallows network errors so intervention polling never
   * crashes the reading session.
   */
  async getActive(
    token:     string,
    sessionId: number,
  ): Promise<ActiveIntervention[]> {
    try {
      return await apiRequest<ActiveIntervention[]>(
        `/sessions/${sessionId}/interventions/active`,
        { token },
      );
    } catch {
      return [];
    }
  },

  /**
   * Dev/test only — fire any intervention type directly, bypassing the LLM.
   * Uses canned template content unless ``content`` is provided.
   */
  async manualTrigger(
    token:     string,
    sessionId: number,
    type:      InterventionType,
    tier:      InterventionTier,
    content?:  AnyInterventionContent,
  ): Promise<ActiveIntervention> {
    return apiRequest<ActiveIntervention>(
      `/sessions/${sessionId}/interventions/manual`,
      {
        token,
        method: "POST",
        body:   { type, tier, ...(content ? { content } : {}) },
      },
    );
  },

  /**
   * Call the full LLM intervention pipeline for this session.
   * Returns the newly-fired intervention if the LLM decided to intervene,
   * or null if the LLM said no / cooldown blocked / engine gate rejected.
   *
   * Called by the frontend every ~30 s after the first RF classification
   * is available.  Non-critical — silently swallows all errors.
   */
  async trigger(
    token:     string,
    sessionId: number,
  ): Promise<ActiveIntervention | null> {
    try {
      const result = await apiRequest<ActiveIntervention>(
        `/sessions/${sessionId}/interventions/trigger`,
        { token, method: "POST" },
      );
      return result.intervene ? result : null;
    } catch {
      return null;
    }
  },

  /**
   * User has dismissed / acknowledged an intervention.
   * Frees its slot in the backend ActiveInterventionTracker.
   */
  async acknowledge(
    token:          string,
    sessionId:      number,
    interventionId: number,
  ): Promise<void> {
    try {
      await apiRequest<void>(
        `/sessions/${sessionId}/interventions/${interventionId}/acknowledge`,
        { token, method: "POST" },
      );
    } catch {
      // Acknowledge is best-effort — a failed call should never prevent
      // the user from dismissing the UI widget.
    }
  },
};
