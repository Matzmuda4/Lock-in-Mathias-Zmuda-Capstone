import { apiRequest } from "./apiClient";

export type SessionMode = "baseline" | "adaptive";
export type SessionStatus = "active" | "paused" | "ended" | "completed";

export interface Session {
  id: number;
  user_id: number;
  document_id: number;
  name: string;
  mode: SessionMode;
  status: SessionStatus;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number | null;
  created_at: string;
}

export interface SessionList {
  sessions: Session[];
  total: number;
}

export const sessionService = {
  async list(token: string): Promise<SessionList> {
    return apiRequest<SessionList>("/sessions", { token });
  },

  async start(
    token: string,
    documentId: number,
    name: string,
    mode: SessionMode,
  ): Promise<Session> {
    return apiRequest<Session>("/sessions/start", {
      method: "POST",
      token,
      body: { document_id: documentId, name, mode },
    });
  },

  async end(token: string, sessionId: number): Promise<Session> {
    return apiRequest<Session>(`/sessions/${sessionId}/end`, {
      method: "POST",
      token,
    });
  },

  async complete(token: string, sessionId: number): Promise<Session> {
    return apiRequest<Session>(`/sessions/${sessionId}/complete`, {
      method: "POST",
      token,
    });
  },
};
