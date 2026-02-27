import { apiRequest } from "./apiClient";
import type { AssetSummary, Chunk } from "./documentService";

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

export interface SessionReaderData {
  session: Session;
  document_id: number;
  parse_status: string;
  chunks: Chunk[];
  assets: AssetSummary[];
  total_chunks: number;
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

  async pause(token: string, sessionId: number): Promise<Session> {
    return apiRequest<Session>(`/sessions/${sessionId}/pause`, {
      method: "POST",
      token,
    });
  },

  async resume(token: string, sessionId: number): Promise<Session> {
    return apiRequest<Session>(`/sessions/${sessionId}/resume`, {
      method: "POST",
      token,
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

  async getReader(
    token: string,
    sessionId: number,
    offset = 0,
    limit = 30,
  ): Promise<SessionReaderData> {
    return apiRequest<SessionReaderData>(
      `/sessions/${sessionId}/reader?offset=${offset}&limit=${limit}`,
      { token },
    );
  },
};
