import { apiRequest } from "./apiClient";

export interface Document {
  id: number;
  user_id: number;
  title: string;
  filename: string;
  file_size: number;
  uploaded_at: string;
}

export interface DocumentList {
  documents: Document[];
  total: number;
}

export interface ParseJobStatus {
  document_id: number;
  status: "pending" | "running" | "succeeded" | "failed" | "unknown";
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface Chunk {
  id: number;
  chunk_index: number;
  page_start: number | null;
  page_end: number | null;
  text: string;
  meta: Record<string, unknown>;
}

export interface AssetSummary {
  id: number;
  asset_type: string;
  page: number | null;
  bbox: Record<string, number> | null;
  meta: Record<string, unknown>;
}

export interface ParsedDocument {
  document_id: number;
  chunks: Chunk[];
  assets: AssetSummary[];
  total_chunks: number;
  offset: number;
  limit: number;
}

export const documentService = {
  async list(token: string): Promise<DocumentList> {
    return apiRequest<DocumentList>("/documents", { token });
  },

  async upload(token: string, title: string, file: File): Promise<Document> {
    const fd = new FormData();
    fd.append("title", title);
    fd.append("file", file);
    return apiRequest<Document>("/documents/upload", {
      method: "POST",
      token,
      formData: fd,
    });
  },

  async remove(token: string, docId: number): Promise<void> {
    return apiRequest<void>(`/documents/${docId}`, {
      method: "DELETE",
      token,
    });
  },

  async getParseStatus(token: string, docId: number): Promise<ParseJobStatus> {
    return apiRequest<ParseJobStatus>(`/documents/${docId}/parse-status`, { token });
  },

  async getParsed(
    token: string,
    docId: number,
    offset = 0,
    limit = 30,
  ): Promise<ParsedDocument> {
    return apiRequest<ParsedDocument>(
      `/documents/${docId}/parsed?offset=${offset}&limit=${limit}`,
      { token },
    );
  },

  async reparse(token: string, docId: number): Promise<ParseJobStatus> {
    return apiRequest<ParseJobStatus>(`/documents/${docId}/reparse`, {
      method: "POST",
      token,
    });
  },

  getAssetUrl(docId: number, assetId: number): string {
    return `http://localhost:8000/documents/${docId}/assets/${assetId}`;
  },
};
