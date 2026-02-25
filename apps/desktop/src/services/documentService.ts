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
};
