const API_BASE = "http://localhost:8000";

type HttpMethod = "GET" | "POST" | "DELETE" | "PATCH" | "PUT";

interface RequestOptions {
  method?: HttpMethod;
  body?: unknown;
  token?: string | null;
  formData?: FormData;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Central fetch wrapper.
 * - Injects Bearer token when provided.
 * - Throws ApiError (with HTTP status) on non-2xx responses.
 * - Returns undefined for 204 No Content.
 */
export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { method = "GET", body, token, formData } = options;

  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (body && !formData) headers["Content-Type"] = "application/json";

  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: formData ?? (body ? JSON.stringify(body) : undefined),
  });

  if (!response.ok) {
    const errorBody = await response
      .json()
      .catch(() => ({ detail: response.statusText }));

    // FastAPI validation errors (422) return detail as an array of objects:
    // [{ loc: [...], msg: "...", type: "..." }]
    // All other errors return detail as a plain string.
    const detail = errorBody.detail;
    const message = Array.isArray(detail)
      ? detail.map((e: { msg?: string }) => e.msg ?? "Validation error").join(", ")
      : (detail ?? "Request failed");

    throw new ApiError(response.status, message);
  }

  if (response.status === 204) return undefined as unknown as T;
  return response.json() as Promise<T>;
}
