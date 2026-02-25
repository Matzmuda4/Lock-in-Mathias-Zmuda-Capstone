import { describe, it, expect, vi, beforeEach } from "vitest";
import { apiRequest, ApiError } from "../services/apiClient";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function makeResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: () => Promise.resolve(body),
  };
}

beforeEach(() => {
  mockFetch.mockReset();
});

describe("apiRequest", () => {
  it("sends GET by default and returns parsed JSON", async () => {
    mockFetch.mockResolvedValue(makeResponse({ id: 1 }));

    const result = await apiRequest<{ id: number }>("/test");

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/test",
      expect.objectContaining({ method: "GET" }),
    );
    expect(result).toEqual({ id: 1 });
  });

  it("injects Authorization header when token is provided", async () => {
    mockFetch.mockResolvedValue(makeResponse({}));

    await apiRequest("/secure", { token: "my-jwt-token" });

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>)["Authorization"]).toBe(
      "Bearer my-jwt-token",
    );
  });

  it("does NOT inject Authorization header when token is null", async () => {
    mockFetch.mockResolvedValue(makeResponse({}));

    await apiRequest("/open", { token: null });

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(
      (init.headers as Record<string, string>)["Authorization"],
    ).toBeUndefined();
  });

  it("serialises body as JSON and sets Content-Type", async () => {
    mockFetch.mockResolvedValue(makeResponse({}));

    await apiRequest("/data", { method: "POST", body: { key: "value" } });

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe(
      "application/json",
    );
    expect(init.body).toBe(JSON.stringify({ key: "value" }));
  });

  it("throws ApiError with the status code on non-2xx responses", async () => {
    mockFetch.mockResolvedValue(makeResponse({ detail: "Not found" }, 404));

    await expect(apiRequest("/missing")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "Not found",
    });
  });

  it("returns undefined for 204 No Content", async () => {
    mockFetch.mockResolvedValue(makeResponse(undefined, 204));

    const result = await apiRequest("/delete-endpoint", { method: "DELETE" });
    expect(result).toBeUndefined();
  });
});
