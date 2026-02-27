import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, renderHook, screen, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AuthProvider, useAuth } from "../contexts/AuthContext";

// ── Mocks ──────────────────────────────────────────────────────────────────

const mockLogin = vi.fn();
const mockRegister = vi.fn();
const mockGetMe = vi.fn();

vi.mock("../services/authService", () => ({
  authService: {
    login: (...args: unknown[]) => mockLogin(...args),
    register: (...args: unknown[]) => mockRegister(...args),
    getMe: (...args: unknown[]) => mockGetMe(...args),
  },
}));

const FAKE_TOKEN = "fake.jwt.token";
const FAKE_USER = { id: 1, username: "tester", email: "t@test.com", created_at: "2024-01-01" };

// ── Helper: a simple consumer component ───────────────────────────────────

function Consumer() {
  const { user, token, isLoading, login, register, logout } = useAuth();
  return (
    <div>
      <span data-testid="loading">{String(isLoading)}</span>
      <span data-testid="token">{token ?? "null"}</span>
      <span data-testid="username">{user?.username ?? "null"}</span>
      <button onClick={() => login("tester", "pass123")}>login</button>
      <button onClick={() => register("newuser", "n@test.com", "pass123")}>register</button>
      <button onClick={logout}>logout</button>
    </div>
  );
}

function renderWithAuth() {
  return render(
    <AuthProvider>
      <Consumer />
    </AuthProvider>,
  );
}

// ── Tests ──────────────────────────────────────────────────────────────────

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
});

afterEach(() => {
  localStorage.clear();
});

describe("AuthContext — initial load", () => {
  it("starts with isLoading=true then resolves to unauthenticated when no token", async () => {
    mockGetMe.mockResolvedValue(FAKE_USER);
    renderWithAuth();

    // Loading resolves after effect completes
    await waitFor(() => {
      expect(screen.getByTestId("loading").textContent).toBe("false");
    });
    expect(screen.getByTestId("token").textContent).toBe("null");
    expect(screen.getByTestId("username").textContent).toBe("null");
  });

  it("restores session when a valid token is in localStorage", async () => {
    localStorage.setItem("lockin_token", FAKE_TOKEN);
    mockGetMe.mockResolvedValue(FAKE_USER);

    renderWithAuth();

    await waitFor(() =>
      expect(screen.getByTestId("username").textContent).toBe("tester"),
    );
    expect(screen.getByTestId("token").textContent).toBe(FAKE_TOKEN);
    expect(mockGetMe).toHaveBeenCalledWith(FAKE_TOKEN);
  });

  it("clears storage if the stored token is invalid (getMe rejects)", async () => {
    localStorage.setItem("lockin_token", "expired-token");
    mockGetMe.mockRejectedValue(new Error("401 Unauthorized"));

    renderWithAuth();

    await waitFor(() =>
      expect(screen.getByTestId("loading").textContent).toBe("false"),
    );
    expect(screen.getByTestId("token").textContent).toBe("null");
    expect(localStorage.getItem("lockin_token")).toBeNull();
  });
});

describe("AuthContext — login", () => {
  it("stores token in localStorage and populates user on successful login", async () => {
    mockLogin.mockResolvedValue({ access_token: FAKE_TOKEN, token_type: "bearer" });
    mockGetMe.mockResolvedValue(FAKE_USER);

    renderWithAuth();
    await waitFor(() =>
      expect(screen.getByTestId("loading").textContent).toBe("false"),
    );

    await act(async () => {
      await userEvent.click(screen.getByText("login"));
    });

    expect(localStorage.getItem("lockin_token")).toBe(FAKE_TOKEN);
    expect(screen.getByTestId("username").textContent).toBe("tester");
    expect(screen.getByTestId("token").textContent).toBe(FAKE_TOKEN);
  });

  it("propagates errors thrown by authService.login (hook-level test)", async () => {
    // No stored token → getMe is never called on mount
    mockLogin.mockRejectedValue(new Error("Invalid credentials"));

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <AuthProvider>{children}</AuthProvider>
    );
    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await expect(
      act(() => result.current.login("tester", "pass123")),
    ).rejects.toThrow("Invalid credentials");

    expect(localStorage.getItem("lockin_token")).toBeNull();
  });
});

describe("AuthContext — register", () => {
  it("stores token and populates user after successful registration", async () => {
    mockRegister.mockResolvedValue({ access_token: FAKE_TOKEN, token_type: "bearer" });
    mockGetMe.mockResolvedValue(FAKE_USER);

    renderWithAuth();
    await waitFor(() =>
      expect(screen.getByTestId("loading").textContent).toBe("false"),
    );

    await act(async () => {
      await userEvent.click(screen.getByText("register"));
    });

    expect(localStorage.getItem("lockin_token")).toBe(FAKE_TOKEN);
    expect(screen.getByTestId("username").textContent).toBe("tester");
  });
});

describe("AuthContext — logout", () => {
  it("clears token from state and localStorage", async () => {
    localStorage.setItem("lockin_token", FAKE_TOKEN);
    mockGetMe.mockResolvedValue(FAKE_USER);

    renderWithAuth();
    await waitFor(() =>
      expect(screen.getByTestId("username").textContent).toBe("tester"),
    );

    await act(async () => {
      await userEvent.click(screen.getByText("logout"));
    });

    expect(screen.getByTestId("token").textContent).toBe("null");
    expect(screen.getByTestId("username").textContent).toBe("null");
    expect(localStorage.getItem("lockin_token")).toBeNull();
  });
});
