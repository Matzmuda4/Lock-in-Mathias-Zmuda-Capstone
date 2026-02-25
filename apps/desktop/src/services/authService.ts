import { ApiError, apiRequest } from "./apiClient";

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
  created_at: string;
}

export const authService = {
  /**
   * Login uses form-encoded body (OAuth2PasswordRequestForm on the backend)
   * so Swagger's Authorize button works. We handle this explicitly here.
   */
  async login(username: string, password: string): Promise<TokenResponse> {
    const response = await fetch("http://localhost:8000/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ username, password }),
    });
    if (!response.ok) {
      const err = await response
        .json()
        .catch(() => ({ detail: "Login failed" }));
      throw new ApiError(response.status, err.detail);
    }
    return response.json();
  },

  async register(
    username: string,
    email: string,
    password: string,
  ): Promise<TokenResponse> {
    return apiRequest<TokenResponse>("/auth/register", {
      method: "POST",
      body: { username, email, password },
    });
  },

  async getMe(token: string): Promise<User> {
    return apiRequest<User>("/auth/me", { token });
  },
};
