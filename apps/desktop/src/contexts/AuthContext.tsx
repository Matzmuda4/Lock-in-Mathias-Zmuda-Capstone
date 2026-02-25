import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import { authService, type User } from "../services/authService";

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
}

interface AuthContextValue extends AuthState {
  login: (username: string, password: string) => Promise<void>;
  register: (
    username: string,
    email: string,
    password: string,
  ) => Promise<void>;
  logout: () => void;
}

const TOKEN_KEY = "lockin_token";

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    token: localStorage.getItem(TOKEN_KEY),
    isLoading: true,
  });

  // On mount: validate the stored token and hydrate the user object.
  // If the token is expired or invalid, clear it silently.
  useEffect(() => {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) {
      setState((s) => ({ ...s, isLoading: false }));
      return;
    }
    authService
      .getMe(token)
      .then((user) => setState({ user, token, isLoading: false }))
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY);
        setState({ user: null, token: null, isLoading: false });
      });
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const { access_token } = await authService.login(username, password);
    localStorage.setItem(TOKEN_KEY, access_token);
    const user = await authService.getMe(access_token);
    setState({ user, token: access_token, isLoading: false });
  }, []);

  const register = useCallback(
    async (username: string, email: string, password: string) => {
      const { access_token } = await authService.register(
        username,
        email,
        password,
      );
      localStorage.setItem(TOKEN_KEY, access_token);
      const user = await authService.getMe(access_token);
      setState({ user, token: access_token, isLoading: false });
    },
    [],
  );

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setState({ user: null, token: null, isLoading: false });
  }, []);

  return (
    <AuthContext.Provider value={{ ...state, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within <AuthProvider>");
  return ctx;
}
