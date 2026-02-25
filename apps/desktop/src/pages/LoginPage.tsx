import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

type Tab = "login" | "register";

export function LoginPage() {
  const { login, register } = useAuth();
  const navigate = useNavigate();

  const [tab, setTab] = useState<Tab>("login");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Shared field state
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  function switchTab(next: Tab) {
    setTab(next);
    setError(null);
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      if (tab === "login") {
        await login(username, password);
      } else {
        await register(username, email, password);
      }
      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        {/* Logo / brand */}
        <div className="login-brand">
          <span className="login-brand__dot" />
          <h1 className="login-brand__name">Lock‑In</h1>
          <p className="login-brand__tagline">Focused reading for curious minds</p>
        </div>

        {/* Tabs */}
        <div className="login-tabs">
          <button
            className={`login-tab ${tab === "login" ? "login-tab--active" : ""}`}
            onClick={() => switchTab("login")}
            type="button"
          >
            Sign in
          </button>
          <button
            className={`login-tab ${tab === "register" ? "login-tab--active" : ""}`}
            onClick={() => switchTab("register")}
            type="button"
          >
            Create account
          </button>
        </div>

        {/* Form */}
        <form className="login-form" onSubmit={handleSubmit} noValidate>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              autoComplete="username"
              placeholder="your_username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          {tab === "register" && (
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="you@university.edu"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          )}

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              autoComplete={tab === "login" ? "current-password" : "new-password"}
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {error && <p className="error-banner">{error}</p>}

          <button
            type="submit"
            className="btn btn--primary btn--full"
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <span className="spinner" />
            ) : tab === "login" ? (
              "Sign in"
            ) : (
              "Create account"
            )}
          </button>
        </form>
      </div>

      <style>{`
        .login-page {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg-base);
          padding: 24px;
        }

        .login-card {
          width: 100%;
          max-width: 400px;
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 36px 32px;
          display: flex;
          flex-direction: column;
          gap: 24px;
        }

        .login-brand {
          text-align: center;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
        }

        .login-brand__dot {
          display: block;
          width: 10px;
          height: 10px;
          background: var(--accent);
          border-radius: 50%;
          box-shadow: 0 0 12px var(--accent);
        }

        .login-brand__name {
          font-size: 26px;
          font-weight: 700;
          letter-spacing: -0.02em;
        }

        .login-brand__tagline {
          font-size: 13px;
          color: var(--text-muted);
        }

        .login-tabs {
          display: grid;
          grid-template-columns: 1fr 1fr;
          background: var(--bg-elevated);
          border-radius: var(--radius-sm);
          padding: 4px;
          gap: 4px;
        }

        .login-tab {
          background: none;
          border: none;
          border-radius: calc(var(--radius-sm) - 2px);
          color: var(--text-muted);
          font-size: 13px;
          font-weight: 500;
          padding: 8px;
          transition: background 0.15s, color 0.15s;
        }

        .login-tab--active {
          background: var(--bg-surface);
          color: var(--text);
        }

        .login-form {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
      `}</style>
    </div>
  );
}
