import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { useTelemetry } from "../hooks/useTelemetry";
import { calibrationService, type BaselineData } from "../services/calibrationService";
import { sessionService, type Session } from "../services/sessionService";
import { apiRequest } from "../services/apiClient";

const DEV = (import.meta as unknown as { env?: { DEV?: boolean } }).env?.DEV ?? false;
const CALIB_MIN_SECONDS = 10;

interface CalibrationContent {
  paragraphs: string[];
  total_words: number;
}

// ─── Timer ────────────────────────────────────────────────────────────────────

function useElapsedSeconds(session: Session | null, running: boolean) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!session) return;
    const base = session.elapsed_seconds ?? 0;
    if (session.status === "active") {
      const ms = Date.now() - new Date(session.started_at).getTime();
      setElapsed(base + Math.max(0, Math.floor(ms / 1000)));
    } else {
      setElapsed(base);
    }
  }, [session]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [running]);

  return elapsed;
}

function fmt(s: number) {
  const m = Math.floor(s / 60).toString().padStart(2, "0");
  const sec = (s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function CalibrationReaderPage() {
  const { id } = useParams<{ id: string }>();
  const sessionId = Number(id);
  const { token } = useAuth();
  const navigate = useNavigate();

  const [session, setSession] = useState<Session | null>(null);
  const [paragraphs, setParagraphs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [calibDone, setCalibDone] = useState(false);
  const [baseline, setBaseline] = useState<BaselineData | null>(null);
  const contentRef = useRef<HTMLElement>(null);

  // Fetch session info (for timer) and text content (from the .txt file via backend)
  useEffect(() => {
    if (!token || !sessionId) return;

    // Session info for timer
    sessionService
      .list(token)
      .then(({ sessions }) => {
        const s = sessions.find((x) => x.id === sessionId);
        if (s) setSession(s);
        else setError("Session not found.");
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load session"));

    // Calibration text — read live from callibration.txt via the backend
    apiRequest<CalibrationContent>(`/calibration/session/${sessionId}`, { token })
      .then(({ paragraphs: ps }) => {
        if (ps && ps.length > 0) setParagraphs(ps);
      })
      .catch(() => {
        // Silently ignore — paragraphs stay empty (no text shown, error banner already handled above)
      });
  }, [token, sessionId]);

  const isActive = !calibDone && session?.status === "active";
  const elapsed = useElapsedSeconds(session, isActive);
  const canFinish = elapsed >= CALIB_MIN_SECONDS;

  const { lastBatch } = useTelemetry({
    sessionId,
    token,
    active: isActive,
    containerRef: contentRef,
  });

  const handleFinish = useCallback(async () => {
    if (!token) return;
    setBusy(true);
    try {
      const result = await calibrationService.complete(token, sessionId);
      setBaseline(result.baseline);
      setCalibDone(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to complete calibration");
    } finally {
      setBusy(false);
    }
  }, [token, sessionId]);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div style={s.page}>
      {/* Top bar */}
      <header style={s.topbar}>
        <div style={s.calibBadge}>◎&nbsp;&nbsp;Calibration Session</div>
        <span style={s.topbarTitle}>Reading Calibration</span>
        <div style={s.topbarRight}>
          {session && <span style={s.timer}>{fmt(elapsed)}</span>}
          <button
            style={{
              ...s.btn,
              opacity: !canFinish || busy || calibDone || !session ? 0.45 : 1,
              cursor: !canFinish || busy || calibDone || !session ? "not-allowed" : "pointer",
            }}
            type="button"
            onClick={handleFinish}
            disabled={!canFinish || busy || calibDone || !session}
          >
            {busy ? "…" : "Done"}
          </button>
          {DEV && (
            <span style={lastBatch ? s.telOn : s.telOff}>
              Telemetry: {lastBatch ? "ON" : "OFF"}
            </span>
          )}
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div style={{ padding: "12px 24px", background: "#2a1515", color: "#ef4444", fontSize: 13 }}>
          {error}
        </div>
      )}

      {/* Scrollable reading area */}
      <main style={s.content} ref={contentRef}>
        <div style={s.doc}>
          {paragraphs.length === 0 ? (
            <p style={{ color: "#888", fontSize: 14 }}>Loading text…</p>
          ) : (
            paragraphs.map((para, i) => (
              <p
                key={i}
                style={s.para}
                data-paragraph-id={`calib-${i}`}
                data-word-count={para.split(" ").length}
                data-chunk-index={i}
              >
                {para}
              </p>
            ))
          )}
        </div>
      </main>

      {/* Dev: last telemetry batch */}
      {DEV && lastBatch && (
        <details style={s.debug}>
          <summary style={{ cursor: "pointer", color: "#888" }}>Last batch</summary>
          <pre style={{ color: "#e5e5e5", fontSize: 11, marginTop: 6 }}>
            {JSON.stringify(lastBatch, null, 2)}
          </pre>
        </details>
      )}

      {/* Baseline summary overlay */}
      {calibDone && baseline && (
        <div style={s.overlay}>
          <div style={s.card}>
            <div style={{ display: "flex", alignItems: "flex-start", gap: 16, marginBottom: 24 }}>
              <span style={{ fontSize: 28, color: "#22c55e", lineHeight: 1 }}>✓</span>
              <div>
                <h2 style={{ fontSize: 20, fontWeight: 700, color: "#e5e5e5", margin: "0 0 4px" }}>
                  Calibration Complete
                </h2>
                <p style={{ fontSize: 14, color: "#888", margin: 0 }}>
                  Your reading baseline has been saved.
                </p>
              </div>
            </div>
            <div style={s.grid}>
              {[
                ["WPM estimate", Math.round(baseline.wpm_mean).toString()],
                ["Scroll velocity", `${(baseline.scroll_velocity_mean ?? 0).toFixed(1)} px/s`],
                ["Idle ratio", `${((baseline.idle_ratio_mean ?? 0) * 100).toFixed(0)}%`],
                ["Duration", fmt(baseline.calibration_duration_seconds)],
              ].map(([label, value]) => (
                <div key={label} style={s.stat}>
                  <span style={{ fontSize: 22, fontWeight: 700, color: "#6366f1" }}>{value}</span>
                  <span style={{ fontSize: 12, color: "#888" }}>{label}</span>
                </div>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button
                style={{ ...s.btn, padding: "10px 28px", fontSize: 15 }}
                onClick={() => navigate("/", { replace: true })}
              >
                Go to Dashboard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Inline styles ────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  page: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    background: "#0d0d0d",
    color: "#e5e5e5",
    fontFamily: "'Inter', system-ui, sans-serif",
    overflow: "hidden",
  },
  topbar: {
    display: "flex",
    alignItems: "center",
    background: "#161616",
    borderBottom: "1px solid #2a2a2a",
    minHeight: 52,
    flexShrink: 0,
  },
  calibBadge: {
    background: "linear-gradient(90deg,#6366f1,#7c3aed)",
    color: "#fff",
    fontSize: 13,
    fontWeight: 500,
    padding: "0 20px",
    display: "flex",
    alignItems: "center",
    alignSelf: "stretch",
    whiteSpace: "nowrap",
    flexShrink: 0,
  },
  topbarTitle: {
    flex: 1,
    fontSize: 14,
    fontWeight: 600,
    padding: "0 20px",
  },
  topbarRight: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "0 20px",
    flexShrink: 0,
  },
  timer: {
    fontSize: 14,
    color: "#888",
    fontVariantNumeric: "tabular-nums",
    fontWeight: 500,
  },
  btn: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "6px 18px",
    borderRadius: 8,
    fontSize: 13,
    fontWeight: 500,
    background: "#6366f1",
    color: "#fff",
    border: "none",
    transition: "opacity .15s",
  },
  telOn: {
    fontSize: 11,
    fontWeight: 600,
    padding: "2px 8px",
    borderRadius: 999,
    background: "rgba(34,197,94,0.15)",
    color: "#22c55e",
  },
  telOff: {
    fontSize: 11,
    fontWeight: 600,
    padding: "2px 8px",
    borderRadius: 999,
    background: "rgba(239,68,68,0.12)",
    color: "#ef4444",
  },
  content: {
    flex: 1,
    overflowY: "auto",
    padding: "48px 24px 80px",
    display: "flex",
    justifyContent: "center",
  },
  doc: {
    width: "100%",
  },
  para: {
    fontSize: 19,
    lineHeight: 1.75,
    color: "#e5e5e5",
    marginBottom: "1.6em",
    fontFamily: "'Inter', system-ui, sans-serif",
    maxWidth: "85ch",
    marginLeft: "auto",
    marginRight: "auto",
  },
  debug: {
    position: "fixed",
    bottom: 16,
    right: 16,
    background: "#161616",
    border: "1px solid #2a2a2a",
    borderRadius: 8,
    fontSize: 11,
    padding: "8px 12px",
    maxWidth: 360,
    maxHeight: 280,
    overflow: "auto",
    zIndex: 50,
  },
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.75)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 100,
  },
  card: {
    background: "#161616",
    border: "1px solid #2a2a2a",
    borderRadius: 12,
    maxWidth: 480,
    width: "90%",
    padding: 32,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 16,
    marginBottom: 28,
  },
  stat: {
    background: "#0d0d0d",
    border: "1px solid #2a2a2a",
    borderRadius: 8,
    padding: "14px 16px",
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
};
