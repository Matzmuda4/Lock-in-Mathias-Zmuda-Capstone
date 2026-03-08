import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { useTelemetry } from "../hooks/useTelemetry";
import { calibrationService, type BaselineData } from "../services/calibrationService";
import { documentService, type Chunk } from "../services/documentService";
import { sessionService, type Session, type SessionReaderData } from "../services/sessionService";
import { driftService, driftColor, type DriftState, type DriftDebug } from "../services/driftService";

const API_BASE_URL = "http://localhost:8000";

const DEV = (import.meta as unknown as { env?: { DEV?: boolean } }).env?.DEV ?? false;

// Minimum calibration reading time before "Finish" is enabled
// Minimum seconds before the finish button is enabled (prevents instant accidental clicks)
const CALIB_MIN_SECONDS = 10;

// ─── Elapsed timer ───────────────────────────────────────────────────────────

function calcInitialSeconds(session: Session): number {
  if (session.status === "ended" || session.status === "completed") {
    return session.duration_seconds ?? session.elapsed_seconds;
  }
  if (session.status === "paused") {
    return session.elapsed_seconds;
  }
  const intervalMs = Date.now() - new Date(session.started_at).getTime();
  return session.elapsed_seconds + Math.max(0, Math.floor(intervalMs / 1000));
}

function useElapsedTimer(session: Session | null) {
  const [seconds, setSeconds] = useState(() =>
    session ? calcInitialSeconds(session) : 0,
  );

  useEffect(() => {
    if (session) setSeconds(calcInitialSeconds(session));
  }, [session?.status, session?.elapsed_seconds, session?.started_at]);

  const running = session?.status === "active";
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [running]);

  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

// ─── Auth-aware asset data URL loader ────────────────────────────────────────

const _dataUrlCache = new Map<string, string>();

function useAssetDataUrl(docId: number, assetId: number, token: string | null) {
  const [url, setUrl] = useState<string | null>(() => {
    const key = `${docId}:${assetId}`;
    return assetId && token ? (_dataUrlCache.get(key) ?? null) : null;
  });
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!token || !docId || !assetId) return;
    const key = `${docId}:${assetId}`;
    const cached = _dataUrlCache.get(key);
    if (cached) { setUrl(cached); return; }

    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch(`${API_BASE_URL}/documents/${docId}/assets/${assetId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!resp.ok) {
          if (!cancelled) setFetchError(`HTTP ${resp.status}`);
          return;
        }
        const buf = await resp.arrayBuffer();
        const bytes = new Uint8Array(buf);
        let binary = "";
        for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
        const contentType = resp.headers.get("content-type") ?? "image/png";
        const dataUrl = `data:${contentType};base64,${btoa(binary)}`;
        _dataUrlCache.set(key, dataUrl);
        if (!cancelled) setUrl(dataUrl);
      } catch (e) {
        if (!cancelled) setFetchError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; };
  }, [docId, assetId, token]);

  return { url, fetchError };
}

// ─── Chunk components ────────────────────────────────────────────────────────

const HEADING_LABELS = new Set(["title", "section_header"]);

function Caption({ text }: { text: string }) {
  return <p className="chunk__caption">{text}</p>;
}

/**
 * Text/heading chunk — rendered with data-paragraph-id and data-word-count so
 * the IntersectionObserver in useTelemetry can track the visible paragraph.
 */
function TextChunk({ chunk, chunkIndex }: { chunk: Chunk; chunkIndex: number }) {
  const label = (chunk.meta?.label as string | undefined) ?? "";
  const isHeading = HEADING_LABELS.has(label);
  const wordCount = chunk.text.split(/\s+/).filter(Boolean).length;

  return (
    <div
      className={`chunk chunk--text${isHeading ? " chunk--heading" : ""}`}
      data-paragraph-id={`chunk-${chunk.id}`}
      data-word-count={wordCount}
      data-chunk-index={chunkIndex}
    >
      {isHeading
        ? <h2 className="chunk__h">{chunk.text}</h2>
        : <p className="chunk__p">{chunk.text}</p>}
    </div>
  );
}

function ImageChunk({ chunk, docId, token }: { chunk: Chunk; docId: number; token: string | null }) {
  const assetId = chunk.meta?.asset_id as number | undefined;
  const caption = chunk.meta?.caption as string | undefined;

  const hasFigCaption = !!caption && /fig\./i.test(caption);

  const { url, fetchError } = useAssetDataUrl(docId, assetId ?? 0, token);

  if (!assetId || !hasFigCaption) return null;

  if (fetchError) {
    return (
      <div className="chunk chunk--figure">
        <p className="chunk__image-error">Image unavailable ({fetchError})</p>
        {caption && <Caption text={caption} />}
      </div>
    );
  }

  return (
    <div className="chunk chunk--figure">
      {url
        ? <img src={url} alt={caption} className="chunk__img" />
        : <p className="chunk__image-loading">Loading figure…</p>
      }
      {caption && <Caption text={caption} />}
    </div>
  );
}

function TableChunk({ chunk, docId, token }: { chunk: Chunk; docId: number; token: string | null }) {
  const assetId = chunk.meta?.asset_id as number | undefined;
  const caption = chunk.meta?.caption as string | undefined;

  const { url, fetchError } = useAssetDataUrl(docId, assetId ?? 0, token);
  const [renderError, setRenderError] = useState(false);

  const imageOk = assetId && url && !fetchError && !renderError;

  return (
    <div className="chunk chunk--table">
      {imageOk && (
        <img
          src={url}
          alt={caption ?? "Table"}
          className="chunk__img"
          loading="lazy"
          onError={() => setRenderError(true)}
        />
      )}
      {!imageOk && chunk.text && (
        <pre className="chunk__table-md">{chunk.text}</pre>
      )}
      {caption && <Caption text={caption} />}
    </div>
  );
}

function ChunkCard({
  chunk,
  chunkIndex,
  docId,
  token,
}: {
  chunk: Chunk;
  chunkIndex: number;
  docId: number;
  token: string | null;
}) {
  const ct = (chunk.meta?.chunk_type as string | undefined) ?? "text";
  if (ct === "image") return <ImageChunk chunk={chunk} docId={docId} token={token} />;
  if (ct === "table") return <TableChunk chunk={chunk} docId={docId} token={token} />;
  return <TextChunk chunk={chunk} chunkIndex={chunkIndex} />;
}

// ─── Session controls ────────────────────────────────────────────────────────

function SessionControls({
  session,
  onAction,
  busy,
}: {
  session: Session;
  onAction: (a: "pause" | "resume" | "complete" | "end") => void;
  busy: boolean;
}) {
  const { status } = session;
  return (
    <div className="session-controls">
      {status === "active" && (
        <>
          <button className="btn btn--sm btn--ghost" onClick={() => onAction("pause")} disabled={busy} type="button">Pause</button>
          <button className="btn btn--sm btn--accent" onClick={() => onAction("complete")} disabled={busy} type="button">Complete</button>
        </>
      )}
      {status === "paused" && (
        <>
          <button className="btn btn--sm btn--accent" onClick={() => onAction("resume")} disabled={busy} type="button">Resume</button>
          <button className="btn btn--sm btn--ghost" onClick={() => onAction("complete")} disabled={busy} type="button">Complete</button>
        </>
      )}
      {(status === "ended" || status === "completed") && (
        <span className="session-controls__done">Session {status}</span>
      )}
    </div>
  );
}

// ─── Calibration finish button ───────────────────────────────────────────────

function CalibrationControls({
  elapsedSeconds,
  onFinish,
  busy,
}: {
  elapsedSeconds: number;
  onFinish: () => void;
  busy: boolean;
}) {
  // Only block for a few seconds to prevent accidental instant clicks
  const canFinish = elapsedSeconds >= CALIB_MIN_SECONDS;

  return (
    <div className="calib-controls">
      <button
        className="btn btn--sm btn--accent"
        type="button"
        onClick={onFinish}
        disabled={!canFinish || busy}
        title="Finish calibration when you're done reading"
      >
        {busy ? <span className="spinner" /> : "Done"}
      </button>
    </div>
  );
}

// ─── Calibration baseline summary ────────────────────────────────────────────

function BaselineSummary({
  baseline,
  sessionId,
  token,
  onDone,
}: {
  baseline: BaselineData;
  sessionId: number;
  token: string | null;
  onDone: () => void;
}) {
  const [exporting, setExporting] = useState(false);

  const exportCsv = async () => {
    if (!token) return;
    setExporting(true);
    try {
      const resp = await fetch(
        `${API_BASE_URL}/sessions/${sessionId}/export.csv`,
        { headers: { Authorization: `Bearer ${token}` } },
      );
      if (!resp.ok) return;
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `session_${sessionId}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="baseline-summary">
      <div className="baseline-summary__header">
        <span className="baseline-summary__icon">✓</span>
        <div>
          <h2 className="baseline-summary__title">Calibration complete!</h2>
          <p className="baseline-summary__sub">Your reading baseline has been saved.</p>
        </div>
      </div>

      <div className="baseline-summary__grid">
        <div className="baseline-stat">
          <span className="baseline-stat__value">{baseline.wpm_mean.toFixed(0)}</span>
          <span className="baseline-stat__label">WPM estimate</span>
        </div>
        <div className="baseline-stat">
          <span className="baseline-stat__value">{baseline.scroll_velocity_mean.toFixed(0)}</span>
          <span className="baseline-stat__label">Scroll velocity (px/s)</span>
        </div>
        <div className="baseline-stat">
          <span className="baseline-stat__value">{(baseline.idle_ratio_mean * 100).toFixed(0)}%</span>
          <span className="baseline-stat__label">Idle ratio</span>
        </div>
        <div className="baseline-stat">
          <span className="baseline-stat__value">{Math.floor(baseline.calibration_duration_seconds / 60)}m {baseline.calibration_duration_seconds % 60}s</span>
          <span className="baseline-stat__label">Session duration</span>
        </div>
      </div>

      <div className="baseline-summary__actions">
        <button
          className="btn btn--ghost btn--sm"
          type="button"
          onClick={exportCsv}
          disabled={exporting}
        >
          {exporting ? "Exporting…" : "Export CSV"}
        </button>
        <button className="btn btn--accent" type="button" onClick={onDone}>
          Go to Dashboard
        </button>
      </div>
    </div>
  );
}

// ─── Export CSV button ───────────────────────────────────────────────────────

function ExportCsvButton({ sessionId, token }: { sessionId: number; token: string | null }) {
  const [busy, setBusy] = useState(false);

  const handleExport = async () => {
    if (!token) return;
    setBusy(true);
    try {
      const resp = await fetch(
        `${API_BASE_URL}/sessions/${sessionId}/export.csv`,
        { headers: { Authorization: `Bearer ${token}` } },
      );
      if (!resp.ok) return;
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `session_${sessionId}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      className="btn btn--ghost btn--sm export-btn"
      type="button"
      onClick={handleExport}
      disabled={busy}
    >
      {busy ? "Exporting…" : "⬇ Export CSV"}
    </button>
  );
}

// ─── Dev-only debug panel ────────────────────────────────────────────────────

function DebugPanel({ batch }: { batch: object | null }) {
  const [open, setOpen] = useState(false);
  if (!DEV) return null;
  return (
    <div className="debug-panel">
      <button
        className="debug-panel__toggle"
        type="button"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? "▾" : "▸"} Last telemetry batch
      </button>
      {open && (
        <pre className="debug-panel__json">
          {batch ? JSON.stringify(batch, null, 2) : "—"}
        </pre>
      )}
    </div>
  );
}

// ─── Drift meter ─────────────────────────────────────────────────────────────

function DriftMeter({
  drift,
  showConfidence,
}: {
  drift: DriftState | null;
  showConfidence: boolean;
}) {
  if (!drift) return null;

  const pct = Math.round(drift.drift_ema * 100);
  const color = driftColor(drift.drift_ema);
  const disruptPct = Math.round((drift.disruption_score ?? 0) * 100);
  const engagePct = Math.round((drift.engagement_score ?? 0) * 100);

  return (
    <div
      className="drift-meter"
      title={`disruption=${disruptPct}% engagement=${engagePct}% level=${Math.round((drift.drift_level ?? drift.drift_score) * 100)}%`}
    >
      <span className="drift-meter__label">Drift</span>
      <span className={`drift-meter__pill drift-meter__pill--${color}`}>
        {pct}%
      </span>
      {showConfidence && (
        <>
          <span className="drift-meter__conf" title="Window confidence">
            conf:{Math.round(drift.confidence * 100)}%
          </span>
          {drift.baseline_used && (
            <span className="drift-meter__baseline" title="Calibration baseline active">
              ✓ baseline
            </span>
          )}
        </>
      )}
    </div>
  );
}

// ─── Drift debug panel (DEV only) ────────────────────────────────────────────

function DriftDebugPanel({
  token,
  sessionId,
}: {
  token: string | null;
  sessionId: number;
}) {
  const [open, setOpen] = useState(false);
  const [debug, setDebug] = useState<DriftDebug | null>(null);
  const [loading, setLoading] = useState(false);

  if (!DEV) return null;

  const fetchDebug = async () => {
    if (!token) return;
    setLoading(true);
    const data = await driftService.getDebug(token, sessionId);
    setDebug(data);
    setLoading(false);
  };

  // Top disruption contributors (skip meta/summary keys)
  const META_KEYS = new Set([
    "disruption_raw", "disruption_score", "up_rate", "down_rate",
    "delta", "engagement_score", "confidence", "prev_drift_level",
  ]);
  const topContribs = debug
    ? Object.entries(debug.beta_components)
        .filter(([k]) => !META_KEYS.has(k))
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .slice(0, 4)
    : [];

  return (
    <div className="debug-panel">
      <button
        className="debug-panel__toggle"
        type="button"
        onClick={() => { setOpen((o) => !o); if (!open) fetchDebug(); }}
      >
        {open ? "▾" : "▸"} Drift debug
      </button>
      {open && (
        <div className="debug-panel__json">
          {loading && <p>Loading…</p>}
          {debug && (
            <>
              <p><strong>baseline_used:</strong> {String(debug.baseline_used)}</p>
              <p>
                <strong>drift_level:</strong> {((debug.drift_level ?? 0) * 100).toFixed(1)}%
                &nbsp;<strong>drift_ema:</strong> {((debug.drift_ema ?? 0) * 100).toFixed(1)}%
              </p>
              <p>
                <strong>disruption:</strong> {((debug.disruption_score ?? 0) * 100).toFixed(1)}%
                &nbsp;<strong>engagement:</strong> {((debug.engagement_score ?? 0) * 100).toFixed(1)}%
              </p>
              <p>
                <strong>confidence:</strong> {(debug.confidence * 100).toFixed(0)}%
                &nbsp;({debug.n_batches_in_window} batches)
              </p>
              <p>
                <strong>pace:</strong> avail={String(debug.pace_available)}
                {debug.pace_ratio != null && <>&nbsp;ratio={debug.pace_ratio.toFixed(2)}</>}
                &nbsp;dev={debug.pace_dev.toFixed(3)}
                &nbsp;wpm={debug.window_wpm_effective.toFixed(0)}
              </p>
              <p><strong>elapsed:</strong> {debug.elapsed_minutes.toFixed(2)} min</p>
              <p><strong>top disruption drivers:</strong></p>
              <ul style={{ margin: 0, paddingLeft: 16 }}>
                {topContribs.map(([k, v]) => (
                  <li key={k}>{k}: {(v as number).toFixed(4)}</li>
                ))}
              </ul>
              <p><strong>z_scores:</strong></p>
              <pre style={{ fontSize: 10, margin: 0 }}>
                {JSON.stringify(debug.z_scores, null, 2)}
              </pre>
              <button
                type="button"
                style={{ marginTop: 4, fontSize: 10, cursor: "pointer" }}
                onClick={fetchDebug}
              >
                ↺ Refresh
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ─── ReaderPage ──────────────────────────────────────────────────────────────

export function ReaderPage() {
  const { id } = useParams<{ id: string }>();
  const sessionId = Number(id);
  const { token } = useAuth();
  const navigate = useNavigate();

  const [data, setData] = useState<SessionReaderData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [allLoaded, setAllLoaded] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);

  // Calibration-specific state
  const [calibBaseline, setCalibBaseline] = useState<BaselineData | null>(null);
  const [calibDone, setCalibDone] = useState(false);

  // Drift state — polled every 3 s while active
  const [driftState, setDriftState] = useState<DriftState | null>(null);

  const timerDisplay = useElapsedTimer(data?.session ?? null);
  // Raw seconds for calibration finish-condition logic
  const [timerSeconds, setTimerSeconds] = useState(0);
  useEffect(() => {
    const session = data?.session;
    if (!session) return;
    const calc = () => {
      if (session.status === "ended" || session.status === "completed")
        return session.duration_seconds ?? session.elapsed_seconds;
      if (session.status === "paused") return session.elapsed_seconds;
      const ms = Date.now() - new Date(session.started_at).getTime();
      return session.elapsed_seconds + Math.max(0, Math.floor(ms / 1000));
    };
    setTimerSeconds(calc());
    if (session.status !== "active") return;
    const id = setInterval(() => setTimerSeconds(calc()), 1000);
    return () => clearInterval(id);
  }, [data?.session?.status, data?.session?.elapsed_seconds, data?.session?.started_at]);

  // Ref for the scrollable content area — passed to useTelemetry
  const contentRef = useRef<HTMLDivElement>(null);

  // Telemetry — active only while the session is in "active" status
  const isActive = data?.session?.status === "active";
  const { lastBatch, collecting } = useTelemetry({
    sessionId,
    token,
    active: isActive,
    containerRef: contentRef,
  });

  useEffect(() => {
    if (!token) return;
    sessionService
      .getReader(token, sessionId)
      .then((d) => {
        setData(d);
        setAllLoaded(d.chunks.length >= d.total_chunks);
      })
      .catch((e) => setError(e.message));
  }, [token, sessionId]);

  // Poll drift every 3 seconds while session is active
  useEffect(() => {
    if (!token || !isActive) return;
    const poll = async () => {
      const state = await driftService.getDrift(token, sessionId);
      if (state) setDriftState(state);
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, [token, sessionId, isActive]);

  const loadMore = useCallback(async () => {
    if (!token || !data) return;
    setLoadingMore(true);
    try {
      const more = await documentService.getParsed(token, data.document_id, data.chunks.length, 30);
      const merged = [...data.chunks, ...more.chunks];
      setData((prev) => prev && { ...prev, chunks: merged });
      setAllLoaded(merged.length >= more.total_chunks);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load more");
    } finally {
      setLoadingMore(false);
    }
  }, [token, data]);

  const handleCalibrationFinish = useCallback(async () => {
    if (!token || !data) return;
    setActionBusy(true);
    try {
      const result = await calibrationService.complete(token, sessionId);
      setCalibBaseline(result.baseline);
      setCalibDone(true);
      // Update local session state to "completed" so telemetry stops
      const updatedSession: Session = {
        ...data.session,
        status: "completed",
        duration_seconds: result.baseline.calibration_duration_seconds,
      };
      setData((prev) => prev && { ...prev, session: updatedSession });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Calibration completion failed");
    } finally {
      setActionBusy(false);
    }
  }, [token, sessionId, data]);

  const handleSessionAction = useCallback(
    async (action: "pause" | "resume" | "complete" | "end") => {
      if (!token || !data) return;
      setActionBusy(true);
      try {
        let updated: Session;
        if (action === "pause") updated = await sessionService.pause(token, sessionId);
        else if (action === "resume") updated = await sessionService.resume(token, sessionId);
        else if (action === "complete") updated = await sessionService.complete(token, sessionId);
        else updated = await sessionService.end(token, sessionId);
        setData((prev) => prev && { ...prev, session: updated });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Action failed");
      } finally {
        setActionBusy(false);
      }
    },
    [token, sessionId, data],
  );

  if (error) {
    return (
      <div className="splash">
        <div style={{ textAlign: "center" }}>
          <p className="error-banner">{error}</p>
          <button className="btn btn--ghost" onClick={() => navigate("/")} type="button">← Back to dashboard</button>
        </div>
      </div>
    );
  }

  if (!data) {
    return <div className="splash"><span className="splash__spinner" /></div>;
  }

  const { session, parse_status, chunks, document_id } = data;
  // "unknown" can occur for calibration docs (no parse job) — treat as succeeded
  // if chunks were actually returned.
  const effectiveParseStatus =
    parse_status === "unknown" && chunks.length > 0 ? "succeeded" : parse_status;
  const isParsing = effectiveParseStatus === "pending" || effectiveParseStatus === "running";
  const parseFailed = effectiveParseStatus === "failed";
  const isCalibration = session.mode === "calibration";

  // Show baseline summary overlay once calibration is complete
  if (isCalibration && calibDone && calibBaseline) {
    return (
      <div className="reader">
        <main className="reader-content" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <BaselineSummary
            baseline={calibBaseline}
            sessionId={sessionId}
            token={token}
            onDone={() => navigate("/")}
          />
        </main>
      </div>
    );
  }

  return (
    <div className="reader">
      {/* Calibration banner */}
      {isCalibration && (
        <div className="calib-banner">
          <span className="calib-banner__icon">⊙</span>
          Read at your natural pace — we are measuring your baseline.
        </div>
      )}

      {/* Top bar */}
      <header className="reader-bar">
        <button className="btn btn--ghost btn--sm" onClick={() => navigate("/")} type="button">
          ← Dashboard
        </button>

        <div className="reader-bar__center">
          <span className="reader-bar__title">{session.name}</span>
          <span className={`badge badge--${session.status}`}>{session.status}</span>
          {!isCalibration && <span className="reader-bar__mode">{session.mode}</span>}
        </div>

        <div className="reader-bar__right">
          <span className="reader-bar__timer">{timerDisplay}</span>
          <DriftMeter drift={driftState} showConfidence={DEV} />
          {DEV && (
            <span className={`telemetry-badge${collecting ? " telemetry-badge--on" : ""}`}>
              ⊙ Telemetry: {collecting ? "ON" : "OFF"}
            </span>
          )}
          {isCalibration
            ? (
              <CalibrationControls
                elapsedSeconds={timerSeconds}
                onFinish={handleCalibrationFinish}
                busy={actionBusy}
              />
            )
            : (
              <SessionControls session={session} onAction={handleSessionAction} busy={actionBusy} />
            )
          }
        </div>
      </header>

      {/* Content */}
      <main className="reader-content" ref={contentRef}>
        {isParsing && (
          <div className="reader-notice">
            <span className="spinner" />
            <span>Document is still being parsed… check back in a moment.</span>
          </div>
        )}
        {parseFailed && (
          <div className="reader-notice reader-notice--error">
            Parse failed. Go back to the dashboard to retry.
          </div>
        )}
        {!isParsing && !parseFailed && chunks.length === 0 && (
          <div className="reader-notice">No content was extracted from this document.</div>
        )}

        {chunks.map((chunk, idx) => (
          <ChunkCard
            key={chunk.id}
            chunk={chunk}
            chunkIndex={idx}
            docId={document_id}
            token={token}
          />
        ))}

        {!allLoaded && !isParsing && (
          <button className="btn btn--ghost btn--load-more" onClick={loadMore} disabled={loadingMore} type="button">
            {loadingMore ? <span className="spinner" /> : "Load more"}
          </button>
        )}

        {/* Export CSV — always visible in calibration, dev-only otherwise */}
        {(isCalibration || DEV) && (
          <ExportCsvButton sessionId={sessionId} token={token} />
        )}
      </main>

      {/* Dev debug panels */}
      {DEV && <DebugPanel batch={lastBatch} />}
      {DEV && <DriftDebugPanel token={token} sessionId={sessionId} />}

      <style>{`
        /* ── Layout ── */
        .reader { display:flex; flex-direction:column; min-height:100vh; }

        .reader-bar {
          background: var(--bg-surface);
          border-bottom: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          padding: 10px 20px;
          position: sticky;
          top: 0;
          z-index: 10;
          flex-wrap: wrap;
        }

        .reader-bar__center {
          display:flex; align-items:center; gap:10px; flex:1; justify-content:center;
        }
        .reader-bar__title {
          font-weight:600; font-size:14px;
          max-width:320px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
        }
        .reader-bar__mode { font-size:11px; color:var(--text-muted); font-style:italic; text-transform:capitalize; }
        .reader-bar__right { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
        .reader-bar__timer { font-family:var(--font-mono); font-size:14px; color:var(--accent); min-width:48px; }

        /* ── Drift meter ── */
        .drift-meter {
          display:flex; align-items:center; gap:5px;
          font-size:12px;
        }
        .drift-meter__label {
          color:var(--text-muted);
          font-size:11px;
        }
        .drift-meter__pill {
          padding:2px 8px;
          border-radius:9999px;
          font-weight:600;
          font-size:12px;
          font-family:var(--font-mono);
        }
        .drift-meter__pill--green  { background:#22c55e22; color:#22c55e; border:1px solid #22c55e44; }
        .drift-meter__pill--yellow { background:#eab30822; color:#eab308; border:1px solid #eab30844; }
        .drift-meter__pill--red    { background:#ef444422; color:#ef4444; border:1px solid #ef444444; }
        .drift-meter__conf { font-size:10px; color:var(--text-muted); }
        .drift-meter__baseline { font-size:10px; color:#22c55e; margin-left:4px; }

        /* ── Telemetry indicator (dev-only) ── */
        .telemetry-badge {
          font-size:11px;
          color:var(--text-muted);
          padding:2px 7px;
          border-radius:9999px;
          border:1px solid var(--border);
          white-space:nowrap;
        }
        .telemetry-badge--on {
          color:#22c55e;
          border-color:#22c55e44;
          background:#22c55e11;
        }

        .session-controls { display:flex; align-items:center; gap:6px; }
        .session-controls__done { font-size:12px; color:var(--text-muted); font-style:italic; text-transform:capitalize; }
        .btn--accent { background:var(--accent); color:#fff; border-color:var(--accent); }
        .btn--accent:hover { opacity:0.88; }

        /* ── Content column ── */
        .reader-content {
          max-width: 90ch;
          width: 100%;
          margin: 0 auto;
          padding: 48px 24px 100px;
          overflow-y: auto;
          height: calc(100vh - 56px);
        }

        /* ── Chunk base ── */
        .chunk { margin-bottom: 28px; }

        /* Text — larger, more readable */
        .chunk__p {
          font-size: 18px;
          line-height: 1.75;
          color: var(--text);
          margin: 0;
          max-width: 75ch;
        }

        /* Headings */
        .chunk--heading {
          margin-top: 44px;
          margin-bottom: 12px;
          padding-bottom: 6px;
          border-bottom: 1px solid var(--border);
        }
        .chunk__h {
          font-size: 20px;
          font-weight: 700;
          color: var(--text);
          margin: 0;
          line-height: 1.35;
        }

        /* Figure / image */
        .chunk--figure {
          margin-top: 36px;
          margin-bottom: 36px;
          text-align: center;
        }
        .chunk__img {
          max-width: 100%;
          height: auto;
          display: block;
          margin: 0 auto;
          border-radius: var(--radius-sm, 4px);
        }
        .chunk__image-loading {
          font-size: 13px;
          color: var(--text-muted);
          margin: 8px 0;
        }
        .chunk__image-error {
          font-size: 13px;
          color: var(--error, #ef4444);
          margin: 8px 0;
        }

        /* Table */
        .chunk--table {
          margin-top: 36px;
          margin-bottom: 36px;
        }
        .chunk__table-md {
          font-size: 14px;
          line-height: 1.6;
          color: var(--text);
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 14px 16px;
          overflow-x: auto;
          white-space: pre;
          font-family: var(--font-mono, monospace);
        }

        /* Caption */
        .chunk__caption {
          font-size: 14px;
          color: var(--text-muted);
          font-style: italic;
          text-align: center;
          margin: 8px 0;
        }

        /* Notices */
        .reader-notice {
          display: flex; align-items: center; gap: 10px;
          color: var(--text-muted); font-size: 14px;
          padding: 20px; margin-bottom: 24px;
          border-radius: var(--radius);
          border: 1px dashed var(--border);
        }
        .reader-notice--error { color: var(--error); border-color: rgba(239,68,68,0.4); }

        /* Load more */
        .btn--load-more { width: 100%; margin-top: 8px; }

        /* ── Calibration banner ── */
        .calib-banner {
          background: linear-gradient(90deg, var(--accent) 0%, #7c3aed 100%);
          color: #fff;
          font-size: 13px;
          font-weight: 500;
          padding: 8px 20px;
          display: flex;
          align-items: center;
          gap: 8px;
          letter-spacing: 0.01em;
        }
        .calib-banner__icon { font-size: 15px; }

        /* ── Calibration controls ── */
        .calib-controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .calib-controls__hint {
          font-size: 11px;
          color: var(--text-muted);
          white-space: nowrap;
        }

        /* ── Baseline summary ── */
        .baseline-summary {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg, 12px);
          max-width: 480px;
          width: 100%;
          padding: 32px;
        }
        .baseline-summary__header {
          display: flex;
          align-items: flex-start;
          gap: 16px;
          margin-bottom: 24px;
        }
        .baseline-summary__icon {
          font-size: 28px;
          color: #22c55e;
          line-height: 1;
        }
        .baseline-summary__title {
          font-size: 20px;
          font-weight: 700;
          color: var(--text);
          margin: 0 0 4px;
        }
        .baseline-summary__sub {
          font-size: 13px;
          color: var(--text-muted);
          margin: 0;
        }
        .baseline-summary__grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-bottom: 24px;
        }
        .baseline-stat {
          background: var(--bg);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 12px 16px;
        }
        .baseline-stat__value {
          display: block;
          font-size: 22px;
          font-weight: 700;
          color: var(--accent);
          font-family: var(--font-mono, monospace);
        }
        .baseline-stat__label {
          font-size: 11px;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .baseline-summary__actions {
          display: flex;
          justify-content: flex-end;
          gap: 10px;
        }

        /* ── Export button ── */
        .export-btn {
          margin-top: 16px;
          display: block;
        }

        /* ── Dev debug panel ── */
        .debug-panel {
          position: fixed;
          bottom: 16px;
          right: 16px;
          z-index: 100;
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          font-size: 12px;
          max-width: 420px;
          box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }
        .debug-panel__toggle {
          background: none;
          border: none;
          color: var(--text-muted);
          padding: 8px 12px;
          cursor: pointer;
          width: 100%;
          text-align: left;
          font-size: 12px;
        }
        .debug-panel__toggle:hover { color: var(--text); }
        .debug-panel__json {
          margin: 0;
          padding: 0 12px 12px;
          color: var(--text);
          font-family: var(--font-mono, monospace);
          font-size: 11px;
          line-height: 1.5;
          max-height: 300px;
          overflow-y: auto;
          white-space: pre-wrap;
          word-break: break-all;
        }
      `}</style>
    </div>
  );
}
