import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { useTelemetry } from "../hooks/useTelemetry";
import { calibrationService, type BaselineData } from "../services/calibrationService";
import { documentService, type Chunk } from "../services/documentService";
import { sessionService, type Session, type SessionReaderData } from "../services/sessionService";
import { driftService, driftColor, type DriftState, type DriftDebug } from "../services/driftService";
import { classificationService, STATE_COLORS, STATE_LABELS, type AttentionalState } from "../services/classificationService";
import { interventionService, type ActiveIntervention, type InterventionTier, type InterventionType } from "../services/interventionService";
import { InterventionList } from "../components/interventions/InterventionList";
import { SectionSummaryCard } from "../components/interventions/SectionSummaryCard";
import { TextReformatBanner } from "../components/interventions/TextReformatBanner";
import JourneyWidget from "../components/interventions/JourneyWidget";
import BreakSuggestionOverlay from "../components/interventions/BreakSuggestionOverlay";
import AudioscapeWidget from "../components/interventions/AudioscapeWidget";
import ChimeWidget from "../components/interventions/ChimeWidget";
import BadgePopup from "../components/interventions/BadgePopup";
import { BADGE_DEFS, type BadgeDef, type BadgeId } from "../types/badges";

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

/**
 * Single-tick timer that returns BOTH a display string (MM:SS) and raw
 * seconds.  Replaces the previous two separate 1-second intervals that were
 * causing two ReaderPage re-renders per second.
 */
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

  const display = useMemo(() => {
    const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
    const ss = String(seconds % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  }, [seconds]);

  return { display, seconds };
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
    <div
      className="chunk chunk--figure"
      data-paragraph-id={`chunk-img-${assetId}`}
      data-chunk-index={-1}
    >
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
    <div
      className="chunk chunk--table"
      data-paragraph-id={`chunk-tbl-${chunk.id}`}
      data-chunk-index={-1}
    >
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

// memo: chunk props never change during a session (only on loadMore), so this
// component skips re-rendering on every timer tick / drift poll / intervention
// update — the most expensive re-render cycle in the document view.
const ChunkCard = memo(function ChunkCard({
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
});

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

// ─── Adaptive assistant panel shell (adaptive mode only) ─────────────────────
// memo: the panel only needs to re-render when interventions, XP, or open state
// changes — not on every 1-second timer tick or 5-second drift poll.

interface AssistantPanelProps {
  panelRef:             React.RefObject<HTMLDivElement>;
  open:                 boolean;
  onToggle:             () => void;
  activeInterventions:  ActiveIntervention[];
  onDismissIntervention:(id: number) => void;
  /** Dev/test only — fire any intervention type directly. */
  onManualFire:         (type: InterventionType, tier: InterventionTier) => void;
  /** Dev/test only — award a specific badge immediately for testing. */
  onDevBadge:           (badgeId: BadgeId) => void;
  /** Accumulated XP — fed to the journey widget when gamification fires. */
  sessionXP:            number;
  /** True once the session has ended — advances journey to final checkpoint. */
  sessionEnded:         boolean;
  /** Badges already earned this session — shown as mini-icons in the journey widget. */
  earnedBadges:         BadgeDef[];
  /** Whether a break is currently in progress — pauses the audioscape. */
  breakActive:          boolean;
  /** True once the journey has been completed (Mountain Peak reached + congrats dismissed). */
  journeyCompleted:     boolean;
  /** Callback to mark the journey as completed from within the journey widget. */
  onJourneyComplete:    () => void;
}

const AssistantPanel = memo(function AssistantPanel({
  panelRef,
  open,
  onToggle,
  activeInterventions,
  onDismissIntervention,
  onManualFire,
  onDevBadge,
  sessionXP,
  sessionEnded,
  earnedBadges,
  breakActive,
  journeyCompleted,
  onJourneyComplete,
}: AssistantPanelProps) {
  // Dev test controls
  const [devType, setDevType] = useState<InterventionType>("focus_point");
  const [devTier, setDevTier] = useState<InterventionTier>("moderate");
  const [devFiring, setDevFiring] = useState(false);

  // Dev panel visibility — persisted in localStorage so it survives refreshes.
  // Toggle off to observe natural LLM behaviour without test buttons.
  const [showDev, setShowDev] = useState<boolean>(() => {
    try { return localStorage.getItem("lockin_dev_open") !== "false"; } catch { return true; }
  });
  const handleToggleDev = useCallback(() => {
    setShowDev((prev) => {
      const next = !prev;
      try { localStorage.setItem("lockin_dev_open", String(next)); } catch {}
      return next;
    });
  }, []);

  const handleDevFire = async () => {
    setDevFiring(true);
    await onManualFire(devType, devTier);
    setDevFiring(false);
  };

  // Text-prompt interventions rendered in this panel
  const textPromptInterventions = activeInterventions.filter((i) =>
    ["focus_point", "re_engagement", "comprehension_check"].includes(i.type ?? ""),
  );
  const textReformatInterventions = activeInterventions.filter(
    (i) => i.type === "text_reformat",
  );
  // Journey widget — only when gamification intervention is active
  const gamificationIntervention = activeInterventions.find((i) => i.type === "gamification") ?? null;
  // Audioscape widget — only when ambient_sound intervention is active
  const ambientSoundIntervention = activeInterventions.find((i) => i.type === "ambient_sound") ?? null;
  // break_suggestion is rendered as a full-screen overlay, not here
  const hasVisibleInterventions =
    textPromptInterventions.length > 0 ||
    textReformatInterventions.length > 0 ||
    gamificationIntervention !== null ||
    ambientSoundIntervention !== null;

  return (
    <aside
      id="assistant-panel-root"
      data-testid="assistant-panel-root"
      ref={panelRef}
      className={`assistant-panel${open ? "" : " assistant-panel--collapsed"}`}
      aria-label="Adaptive assistant panel"
    >
      <div className="assistant-panel__header">
        <span className="assistant-panel__title">
          {open ? "Lock-in Assistant" : ""}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          {/* Dev mode toggle — only visible in Vite dev builds */}
          {DEV && open && (
            <button
              type="button"
              onClick={handleToggleDev}
              title={showDev ? "Hide dev tools" : "Show dev tools"}
              style={{
                background:   showDev ? "rgba(245,158,11,0.15)" : "rgba(0,0,0,0.06)",
                border:       `1px solid ${showDev ? "rgba(245,158,11,0.5)" : "rgba(0,0,0,0.1)"}`,
                borderRadius: "5px",
                color:        showDev ? "#b45309" : "#94a3b8",
                fontSize:     "10px",
                fontWeight:   700,
                padding:      "2px 6px",
                cursor:       "pointer",
                lineHeight:   1.4,
                letterSpacing:"0.02em",
                transition:   "background 0.15s, color 0.15s",
              }}
            >
              {showDev ? "🧪 DEV" : "DEV"}
            </button>
          )}
          <button
            className="assistant-panel__toggle"
            type="button"
            aria-label={open ? "Collapse panel" : "Expand panel"}
            onClick={onToggle}
          >
            {open ? "›" : "‹"}
          </button>
        </div>
      </div>

      {open && (
        <div className="assistant-panel__body">

          {/* ── Journey widget — only when a gamification intervention is active
                 and the journey hasn't been completed yet ── */}
          {gamificationIntervention && !journeyCompleted && (
            <JourneyWidget
              intervention={gamificationIntervention}
              xp={sessionXP}
              sessionEnded={sessionEnded}
              earnedBadges={earnedBadges}
              onComplete={onJourneyComplete}
            />
          )}

          {/* ── Audioscape — only when ambient_sound intervention is active ── */}
          {ambientSoundIntervention && (
            <AudioscapeWidget
              intervention={ambientSoundIntervention}
              onDismiss={onDismissIntervention}
              forcePause={breakActive}
            />
          )}

          {/* ── Text reformat banner ─────────────────────────────────────── */}
          {textReformatInterventions.length > 0 && (
            <div className="panel-section">
              <p className="panel-section__label">Text Adaptation</p>
              {textReformatInterventions.map((i) => (
                <TextReformatBanner
                  key={i.intervention_id}
                  intervention={i}
                  onStop={onDismissIntervention}
                />
              ))}
            </div>
          )}

          {/* ── Active text-prompt interventions ─────────────────────────── */}
          {textPromptInterventions.length > 0 && (
            <div className="panel-section">
              <p className="panel-section__label">Active</p>
              <InterventionList
                interventions={textPromptInterventions}
                onDismiss={onDismissIntervention}
              />
            </div>
          )}

          {/* ── Dev: manual fire (hidden when showDev is off) ────────────── */}
          {DEV && showDev && (
            <div className="panel-section">
              <p className="panel-section__label" style={{ color: "#f59e0b" }}>
                🧪 Dev — Test Intervention
              </p>
              <div style={{ display: "flex", gap: "6px", marginBottom: "8px" }}>
                <select
                  value={devType}
                  onChange={(e) => setDevType(e.target.value as InterventionType)}
                  style={{
                    flex: 1, fontSize: "12px", padding: "5px 8px",
                    border: "1px solid var(--border)", borderRadius: "6px",
                    background: "var(--bg-surface)", color: "var(--text)",
                  }}
                >
                  <option value="focus_point">Focus Point</option>
                  <option value="re_engagement">Re-engagement</option>
                  <option value="comprehension_check">Comprehension</option>
                  <option value="section_summary">Section Summary</option>
                  <option value="text_reformat">Text Reformat</option>
                  <option value="break_suggestion">Break Suggestion</option>
                  <option value="chime">Chime</option>
                  <option value="gamification">Gamification (badge)</option>
                  <option value="ambient_sound">Audioscape (ambient)</option>
                </select>
                <select
                  value={devTier}
                  onChange={(e) => setDevTier(e.target.value as InterventionTier)}
                  style={{
                    width: "96px", fontSize: "12px", padding: "5px 8px",
                    border: "1px solid var(--border)", borderRadius: "6px",
                    background: "var(--bg-surface)", color: "var(--text)",
                  }}
                >
                  <option value="subtle">Subtle</option>
                  <option value="moderate">Moderate</option>
                  <option value="strong">Strong</option>
                  <option value="special">Special</option>
                </select>
              </div>
              <button
                type="button"
                onClick={handleDevFire}
                disabled={devFiring}
                style={{
                  width: "100%", padding: "8px 12px",
                  background: devFiring ? "var(--bg)" : "rgba(245,158,11,0.12)",
                  border: "1.5px solid rgba(245,158,11,0.4)",
                  borderRadius: "7px",
                  color: "#b45309", fontSize: "12px", fontWeight: 700,
                  cursor: devFiring ? "wait" : "pointer",
                }}
              >
                {devFiring ? "Firing…" : "▶ Fire Intervention"}
              </button>

              {/* Dev badge awards — only relevant when gamification is active */}
              <p style={{ margin: "10px 0 4px", fontSize: "10px", fontWeight: 700, color: "#7c3aed" }}>
                🏅 Award Badge (dev)
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                {BADGE_DEFS.map((b) => (
                  <button
                    key={b.id}
                    type="button"
                    onClick={() => onDevBadge(b.id)}
                    style={{
                      width: "100%", padding: "5px 8px",
                      background: "rgba(124,58,237,0.08)",
                      border: "1px solid rgba(124,58,237,0.3)",
                      borderRadius: "6px",
                      color: "#6d28d9", fontSize: "11px", fontWeight: 600,
                      cursor: "pointer", textAlign: "left",
                      display: "flex", alignItems: "center", gap: "6px",
                    }}
                  >
                    <img src={b.icon} alt="" style={{ width: "14px", height: "14px", objectFit: "contain" }} />
                    {b.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* ── Placeholder when nothing is active and dev tools are hidden ── */}
          {!hasVisibleInterventions && (!DEV || !showDev) && (
            <div className="panel-section panel-section--muted">
              <p className="assistant-panel__placeholder">
                Interventions will appear here.
              </p>
            </div>
          )}

        </div>
      )}
    </aside>
  );
});

// ─── Dev-only Export Bundle button ───────────────────────────────────────────

interface ExportBundleResult {
  folder: string;
  files: string[];
  state_packet_count: number;
  master_append?: {
    master_jsonl_path: string;
    appended_packet_count: number;
    skipped_packet_count: number;
    baseline_path: string;
    baseline_ref: string;
    baseline_embedded_in_packet: boolean;
  } | null;
}

function ExportBundleButton({ sessionId, token }: { sessionId: number; token: string | null }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ExportBundleResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async () => {
    if (!token) return;
    setBusy(true);
    setResult(null);
    setError(null);
    try {
      const resp = await fetch(
        `${API_BASE_URL}/sessions/${sessionId}/export/bundle?append_to_master=1`,
        { headers: { Authorization: `Bearer ${token}` } },
      );
      if (!resp.ok) {
        const msg = await resp.text();
        console.error("Export bundle failed:", resp.status, msg);
        setError(`Export failed (${resp.status})`);
        return;
      }
      const data: ExportBundleResult = await resp.json();
      setResult(data);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ display: "inline-flex", flexDirection: "column", gap: 4 }}>
      <button
        className="btn btn--ghost btn--sm export-btn"
        type="button"
        onClick={handleExport}
        disabled={busy}
      >
        {busy ? "Exporting…" : "📦 Export Bundle"}
      </button>
      {error && (
        <div style={{ fontSize: 10, color: "var(--error, #e55)", maxWidth: 260 }}>
          {error}
        </div>
      )}
      {result && (
        <div style={{ fontSize: 10, color: "var(--text-muted)", maxWidth: 280, wordBreak: "break-all" }}>
          <div><strong>Bundle:</strong> {result.folder}</div>
          <div style={{ color: "var(--text-muted)", marginTop: 2 }}>
            {result.files.join(", ")}
          </div>
          {result.master_append && (
            <div style={{ marginTop: 4, borderTop: "1px solid var(--border)", paddingTop: 4 }}>
              <div>
                <strong>Master JSONL:</strong>{" "}
                {result.master_append.appended_packet_count} packets appended
                {result.master_append.skipped_packet_count > 0 &&
                  ` (${result.master_append.skipped_packet_count} skipped — already present)`}
              </div>
              <div style={{ marginTop: 2 }}>{result.master_append.master_jsonl_path}</div>
              <div style={{ marginTop: 2 }}>
                <strong>Baseline:</strong> {result.master_append.baseline_path}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Dev-only debug panel ────────────────────────────────────────────────────

import type { TelemetrySanityWarnings } from "../hooks/useTelemetry";

function DebugPanel({
  batch,
  warnings,
}: {
  batch: object | null;
  warnings: TelemetrySanityWarnings | null;
}) {
  const [open, setOpen] = useState(false);
  if (!DEV) return null;

  const hasWarnings = warnings && (
    warnings.idleExceedsWindow ||
    warnings.scrollZeroWithProgress ||
    warnings.paragraphMissing
  );

  return (
    <div className="debug-panel">
      <button
        className="debug-panel__toggle"
        type="button"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? "▾" : "▸"} Last telemetry batch
        {hasWarnings && <span style={{ color: "orange", marginLeft: 6 }}>⚠ warnings</span>}
      </button>
      {open && (
        <>
          {hasWarnings && (
            <div style={{ padding: "4px 12px", fontSize: 11, color: "orange" }}>
              {warnings?.idleExceedsWindow && <div>⚠ idle_seconds &gt; 2.0 (should be 0–2)</div>}
              {warnings?.scrollZeroWithProgress && <div>⚠ scroll=0 but progress ratio changed — scroll capture may be broken</div>}
              {warnings?.paragraphMissing && <div>⚠ current_paragraph_id missing — IntersectionObserver may not be attached</div>}
            </div>
          )}
          <pre className="debug-panel__json">
            {batch ? JSON.stringify(batch, null, 2) : "—"}
          </pre>
        </>
      )}
    </div>
  );
}

// ─── Drift meter ─────────────────────────────────────────────────────────────

// memo: only re-renders when drift state actually changes (every 5 s)
const DriftMeter = memo(function DriftMeter({
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
});

// ─── Attentional state meter ──────────────────────────────────────────────────

/**
 * Displays the RF classifier's 4-state probability distribution beside the
 * drift meter in the top bar.
 *
 * Shows a compact bar for each state scaled to the probability [0,1].
 * The primary state label is highlighted.  A spinner is shown while the
 * first full-window packet has not yet been classified (< ~30 s).
 */
// memo: only re-renders when the RF classification changes (every 10 s)
const AttentionalStateMeter = memo(function AttentionalStateMeter({
  state,
}: {
  state: AttentionalState | null;
}) {
  if (!state) {
    return (
      <div className="attn-meter attn-meter--pending" title="Waiting for first full-window classification (~30s)">
        <span className="attn-meter__label">State</span>
        <span className="attn-meter__waiting">…</span>
      </div>
    );
  }

  const { distribution, primary_state, confidence } = state;
  const entries = Object.entries(distribution) as [string, number][];

  return (
    <div
      className="attn-meter"
      title={`Primary: ${STATE_LABELS[primary_state]} (${(confidence * 100).toFixed(0)}%)\n${state.rationale}`}
    >
      <span className="attn-meter__label">State</span>
      <div className="attn-meter__bars">
        {entries.map(([stateKey, prob]) => (
          <div
            key={stateKey}
            className={`attn-meter__bar${stateKey === primary_state ? " attn-meter__bar--primary" : ""}`}
            title={`${STATE_LABELS[stateKey]}: ${(prob * 100).toFixed(1)}%`}
          >
            <div
              className="attn-meter__bar-fill"
              style={{
                height: `${Math.max(3, prob * 28)}px`,
                background: STATE_COLORS[stateKey],
                opacity: stateKey === primary_state ? 1 : 0.45,
              }}
            />
            <span
              className="attn-meter__bar-label"
              style={{ color: stateKey === primary_state ? STATE_COLORS[stateKey] : undefined }}
            >
              {STATE_LABELS[stateKey].slice(0, 3)}
            </span>
          </div>
        ))}
      </div>
      <span
        className="attn-meter__primary"
        style={{ color: STATE_COLORS[primary_state] }}
      >
        {STATE_LABELS[primary_state]}
      </span>
    </div>
  );
});

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

  // Attentional state — polled every 10 s (matches classifier cadence)
  const [attentionalState, setAttentionalState] = useState<AttentionalState | null>(null);

  // Active interventions — polled every 10 s while session is active
  const [activeInterventions, setActiveInterventions] = useState<ActiveIntervention[]>([]);

  // XP accumulated this session (focused = +10, hyperfocused = +20 per window)
  const [sessionXP, setSessionXP] = useState(0);
  const lastXpSeqRef = useRef<number | null>(null);

  // Becomes true after the first RF classification — used to gate the LLM trigger poll
  const [hasFirstClassification, setHasFirstClassification] = useState(false);

  // ── Badge system ──────────────────────────────────────────────────────────
  // earnedBadges: ordered list of badges already collected this session
  const [earnedBadges, setEarnedBadges]   = useState<BadgeDef[]>([]);
  // pendingBadge: badge whose popup is currently showing (null = no popup)
  const [pendingBadge, setPendingBadge]   = useState<BadgeDef | null>(null);
  const earnedIdsRef         = useRef<Set<BadgeId>>(new Set());
  const consecutiveFocusRef  = useRef(0);  // consecutive focused OR hyperfocused windows
  const consecutiveHyperRef  = useRef(0);  // consecutive hyperfocused-only windows
  const consecutiveCleanRef  = useRef(0);  // consecutive non-drifting windows
  const prevAttStateRef      = useRef<string | null>(null);
  // Track how many consecutive drifting windows before comeback_kid
  const consecutiveDriftRef  = useRef(0);

  // ── Journey completion — hide widget after "Amazing!" but allow restart ─
  const [journeyCompleted, setJourneyCompleted] = useState(false);
  // Track the last gamification intervention id so we know when a NEW one fires
  const lastGamificationIdRef = useRef<number | null>(null);

  // ── Break-active flag (pauses audioscape during a break) ─────────────────
  const [breakActive, setBreakActive] = useState(false);

  // Section-summary insertion points: intervention_id → chunk index where the
  // card should appear (after that chunk index in the document).
  // Populated when a new section_summary enters activeInterventions —
  // either from the backend _chunk_index field or the current scroll position.
  const [summaryInsertions, setSummaryInsertions] = useState<Map<number, number>>(new Map());

  // Tracks the chunk index that's at the top of the viewport (scroll tracking).
  const viewportChunkIdxRef = useRef(0);

  // Single timer — provides both the formatted display string and raw seconds.
  // Replaces two separate 1-second intervals that used to cause 2 re-renders/sec.
  const { display: timerDisplay, seconds: timerSeconds } = useElapsedTimer(data?.session ?? null);

  // Ref for the scrollable content area — passed to useTelemetry
  const contentRef = useRef<HTMLDivElement>(null);
  // Ref for the adaptive assistant panel — panel zone detection
  const panelRef = useRef<HTMLDivElement>(null);

  // Adaptive panel state (only meaningful in adaptive mode)
  const isAdaptive = data?.session?.mode === "adaptive";
  const [panelOpen, setPanelOpen] = useState(true);
  const handlePanelToggle = useCallback(() => setPanelOpen((o) => !o), []);

  // Telemetry — active only while the session is in "active" status
  const isActive = data?.session?.status === "active";
  const isPaused = data?.session?.status === "paused";

  const { lastBatch, collecting, warnings } = useTelemetry({
    sessionId,
    token,
    active: isActive,
    containerRef: contentRef,
    sessionPaused: isPaused,
    panelContainerRef: isAdaptive ? panelRef : undefined,
    panelOpen: isAdaptive ? panelOpen : false,
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
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [token, sessionId, isActive]);

  // Poll attentional-state classification every 10 s (RF classifier cadence).
  // Returns null for the first ~30 s (window not full yet) — shown as "…" in the meter.
  useEffect(() => {
    if (!token || !isActive) return;
    const poll = async () => {
      try {
        const result = await classificationService.getAttentionalState(token, sessionId);
        if (result) setAttentionalState(result);
      } catch {
        // Unexpected errors are silently swallowed — classification is non-critical
      }
    };
    // Delay initial poll by 30 s — no point polling before the first full window
    const initialDelay = setTimeout(poll, 30_000);
    const id = setInterval(poll, 10_000);
    return () => { clearTimeout(initialDelay); clearInterval(id); };
  }, [token, sessionId, isActive]);

  // Poll active interventions every 10 s while session is active.
  // The backend fires interventions autonomously; we just poll to discover them.
  useEffect(() => {
    if (!token || !isAdaptive || !isActive) return;
    const poll = () =>
      interventionService.getActive(token, sessionId).then(setActiveInterventions);
    poll();
    const id = setInterval(poll, 10_000);
    return () => clearInterval(id);
  }, [token, sessionId, isAdaptive, isActive]);

  // Track which chunk is near the top of the viewport so section summaries
  // fire at the user's current reading position, not the top of the document.
  useEffect(() => {
    const container = contentRef.current;
    if (!container || !isAdaptive) return;
    const onScroll = () => {
      // Approximate: progress through scroll height → chunk index
      const progress = container.scrollTop /
        Math.max(1, container.scrollHeight - container.clientHeight);
      const chunkCount = data?.chunks?.length ?? 1;
      viewportChunkIdxRef.current = Math.max(
        0,
        Math.floor(progress * chunkCount) - 1,
      );
    };
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => container.removeEventListener("scroll", onScroll);
  }, [isAdaptive, data?.chunks?.length]);

  // ── Badge award helper (stable — no deps, reads/writes only refs and setters) ──
  const awardBadge = useCallback((badgeId: BadgeId) => {
    if (earnedIdsRef.current.has(badgeId)) return; // already earned
    const def = BADGE_DEFS.find((b) => b.id === badgeId);
    if (!def) return;
    earnedIdsRef.current.add(badgeId);
    setSessionXP((p) => p + def.xp);        // bonus XP
    setEarnedBadges((prev) => [...prev, def]);
    setPendingBadge(def);                    // triggers popup
  }, []);

  // Accumulate XP from each new attentional-state window.
  // Tracks consecutive focus / non-distraction streaks and awards badges.
  // Badges only matter when gamification is active (checked via ref to avoid
  // stale closure issues with frequent state reads).
  const gamificationActiveRef = useRef(false);
  useEffect(() => {
    gamificationActiveRef.current = activeInterventions.some((i) => i.type === "gamification");

    // When a NEW gamification intervention fires (different id from last seen),
    // restart the journey: reset XP and clear the completed flag so widget shows.
    const current = activeInterventions.find((i) => i.type === "gamification");
    if (current && current.intervention_id !== lastGamificationIdRef.current) {
      const isFirstEver = lastGamificationIdRef.current === null;
      lastGamificationIdRef.current = current.intervention_id;
      // If this is a re-fire (not first time), restart the journey
      if (!isFirstEver) {
        setJourneyCompleted(false);
        setSessionXP(0);
      }
    }
  }, [activeInterventions]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!attentionalState) return;
    const seq = attentionalState.packet_seq;
    if (seq === lastXpSeqRef.current) return;
    lastXpSeqRef.current = seq;

    const state = attentionalState.primary_state;
    const isFocused  = state === "focused" || state === "hyperfocused";
    const isDrifting = state === "drifting" || state === "cognitive_overload";

    if (!hasFirstClassification) setHasFirstClassification(true);

    // XP only accumulates once gamification is active (Journey has started).
    // This means the XP counter starts at 0 and builds from the moment the
    // gamification intervention fires.
    if (gamificationActiveRef.current) {
      if (state === "focused")
        setSessionXP((p) => p + 3 + Math.floor(Math.random() * 3));
      else if (state === "hyperfocused")
        setSessionXP((p) => p + 5 + Math.floor(Math.random() * 4));
    }

    // ── Badge tracking (only when gamification intervention is running) ────
    if (gamificationActiveRef.current) {
      const isHyper = state === "hyperfocused";

      if (isFocused) {
        consecutiveFocusRef.current++;
        consecutiveCleanRef.current++;
        // Comeback kid: only after 3+ consecutive drifting/overload windows
        if (consecutiveDriftRef.current >= 3) {
          awardBadge("comeback_kid");
        }
        consecutiveDriftRef.current = 0;
      } else {
        consecutiveFocusRef.current = 0;
        if (isDrifting) {
          consecutiveCleanRef.current = 0;
          consecutiveDriftRef.current++;
        }
      }

      if (isHyper) {
        consecutiveHyperRef.current++;
      } else {
        consecutiveHyperRef.current = 0;
      }

      // Streak badges — fire exactly once at each threshold
      // first_focus_streak: 5 consecutive focused/hyperfocused
      if (consecutiveFocusRef.current === 5)  awardBadge("first_focus_streak");
      // deep_reader: 8 consecutive focused/hyperfocused
      if (consecutiveFocusRef.current === 8)  awardBadge("deep_reader");
      // focus_master: 4 consecutive hyperfocused windows
      if (consecutiveHyperRef.current  === 4)  awardBadge("focus_master");
      // no_distraction: 12 consecutive non-drifting windows
      if (consecutiveCleanRef.current === 12) awardBadge("no_distraction");
    }

    prevAttStateRef.current = state;
  }, [attentionalState, hasFirstClassification, awardBadge]);

  // Reading Marathon: award when the session timer reaches 15 minutes (900 s).
  // Uses timerSeconds so it fires regardless of whether the user has been
  // focused — it just requires staying in the session for that long.
  useEffect(() => {
    if (timerSeconds >= 900 && gamificationActiveRef.current) {
      awardBadge("reading_marathon");
    }
  }, [timerSeconds, awardBadge]);

  // Call the LLM trigger endpoint every 10 s once the first classification is ready.
  // After each call, refresh the active interventions list so new ones appear immediately.
  // Uses a "skip if already running" guard so slow LLM responses (20-30 s on low-end
  // hardware) never queue up concurrent calls.
  useEffect(() => {
    if (!token || !isAdaptive || !isActive || !hasFirstClassification) return;
    let inflight = false;
    const triggerAndRefresh = async () => {
      if (inflight) return;
      inflight = true;
      try {
        await interventionService.trigger(token, sessionId);
        const active = await interventionService.getActive(token, sessionId);
        setActiveInterventions(active);
      } catch {
        // Non-critical — LLM trigger failure must never crash the reading session
      } finally {
        inflight = false;
      }
    };
    triggerAndRefresh();
    const id = setInterval(triggerAndRefresh, 10_000);
    return () => clearInterval(id);
  }, [token, sessionId, isAdaptive, isActive, hasFirstClassification]);

  // Record where each new section_summary should be injected.
  // Uses content._chunk_index when available (set by trigger endpoint),
  // falling back to the current viewport position.
  useEffect(() => {
    const summaries = activeInterventions.filter((i) => i.type === "section_summary");
    setSummaryInsertions((prev) => {
      const next = new Map(prev);
      // Add new entries
      for (const s of summaries) {
        const id = s.intervention_id!;
        if (!next.has(id)) {
          const backendIdx = (s.content as Record<string, unknown> | null)
            ?._chunk_index as number | undefined;
          next.set(id, backendIdx ?? viewportChunkIdxRef.current);
        }
      }
      // Prune dismissed entries
      const activeIds = new Set(summaries.map((s) => s.intervention_id!));
      for (const id of next.keys()) {
        if (!activeIds.has(id)) next.delete(id);
      }
      return next;
    });
  }, [activeInterventions]);

  // User dismissed an intervention card — acknowledge on backend and remove from local state.
  const handleDismissIntervention = useCallback(async (interventionId: number) => {
    if (!token) return;
    // Optimistic removal first so the card disappears immediately
    setActiveInterventions((prev) => prev.filter((i) => i.intervention_id !== interventionId));
    await interventionService.acknowledge(token, sessionId, interventionId);
  }, [token, sessionId]);

  // ── Break suggestion handlers ──────────────────────────────────────────────
  //
  // IMPORTANT: we do NOT acknowledge the break_suggestion when the user confirms.
  // Acknowledging at confirm time would remove it from the backend's active list,
  // which would drop it from our local state poll and close the overlay mid-break.
  // Instead we acknowledge only when the break actually ends (auto-resume or dismiss).

  // Journey complete — called by JourneyWidget after user dismisses the congrats overlay.
  const handleJourneyComplete = useCallback(() => setJourneyCompleted(true), []);

  // User confirmed the break → pause session only; do NOT acknowledge yet.
  // Also sets breakActive so the audioscape is paused for the duration.
  const handleBreakConfirm = useCallback(async (_interventionId: number) => {
    if (!token) return;
    setBreakActive(true);
    try {
      const updated = await sessionService.pause(token, sessionId);
      setData((prev) => prev ? { ...prev, session: updated } : prev);
    } catch (e) {
      console.error("Break confirm failed:", e);
    }
  }, [token, sessionId]);

  // User dismissed the overlay without taking a break → acknowledge + close.
  const handleBreakDismiss = useCallback((interventionId: number) => {
    if (!token) return;
    setBreakActive(false);
    setActiveInterventions((prev) => prev.filter((i) => i.intervention_id !== interventionId));
    interventionService.acknowledge(token, sessionId, interventionId);
  }, [token, sessionId]);

  // Break timer finished or "I'm ready" → acknowledge (starts cooldown) + resume.
  const handleBreakAutoResume = useCallback(async (interventionId: number) => {
    if (!token) return;
    setBreakActive(false);
    setActiveInterventions((prev) => prev.filter((i) => i.intervention_id !== interventionId));
    try {
      await interventionService.acknowledge(token, sessionId, interventionId);
      const updated = await sessionService.resume(token, sessionId);
      setData((prev) => prev ? { ...prev, session: updated } : prev);
    } catch (e) {
      console.error("Break auto-resume failed:", e);
    }
  }, [token, sessionId]);

  // ──────────────────────────────────────────────────────────────────────────

  // Dev/test: immediately award any badge (for UI testing).
  const handleDevBadge = useCallback((badgeId: BadgeId) => {
    awardBadge(badgeId);
  }, [awardBadge]);

  // Dev/test: fire any intervention type manually, then immediately refresh the active list.
  const handleManualFire = useCallback(async (
    type: InterventionType,
    tier: InterventionTier,
  ) => {
    if (!token) return;
    try {
      const fired = await interventionService.manualTrigger(token, sessionId, type, tier);
      setActiveInterventions((prev) => {
        // Avoid duplicates in local state if the poll already picked it up
        if (prev.some((i) => i.intervention_id === fired.intervention_id)) return prev;
        return [fired, ...prev];
      });
    } catch (err) {
      console.error("Manual intervention fire failed:", err);
    }
  }, [token, sessionId]);

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

  // Derive break overlay and journey state
  const breakIntervention = activeInterventions.find((i) => i.type === "break_suggestion") ?? null;
  const sessionEnded = session.status === "ended" || session.status === "completed";

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
          {!isCalibration && (
            <AttentionalStateMeter state={attentionalState} />
          )}
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

      {/* Main content area + adaptive panel (side-by-side) */}
      <div
        id="reader-root"
        data-testid="reader-root"
        className={`reader-body${isAdaptive ? " reader-body--adaptive" : ""}`}
      >
        <main
          className={[
            "reader-content",
            isAdaptive && activeInterventions.some((i) => i.type === "text_reformat")
              ? "reader-content--chunked"
              : "",
          ].filter(Boolean).join(" ")}
          ref={contentRef}
        >
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
            <div key={chunk.id} data-chunk-idx={idx}>
              <ChunkCard
                chunk={chunk}
                chunkIndex={idx}
                docId={document_id}
                token={token}
              />
              {/* ── Section summary cards injected after their target chunk ── */}
              {isAdaptive && activeInterventions
                .filter(
                  (i) =>
                    i.type === "section_summary" &&
                    summaryInsertions.get(i.intervention_id!) === idx,
                )
                .map((i) => (
                  <SectionSummaryCard
                    key={i.intervention_id}
                    intervention={i}
                    onDismiss={handleDismissIntervention}
                  />
                ))}
            </div>
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

          {/* Export Bundle (training data) — dev only */}
          {DEV && <ExportBundleButton sessionId={sessionId} token={token} />}
        </main>

        {/* Adaptive assistant panel — mounted whenever mode=adaptive, never for baseline/calibration */}
        {isAdaptive && (
          <AssistantPanel
            panelRef={panelRef}
            open={panelOpen}
            onToggle={handlePanelToggle}
            activeInterventions={activeInterventions}
            onDismissIntervention={handleDismissIntervention}
            onManualFire={handleManualFire}
            onDevBadge={handleDevBadge}
            sessionXP={sessionXP}
            sessionEnded={sessionEnded}
            earnedBadges={earnedBadges}
            breakActive={breakActive}
            journeyCompleted={journeyCompleted}
            onJourneyComplete={handleJourneyComplete}
          />
        )}

        {/* Chime toast — floats in the viewport for ~5 s then auto-dismisses */}
        {isAdaptive && activeInterventions
          .filter((i) => i.type === "chime")
          .map((i) => (
            <ChimeWidget
              key={i.intervention_id}
              intervention={i}
              onDismiss={handleDismissIntervention}
            />
          ))
        }

        {/* Break suggestion full-screen overlay — rendered outside the panel */}
        {isAdaptive && breakIntervention && (
          <BreakSuggestionOverlay
            intervention={breakIntervention}
            onConfirm={handleBreakConfirm}
            onDismiss={handleBreakDismiss}
            onAutoResume={handleBreakAutoResume}
          />
        )}

        {/* Badge popup — shown when a badge is earned, stays on top of everything */}
        {isAdaptive && pendingBadge && (
          <BadgePopup
            badge={pendingBadge}
            onDismiss={() => setPendingBadge(null)}
          />
        )}
      </div>

      {/* Dev debug panels */}
      {DEV && <DebugPanel batch={lastBatch} warnings={warnings} />}
      {DEV && <DriftDebugPanel token={token} sessionId={sessionId} />}

      <style>{`
        /* ── Layout ── */
        .reader { display:flex; flex-direction:column; min-height:100vh; }

        /* reader-body is the flex row containing main content + optional panel */
        .reader-body { display:flex; flex:1; overflow:hidden; }
        .reader-body--adaptive .reader-content {
          flex: 1;
          min-width: 0;
          max-width: none;
          margin: 0;
        }

        /* ── Adaptive assistant panel ── */
        .assistant-panel {
          width: 490px;
          min-width: 490px;
          height: calc(100vh - 56px);
          background: var(--bg-surface);
          border-left: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          transition: width 0.2s ease, min-width 0.2s ease;
          overflow: hidden;
          flex-shrink: 0;
        }
        .assistant-panel--collapsed {
          width: 44px;
          min-width: 44px;
        }
        .assistant-panel__header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 12px;
          border-bottom: 1px solid var(--border);
          gap: 8px;
          min-height: 44px;
        }
        .assistant-panel__title {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          flex: 1;
        }
        .assistant-panel__toggle {
          background: none;
          border: 1px solid var(--border);
          border-radius: 4px;
          color: var(--text-muted);
          cursor: pointer;
          font-size: 14px;
          line-height: 1;
          padding: 2px 6px;
          flex-shrink: 0;
        }
        .assistant-panel__toggle:hover { color: var(--text); border-color: var(--text-muted); }
        .assistant-panel__body {
          flex: 1;
          padding: 16px 12px;
          overflow-y: auto;
        }
        .assistant-panel__placeholder {
          font-size: 12px;
          color: var(--text-muted);
          font-style: italic;
          margin: 0;
          text-align: center;
          padding-top: 24px;
        }
        .panel-section {
          margin-bottom: 20px;
        }
        .panel-section--muted {
          opacity: 0.6;
        }
        .panel-section__label {
          font-size: 11px;
          font-weight: 700;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin: 0 0 6px;
        }
        .panel-section__hint {
          font-size: 12px;
          color: var(--text-muted);
          line-height: 1.5;
          margin: 0 0 12px;
        }
        .panel-interact-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          width: 100%;
          padding: 10px 14px;
          background: var(--bg);
          border: 1.5px solid var(--border);
          border-radius: 8px;
          color: var(--text);
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s, border-color 0.15s, transform 0.1s;
          justify-content: center;
        }
        .panel-interact-btn:hover {
          background: var(--bg-hover, rgba(0,0,0,0.04));
          border-color: var(--accent, #4f6ef7);
        }
        .panel-interact-btn:active {
          transform: scale(0.97);
        }
        .panel-interact-btn--flash {
          background: var(--accent-subtle, rgba(79,110,247,0.12));
          border-color: var(--accent, #4f6ef7);
        }
        .panel-interact-btn__count {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: var(--accent, #4f6ef7);
          color: #fff;
          font-size: 11px;
          font-weight: 700;
          border-radius: 20px;
          padding: 1px 7px;
          min-width: 20px;
        }

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

        /* ── Attentional state meter ── */
        .attn-meter {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          border-left: 1px solid var(--border);
          padding-left: 10px;
          margin-left: 4px;
        }
        .attn-meter__label {
          color: var(--text-muted);
          font-size: 11px;
          white-space: nowrap;
        }
        .attn-meter--pending .attn-meter__waiting {
          color: var(--text-muted);
          font-size: 13px;
          letter-spacing: 0.1em;
        }
        .attn-meter__bars {
          display: flex;
          align-items: flex-end;
          gap: 3px;
          height: 32px;
        }
        .attn-meter__bar {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
          width: 20px;
          cursor: default;
        }
        .attn-meter__bar-fill {
          width: 100%;
          border-radius: 2px 2px 0 0;
          transition: height 0.4s ease;
          min-height: 3px;
        }
        .attn-meter__bar-label {
          font-size: 9px;
          color: var(--text-muted);
          font-family: var(--font-mono, monospace);
          letter-spacing: 0;
        }
        .attn-meter__bar--primary .attn-meter__bar-label {
          font-weight: 700;
        }
        .attn-meter__primary {
          font-size: 11px;
          font-weight: 700;
          white-space: nowrap;
          font-family: var(--font-mono, monospace);
        }

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

        /* ── Adaptive text chunking mode ── */
        /* Applied when a text_reformat intervention is active.
           Increases visual separation between paragraphs and adds a subtle
           left accent to each chunk — reduces visual density and supports
           chunked information processing (Mayer, 2009 Cognitive Theory of
           Multimedia Learning). Reverts automatically when intervention is
           acknowledged / dismissed. */
        .reader-content--chunked .chunk {
          margin-bottom: 44px;
          padding-left: 14px;
          border-left: 3px solid rgba(79,110,247,0.25);
          transition: border-left 0.3s ease, padding-left 0.3s ease;
        }
        .reader-content--chunked .chunk__p {
          line-height: 2.0;
          letter-spacing: 0.012em;
        }

        /* ── Content column ── */
        .reader-content {
          width: 100%;
          margin: 0 auto;
          padding: 48px 24px 100px;
          overflow-y: auto;
          height: calc(100vh - 56px);
        }

        /* ── Chunk base ── */
        /* Chunk wrapper — constrain ALL content (headings, paragraphs, figures)
           to the same 85ch column. font-size:19px ensures ch is calculated at
           the reading font size, matching calibration and adaptive mode exactly. */
        .chunk {
          margin-bottom: 28px;
          max-width: 85ch;
          font-size: 19px;
          margin-left: auto;
          margin-right: auto;
        }

        /* Adaptive mode: left-align the chunk block flush against the panel */
        .reader-body--adaptive .chunk {
          margin-left: 0;
          margin-right: 0;
        }

        /* Section summary card — same width/margin rules as .chunk */
        .section-summary-card {
          max-width: 85ch;
          font-size: 19px;
          margin-left: auto;
          margin-right: auto;
        }
        .reader-body--adaptive .section-summary-card {
          margin-left: 0;
          margin-right: 0;
          max-width: none;
        }

        /* Text */
        .chunk__p {
          font-size: 19px;
          line-height: 1.75;
          color: var(--text);
          margin: 0;
          max-width: 100%;
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
