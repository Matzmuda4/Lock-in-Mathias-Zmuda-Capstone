import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { documentService, type Chunk } from "../services/documentService";
import { sessionService, type Session, type SessionReaderData } from "../services/sessionService";

const API_BASE = "http://localhost:8000";

// ─── Elapsed timer ──────────────────────────────────────────────────────────

function useElapsedTimer(session: Session | null) {
  const [seconds, setSeconds] = useState(0);
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

// ─── Open PDF with auth ──────────────────────────────────────────────────────

async function openPdfWithAuth(docId: number, token: string) {
  try {
    const resp = await fetch(`${API_BASE}/documents/${docId}/file`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!resp.ok) throw new Error(`${resp.status}`);
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const tab = window.open(url, "_blank");
    // Keep the blob URL alive long enough for the browser to load it
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
    if (!tab) alert("Pop-up blocked — allow pop-ups for this page.");
  } catch (e) {
    alert(`Could not open PDF: ${e instanceof Error ? e.message : e}`);
  }
}

// ─── Chunk components ────────────────────────────────────────────────────────

const HEADING_LABELS = new Set(["title", "section_header"]);

function Caption({ text }: { text: string }) {
  return <p className="chunk__caption">{text}</p>;
}

function TextChunk({ chunk }: { chunk: Chunk }) {
  const label = (chunk.meta?.label as string | undefined) ?? "";
  const isHeading = HEADING_LABELS.has(label);

  return (
    <div className={`chunk chunk--text${isHeading ? " chunk--heading" : ""}`}>
      {isHeading ? <h2 className="chunk__h">{chunk.text}</h2> : <p className="chunk__p">{chunk.text}</p>}
    </div>
  );
}

function ImageChunk({ chunk, docId }: { chunk: Chunk; docId: number }) {
  const assetId = chunk.meta?.asset_id as number | undefined;
  const caption = chunk.meta?.caption as string | undefined;

  if (!assetId) return null;

  return (
    <div className="chunk chunk--figure">
      <img
        src={`${API_BASE}/documents/${docId}/assets/${assetId}`}
        alt={caption ?? "Figure"}
        className="chunk__img"
        loading="lazy"
        onError={(e) => {
          const el = (e.target as HTMLElement).closest(".chunk--figure");
          if (el) (el as HTMLElement).style.display = "none";
        }}
      />
      {caption && <Caption text={caption} />}
    </div>
  );
}

function TableChunk({ chunk, docId }: { chunk: Chunk; docId: number }) {
  const assetId = chunk.meta?.asset_id as number | undefined;
  const caption = chunk.meta?.caption as string | undefined;

  return (
    <div className="chunk chunk--table">
      {assetId ? (
        <img
          src={`${API_BASE}/documents/${docId}/assets/${assetId}`}
          alt={caption ?? "Table"}
          className="chunk__img"
          loading="lazy"
          onError={(e) => {
            const el = e.target as HTMLImageElement;
            el.style.display = "none";
            el.nextElementSibling?.removeAttribute("style");
          }}
        />
      ) : null}
      {chunk.text && (
        <pre
          className="chunk__table-md"
          style={assetId ? { display: "none" } : undefined}
        >
          {chunk.text}
        </pre>
      )}
      {caption && <Caption text={caption} />}
    </div>
  );
}

function ChunkCard({ chunk, docId }: { chunk: Chunk; docId: number }) {
  const ct = (chunk.meta?.chunk_type as string | undefined) ?? "text";
  if (ct === "image") return <ImageChunk chunk={chunk} docId={docId} />;
  if (ct === "table") return <TableChunk chunk={chunk} docId={docId} />;
  return <TextChunk chunk={chunk} />;
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

  const timerDisplay = useElapsedTimer(data?.session ?? null);

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
  const isParsing = parse_status === "pending" || parse_status === "running";
  const parseFailed = parse_status === "failed";

  return (
    <div className="reader">
      {/* Top bar */}
      <header className="reader-bar">
        <button className="btn btn--ghost btn--sm" onClick={() => navigate("/")} type="button">
          ← Dashboard
        </button>

        <div className="reader-bar__center">
          <span className="reader-bar__title">{session.name}</span>
          <span className={`badge badge--${session.status}`}>{session.status}</span>
          <span className="reader-bar__mode">{session.mode}</span>
        </div>

        <div className="reader-bar__right">
          <span className="reader-bar__timer">{timerDisplay}</span>
          <SessionControls session={session} onAction={handleSessionAction} busy={actionBusy} />
          <button
            className="btn btn--ghost btn--sm"
            type="button"
            onClick={() => openPdfWithAuth(document_id, token!)}
          >
            View PDF ↗
          </button>
        </div>
      </header>

      {/* Content */}
      <main className="reader-content">
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

        {chunks.map((chunk) => (
          <ChunkCard key={chunk.id} chunk={chunk} docId={document_id} />
        ))}

        {!allLoaded && !isParsing && (
          <button className="btn btn--ghost btn--load-more" onClick={loadMore} disabled={loadingMore} type="button">
            {loadingMore ? <span className="spinner" /> : "Load more"}
          </button>
        )}
      </main>

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

        .session-controls { display:flex; align-items:center; gap:6px; }
        .session-controls__done { font-size:12px; color:var(--text-muted); font-style:italic; text-transform:capitalize; }
        .btn--accent { background:var(--accent); color:#fff; border-color:var(--accent); }
        .btn--accent:hover { opacity:0.88; }

        /* ── Content column ── */
        .reader-content {
          max-width: 740px;
          width: 100%;
          margin: 0 auto;
          padding: 40px 24px 80px;
        }

        /* ── Chunk base — no box, just spacing ── */
        .chunk {
          margin-bottom: 22px;
        }

        /* Page indicator */
        .chunk__page {
          display:block;
          font-size:10px;
          color:var(--text-faint);
          text-transform:uppercase;
          letter-spacing:0.07em;
          margin-bottom:4px;
        }

        /* Text */
        .chunk--text {}
        .chunk__p {
          font-size:15px;
          line-height:1.8;
          color:var(--text);
          margin:0;
        }

        /* Headings */
        .chunk--heading {
          margin-top:36px;
          margin-bottom:10px;
          padding-bottom:6px;
          border-bottom:1px solid var(--border);
        }
        .chunk__h {
          font-size:18px;
          font-weight:700;
          color:var(--text);
          margin:0;
          line-height:1.4;
        }

        /* Figure / image */
        .chunk--figure {
          margin-top:28px;
          margin-bottom:28px;
          text-align:center;
        }
        .chunk__img {
          max-width:100%;
          height:auto;
          display:block;
          margin:0 auto;
          border-radius:var(--radius-sm, 4px);
        }

        /* Table */
        .chunk--table {
          margin-top:28px;
          margin-bottom:28px;
        }
        .chunk__table-md {
          font-size:13px;
          line-height:1.6;
          color:var(--text);
          background:var(--bg-surface);
          border:1px solid var(--border);
          border-radius:var(--radius);
          padding:14px 16px;
          overflow-x:auto;
          white-space:pre;
          font-family:var(--font-mono, monospace);
        }

        /* Caption (figure / table label) */
        .chunk__caption {
          font-size:13px;
          color:var(--text-muted);
          font-style:italic;
          text-align:center;
          margin:6px 0;
        }

        /* Notices */
        .reader-notice {
          display:flex; align-items:center; gap:10px;
          color:var(--text-muted); font-size:14px;
          padding:20px; margin-bottom:24px;
          border-radius:var(--radius);
          border:1px dashed var(--border);
        }
        .reader-notice--error { color:var(--error); border-color:rgba(239,68,68,0.4); }

        /* Load more */
        .btn--load-more { width:100%; margin-top:8px; }
      `}</style>
    </div>
  );
}
