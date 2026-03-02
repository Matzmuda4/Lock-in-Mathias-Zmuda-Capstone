import { ChangeEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { Document, documentService, type ParseJobStatus } from "../services/documentService";
import { Session, SessionMode, sessionService } from "../services/sessionService";

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat-card">
      <span className="stat-card__value">{value}</span>
      <span className="stat-card__label">{label}</span>
    </div>
  );
}

function ParseBadge({
  status,
  docId,
  onReparse,
}: {
  status: ParseJobStatus["status"] | undefined;
  docId: number;
  onReparse: (docId: number) => void;
}) {
  if (!status || status === "succeeded") return null;

  if (status === "pending" || status === "running") {
    return (
      <span className="parse-badge parse-badge--running">
        <span className="spinner" style={{ width: 10, height: 10, borderWidth: 1.5 }} />
        Parsing…
      </span>
    );
  }

  if (status === "failed") {
    return (
      <button
        className="parse-badge parse-badge--failed"
        onClick={() => onReparse(docId)}
        type="button"
        title="Click to retry parsing"
      >
        Parse failed — retry
      </button>
    );
  }

  return null;
}

function DocumentRow({
  doc,
  token,
  parseStatus,
  onDelete,
  onStartSession,
  onReparse,
}: {
  doc: Document;
  token: string;
  parseStatus: ParseJobStatus["status"] | undefined;
  onDelete: (id: number) => void;
  onStartSession: (doc: Document) => void;
  onReparse: (docId: number) => void;
}) {
  const [deleting, setDeleting] = useState(false);
  const [opening, setOpening] = useState(false);
  const sizeKb = Math.round(doc.file_size / 1024);

  async function handleDelete() {
    if (!confirm(`Delete "${doc.title}"?`)) return;
    setDeleting(true);
    try {
      await documentService.remove(token, doc.id);
      onDelete(doc.id);
    } catch {
      alert("Failed to delete document.");
      setDeleting(false);
    }
  }

  return (
    <div className="doc-row">
      <div
        className="doc-row__info doc-row__info--clickable"
        onClick={async () => {
          if (opening) return;
          setOpening(true);
          try {
            const r = await fetch(`http://localhost:8000/documents/${doc.id}/file`, {
              headers: { Authorization: `Bearer ${token}` },
            });
            const blob = await r.blob();
            const url = URL.createObjectURL(blob);
            window.open(url, "_blank");
          } finally {
            setOpening(false);
          }
        }}
        role="button"
        tabIndex={0}
        title={opening ? "Opening…" : "View PDF"}
      >
        <span className="doc-row__icon">📄</span>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <p className="doc-row__title">{doc.title}</p>
            <ParseBadge
              status={parseStatus}
              docId={doc.id}
              onReparse={onReparse}
            />
          </div>
          <p className="doc-row__meta">
            {doc.filename} · {sizeKb} KB ·{" "}
            {new Date(doc.uploaded_at).toLocaleDateString()}
          </p>
        </div>
      </div>
      <div className="doc-row__actions">
        <button
          className="btn btn--sm btn--ghost"
          onClick={() => onStartSession(doc)}
          type="button"
        >
          Start session
        </button>
        <button
          className="btn btn--sm btn--danger"
          onClick={handleDelete}
          disabled={deleting}
          type="button"
        >
          {deleting ? "…" : "Delete"}
        </button>
      </div>
    </div>
  );
}

function SessionCard({ session }: { session: Session }) {
  const navigate = useNavigate();
  const duration =
    session.duration_seconds != null
      ? `${Math.floor(session.duration_seconds / 60)}m ${session.duration_seconds % 60}s`
      : "—";

  // Use created_at for display — started_at changes on each resume
  const startDate = new Date(session.created_at).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div
      className="session-card card session-card--clickable"
      onClick={() => navigate(`/sessions/${session.id}/reader`)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && navigate(`/sessions/${session.id}/reader`)}
    >
      <div className="session-card__header">
        <span className="session-card__name">{session.name}</span>
        <span className={`badge badge--${session.status}`}>{session.status}</span>
      </div>
      <div className="session-card__meta">
        <span className="session-card__mode">{session.mode}</span>
        <span className="session-card__dot">·</span>
        <span>{startDate}</span>
        {session.duration_seconds != null && (
          <>
            <span className="session-card__dot">·</span>
            <span>{duration}</span>
          </>
        )}
      </div>
    </div>
  );
}


// ─── Upload modal ─────────────────────────────────────────────────────────────

function UploadModal({
  onClose,
  onUploaded,
  token,
}: {
  onClose: () => void;
  onUploaded: (doc: Document) => void;
  token: string;
}) {
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  function handleFile(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    if (f && !title) setTitle(f.name.replace(/\.pdf$/i, ""));
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) { setError("Please select a PDF file."); return; }
    setError(null);
    setUploading(true);
    try {
      const doc = await documentService.upload(token, title || file.name, file);
      onUploaded(doc);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">
          <h2 className="modal__title">Upload PDF</h2>
          <button className="modal__close" onClick={onClose} type="button">
            ×
          </button>
        </div>
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="form-group">
            <label htmlFor="doc-title">Title</label>
            <input
              id="doc-title"
              type="text"
              placeholder="e.g. Chapter 3 — Attention Models"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label htmlFor="doc-file">PDF file</label>
            <input
              id="doc-file"
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFile}
              style={{ color: "var(--text)" }}
            />
          </div>
          {error && <p className="error-banner">{error}</p>}
          <button
            type="submit"
            className="btn btn--primary btn--full"
            disabled={uploading}
          >
            {uploading ? <span className="spinner" /> : "Upload"}
          </button>
        </form>
      </div>
    </div>
  );
}

// ─── Start-session modal ──────────────────────────────────────────────────────

function StartSessionModal({
  doc,
  token,
  onClose,
  onStarted,
}: {
  doc: Document;
  token: string;
  onClose: () => void;
  onStarted: (session: Session) => void;
}) {
  const navigate = useNavigate();
  const [name, setName] = useState(`${doc.title} — ${new Date().toLocaleDateString()}`);
  const [mode, setMode] = useState<SessionMode>("baseline");
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setStarting(true);
    try {
      const session = await sessionService.start(token, doc.id, name, mode);
      onStarted(session);
      navigate(`/sessions/${session.id}/reader`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start session");
      setStarting(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">
          <h2 className="modal__title">New Session</h2>
          <button className="modal__close" onClick={onClose} type="button">
            ×
          </button>
        </div>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
          Document: <strong style={{ color: "var(--text)" }}>{doc.title}</strong>
        </p>
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="form-group">
            <label htmlFor="session-name">Session name</label>
            <input
              id="session-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label>Mode</label>
            <div className="mode-selector">
              {(["baseline", "adaptive"] as SessionMode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  className={`mode-btn ${mode === m ? "mode-btn--active" : ""}`}
                  onClick={() => setMode(m)}
                >
                  <span className="mode-btn__label">{m}</span>
                  <span className="mode-btn__desc">
                    {m === "baseline"
                      ? "Log only — no interventions"
                      : "AI interventions enabled"}
                  </span>
                </button>
              ))}
            </div>
          </div>
          {error && <p className="error-banner">{error}</p>}
          <button
            type="submit"
            className="btn btn--primary btn--full"
            disabled={starting}
          >
            {starting ? <span className="spinner" /> : "Start reading"}
          </button>
        </form>
      </div>

      <style>{`
        .mode-selector { display: flex; flex-direction: column; gap: 8px; }
        .mode-btn {
          background: var(--bg-elevated);
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          color: var(--text);
          cursor: pointer;
          padding: 12px 14px;
          text-align: left;
          transition: border-color 0.15s, background 0.15s;
        }
        .mode-btn:hover { background: var(--bg-hover); }
        .mode-btn--active {
          border-color: var(--accent);
          background: var(--accent-glow);
        }
        .mode-btn__label {
          display: block;
          font-weight: 600;
          text-transform: capitalize;
        }
        .mode-btn__desc {
          display: block;
          font-size: 12px;
          color: var(--text-muted);
          margin-top: 2px;
        }
      `}</style>
    </div>
  );
}

// ─── HomePage ─────────────────────────────────────────────────────────────────

export function HomePage() {
  const { user, token, logout } = useAuth();

  const [documents, setDocuments] = useState<Document[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [loadingSessions, setLoadingSessions] = useState(true);

  // parse_status keyed by document_id
  const [parseStatuses, setParseStatuses] = useState<
    Record<number, ParseJobStatus["status"]>
  >({});
  // IDs currently being polled (pending or running)
  const pollingIds = useRef<Set<number>>(new Set());

  const [showUpload, setShowUpload] = useState(false);
  const [sessionTarget, setSessionTarget] = useState<Document | null>(null);

  // ── Poll parse status for a single document until it reaches a terminal state
  const pollParseStatus = useCallback(
    (docId: number) => {
      if (!token || pollingIds.current.has(docId)) return;
      pollingIds.current.add(docId);

      let delay = 1000;
      let handle: ReturnType<typeof setTimeout>;

      const tick = async () => {
        try {
          const status = await documentService.getParseStatus(token, docId);
          setParseStatuses((prev) => ({ ...prev, [docId]: status.status }));
          if (status.status === "pending" || status.status === "running") {
            delay = Math.min(delay * 1.5, 5000); // backoff up to 5 s
            handle = setTimeout(tick, delay);
          } else {
            pollingIds.current.delete(docId);
          }
        } catch {
          pollingIds.current.delete(docId);
        }
      };

      handle = setTimeout(tick, delay);
      // Return a cleanup so the effect can cancel the loop on unmount
      return () => {
        clearTimeout(handle);
        pollingIds.current.delete(docId);
      };
    },
    [token],
  );

  const loadDocuments = useCallback(async () => {
    if (!token) return;
    try {
      const data = await documentService.list(token);
      setDocuments(data.documents);
      // Fetch initial parse status for each document
      data.documents.forEach(async (doc) => {
        try {
          const s = await documentService.getParseStatus(token, doc.id);
          setParseStatuses((prev) => ({ ...prev, [doc.id]: s.status }));
          if (s.status === "pending" || s.status === "running") {
            pollParseStatus(doc.id);
          }
        } catch {
          // document pre-dates Phase 4 — no parse job yet
        }
      });
    } finally {
      setLoadingDocs(false);
    }
  }, [token, pollParseStatus]);

  const loadSessions = useCallback(async () => {
    if (!token) return;
    try {
      const data = await sessionService.list(token);
      setSessions(data.sessions);
    } finally {
      setLoadingSessions(false);
    }
  }, [token]);

  useEffect(() => {
    loadDocuments();
    loadSessions();
  }, [loadDocuments, loadSessions]);

  // ── Stats ──
  const completed = sessions.filter((s) => s.status === "completed").length;
  const total = sessions.length;
  const completionRate =
    total > 0 ? `${Math.round((completed / total) * 100)}%` : "—";

  // ── Handlers ──
  function handleDocDeleted(id: number) {
    setDocuments((prev) => prev.filter((d) => d.id !== id));
    setParseStatuses((prev) => { const n = { ...prev }; delete n[id]; return n; });
    pollingIds.current.delete(id);
  }

  function handleUploaded(doc: Document) {
    setDocuments((prev) => [doc, ...prev]);
    setParseStatuses((prev) => ({ ...prev, [doc.id]: "pending" }));
    pollParseStatus(doc.id);
    setShowUpload(false);
  }

  async function handleReparse(docId: number) {
    if (!token) return;
    try {
      await documentService.reparse(token, docId);
      setParseStatuses((prev) => ({ ...prev, [docId]: "pending" }));
      pollParseStatus(docId);
    } catch (e) {
      alert(e instanceof Error ? e.message : "Reparse failed");
    }
  }

  function handleSessionStarted(session: Session) {
    setSessions((prev) => [session, ...prev]);
    setSessionTarget(null);
  }

  return (
    <div className="home">
      {/* ── Header ── */}
      <header className="home-header">
        <div className="home-header__brand">
          <span className="home-header__dot" />
          <span className="home-header__name">Lock‑In</span>
        </div>
        <div className="home-header__user">
          <span className="home-header__username">{user?.username}</span>
          <button
            className="btn btn--ghost btn--sm"
            onClick={logout}
            type="button"
          >
            Sign out
          </button>
        </div>
      </header>

      <main className="home-main">
        {/* ── Stats ── */}
        <section className="stats-row">
          <StatCard label="Documents" value={documents.length} />
          <StatCard label="Sessions" value={total} />
          <StatCard label="Completed" value={completed} />
          <StatCard label="Completion rate" value={completionRate} />
        </section>

        {/* ── Documents ── */}
        <section className="section">
          <div className="section__header">
            <h2 className="section__title">Documents</h2>
            <button
              className="btn btn--primary btn--sm"
              onClick={() => setShowUpload(true)}
              type="button"
            >
              + Upload PDF
            </button>
          </div>

          {loadingDocs ? (
            <div style={{ padding: 24, textAlign: "center" }}>
              <span className="spinner" />
            </div>
          ) : documents.length === 0 ? (
            <div className="empty-state">
              <span className="empty-state__icon">📚</span>
              <p>No documents yet. Upload a PDF to get started.</p>
            </div>
          ) : (
            <div className="doc-list">
              {documents.map((doc) => (
                <DocumentRow
                  key={doc.id}
                  doc={doc}
                  token={token!}
                  parseStatus={parseStatuses[doc.id]}
                  onDelete={handleDocDeleted}
                  onStartSession={setSessionTarget}
                  onReparse={handleReparse}
                />
              ))}
            </div>
          )}
        </section>

        {/* ── Sessions ── */}
        <section className="section">
          <div className="section__header">
            <h2 className="section__title">Recent sessions</h2>
          </div>

          {loadingSessions ? (
            <div style={{ padding: 24, textAlign: "center" }}>
              <span className="spinner" />
            </div>
          ) : sessions.length === 0 ? (
            <div className="empty-state">
              <span className="empty-state__icon">⏱</span>
              <p>No sessions yet. Pick a document and start reading.</p>
            </div>
          ) : (
            <div className="session-grid">
              {sessions.slice(0, 12).map((s) => (
                <SessionCard key={s.id} session={s} />
              ))}
            </div>
          )}
        </section>
      </main>

      {/* ── Modals ── */}
      {showUpload && (
        <UploadModal
          token={token!}
          onClose={() => setShowUpload(false)}
          onUploaded={handleUploaded}
        />
      )}
      {sessionTarget && (
        <StartSessionModal
          doc={sessionTarget}
          token={token!}
          onClose={() => setSessionTarget(null)}
          onStarted={handleSessionStarted}
        />
      )}

      <style>{`
        .home { min-height: 100vh; display: flex; flex-direction: column; }

        .home-header {
          background: var(--bg-surface);
          border-bottom: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 14px 28px;
          position: sticky;
          top: 0;
          z-index: 10;
        }

        .home-header__brand {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .home-header__dot {
          width: 8px;
          height: 8px;
          background: var(--accent);
          border-radius: 50%;
          box-shadow: 0 0 8px var(--accent);
        }

        .home-header__name {
          font-size: 16px;
          font-weight: 700;
          letter-spacing: -0.01em;
        }

        .home-header__user {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .home-header__username {
          color: var(--text-muted);
          font-size: 13px;
        }

        .home-main {
          flex: 1;
          max-width: 1000px;
          margin: 0 auto;
          padding: 32px 28px;
          width: 100%;
          display: flex;
          flex-direction: column;
          gap: 40px;
        }

        /* Stats */
        .stats-row {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
        }

        .stat-card {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .stat-card__value {
          font-size: 28px;
          font-weight: 700;
          letter-spacing: -0.02em;
        }

        .stat-card__label {
          font-size: 12px;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }

        /* Sections */
        .section { display: flex; flex-direction: column; gap: 16px; }

        .section__header {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .section__title {
          font-size: 16px;
          font-weight: 600;
        }

        /* Document list */
        .doc-list {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          overflow: hidden;
        }

        .doc-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 14px 20px;
          border-bottom: 1px solid var(--border);
        }

        .doc-row:last-child { border-bottom: none; }
        .doc-row:hover { background: var(--bg-hover); }

        .doc-row__info {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .doc-row__info--clickable {
          cursor: pointer;
          border-radius: var(--radius-sm);
          padding: 4px 6px;
          margin: -4px -6px;
          transition: background 0.12s;
        }
        .doc-row__info--clickable:hover { background: var(--bg-hover); }
        .doc-row__info--clickable:focus-visible {
          outline: 2px solid var(--accent);
          outline-offset: 2px;
        }

        .doc-row__icon { font-size: 18px; }

        .doc-row__title {
          font-weight: 500;
          font-size: 14px;
        }

        .doc-row__meta {
          font-size: 12px;
          color: var(--text-muted);
          margin-top: 2px;
        }

        .doc-row__actions { display: flex; gap: 8px; }

        /* Session grid */
        .session-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
          gap: 14px;
        }

        .session-card__header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 8px;
        }

        .session-card__name {
          font-weight: 600;
          font-size: 14px;
        }

        .session-card__meta {
          font-size: 12px;
          color: var(--text-muted);
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
          align-items: center;
        }

        .session-card__mode {
          text-transform: capitalize;
          font-style: italic;
        }

        .session-card__dot { color: var(--text-faint); }

        .session-card--clickable {
          cursor: pointer;
          transition: border-color 0.15s, background 0.15s;
        }
        .session-card--clickable:hover {
          background: var(--bg-hover);
          border-color: var(--accent);
        }
        .session-card--clickable:focus-visible {
          outline: 2px solid var(--accent);
          outline-offset: 2px;
        }

        /* Parse status badges */
        .parse-badge {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          border-radius: 20px;
          font-size: 11px;
          font-weight: 500;
          padding: 2px 8px;
          white-space: nowrap;
        }

        .parse-badge--running {
          background: rgba(99,102,241,0.12);
          color: var(--accent);
        }

        .parse-badge--failed {
          background: rgba(239,68,68,0.1);
          color: var(--error);
          border: none;
          cursor: pointer;
        }
        .parse-badge--failed:hover { background: rgba(239,68,68,0.2); }

        /* Empty state */
        .empty-state {
          background: var(--bg-surface);
          border: 1px dashed var(--border);
          border-radius: var(--radius);
          padding: 40px;
          text-align: center;
          color: var(--text-muted);
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 10px;
        }

        .empty-state__icon { font-size: 32px; }

        @media (max-width: 640px) {
          .stats-row { grid-template-columns: repeat(2, 1fr); }
          .doc-row { flex-direction: column; align-items: flex-start; gap: 10px; }
          .pdf-modal { width: 100vw; height: 100vh; border-radius: 0; }
        }
      `}</style>
    </div>
  );
}
