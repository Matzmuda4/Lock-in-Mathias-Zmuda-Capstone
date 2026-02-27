import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

const API_BASE = "http://localhost:8000";

export function PdfViewerPage() {
  const { id } = useParams<{ id: string }>();
  const docId = Number(id);
  const { token } = useAuth();
  const navigate = useNavigate();

  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token || !docId) return;

    let cancelled = false;
    let objectUrl: string | null = null;

    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/documents/${docId}/file`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const blob = await resp.blob();
        objectUrl = URL.createObjectURL(blob);
        if (!cancelled) setUrl(objectUrl);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load PDF");
        }
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [token, docId]);

  if (error) {
    return (
      <div className="pdf-shell">
        <header className="pdf-shell__bar">
          <button
            className="btn btn--ghost btn--sm"
            type="button"
            onClick={() => navigate(-1)}
          >
            ← Back
          </button>
          <span className="pdf-shell__title">PDF viewer</span>
        </header>
        <main className="pdf-shell__body">
          <p className="error-banner">{error}</p>
        </main>
      </div>
    );
  }

  if (!url) {
    return (
      <div className="pdf-shell">
        <header className="pdf-shell__bar">
          <button
            className="btn btn--ghost btn--sm"
            type="button"
            onClick={() => navigate(-1)}
          >
            ← Back
          </button>
          <span className="pdf-shell__title">PDF viewer</span>
        </header>
        <main className="pdf-shell__body pdf-shell__body--center">
          <span className="spinner" />
        </main>
      </div>
    );
  }

  return (
    <div className="pdf-shell">
      <header className="pdf-shell__bar">
        <button
          className="btn btn--ghost btn--sm"
          type="button"
          onClick={() => navigate(-1)}
        >
          ← Back
        </button>
        <span className="pdf-shell__title">PDF viewer</span>
      </header>
      <main className="pdf-shell__body">
        <iframe
          src={url}
          title="PDF document"
          className="pdf-shell__frame"
        />
      </main>
      <style>{`
        .pdf-shell {
          display: flex;
          flex-direction: column;
          min-height: 100vh;
        }
        .pdf-shell__bar {
          background: var(--bg-surface);
          border-bottom: 1px solid var(--border);
          padding: 8px 16px;
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .pdf-shell__title {
          font-size: 14px;
          font-weight: 600;
        }
        .pdf-shell__body {
          flex: 1;
          background: #111;
        }
        .pdf-shell__body--center {
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .pdf-shell__frame {
          border: none;
          width: 100%;
          height: 100%;
        }
      `}</style>
    </div>
  );
}

