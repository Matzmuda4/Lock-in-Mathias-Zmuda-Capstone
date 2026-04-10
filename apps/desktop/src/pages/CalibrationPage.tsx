import { useCallback, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { calibrationService, type CalibrationStatus } from "../services/calibrationService";

// localStorage is shared across tabs — required so the calibration tab
// (opened via window.open) can read the participant token written by the study tab.
const STUDY_TOKEN_KEY = "study_participant_token";

export function CalibrationPage() {
  const { token: authToken } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const isStudyMode = searchParams.get("study_mode") === "true";
  const token = isStudyMode
    ? (localStorage.getItem(STUDY_TOKEN_KEY) ?? authToken)
    : authToken;

  const [status, setStatus] = useState<CalibrationStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const fetchStatus = useCallback(async () => {
    if (!token) return;
    try {
      const s = await calibrationService.getStatus(token);
      setStatus(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load calibration status");
    }
  }, [token]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleStart = useCallback(async () => {
    if (!token) return;
    setStarting(true);
    setError(null);
    try {
      const { session_id } = await calibrationService.start(token);
      const dest = `/sessions/${session_id}/calibration-reader${isStudyMode ? "?study_mode=true" : ""}`;
      navigate(dest);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start calibration");
      setStarting(false);
    }
  }, [token, navigate, isStudyMode]);

  // Text-file calibration: no parse job — start is always instant once the
  // file exists on disk.  "pending"/"running" states no longer occur.
  const isParsing = false;

  const canStart =
    !starting &&
    status?.calib_available === true;

  return (
    <div className="calib-page">
      <div className="calib-card">
        {/* Header */}
        <div className="calib-card__header">
          <h1 className="calib-card__title">Reading Calibration</h1>
          <p className="calib-card__subtitle">One-time setup · read at your own pace</p>
        </div>

        {/* Body */}
        <div className="calib-card__body">
          <p className="calib-card__desc">
            Before we can track your attention accurately, we need to measure
            your <strong>natural reading pace</strong>.
          </p>

          <ul className="calib-card__steps">
            <li>Read the calibration text at your normal, comfortable speed.</li>
            <li>Do not try to read faster or slower than usual.</li>
            <li>Click <strong>Done</strong> when you have finished reading — no time limit.</li>
            <li>Avoid switching tabs or leaving the window during calibration.</li>
          </ul>

          <p className="calib-card__note">
            Your baseline is stored locally and only used to personalise your
            focus tracking — no raw text is sent anywhere.
          </p>
        </div>

        {/* Status area */}
        {!status && !error && (
          <div className="calib-card__status">
            <span className="spinner" /> Checking calibration status…
          </div>
        )}

        {!status?.calib_available && status !== null && (
          <div className="calib-card__status calib-card__status--error">
            Calibration text file not found on the server. Please contact support.
          </div>
        )}

        {error && (
          <div className="calib-card__status calib-card__status--error">
            {error}
          </div>
        )}

        {/* Action */}
        <div className="calib-card__actions">
          <button
            className="btn btn--accent btn--lg"
            type="button"
            onClick={handleStart}
            disabled={!canStart}
          >
            {starting ? (
              <><span className="spinner" /> Starting…</>
            ) : (
              "Start Calibration"
            )}
          </button>
        </div>
      </div>

      <style>{`
        .calib-page {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg);
          padding: 24px;
        }

        .calib-card {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg, 12px);
          max-width: 560px;
          width: 100%;
          overflow: hidden;
        }

        .calib-card__header {
          padding: 32px 32px 0;
          border-bottom: 1px solid var(--border);
          padding-bottom: 20px;
          margin-bottom: 0;
        }

        .calib-card__title {
          font-size: 22px;
          font-weight: 700;
          color: var(--text);
          margin: 0 0 4px;
        }

        .calib-card__subtitle {
          font-size: 13px;
          color: var(--text-muted);
          margin: 0;
        }

        .calib-card__body {
          padding: 24px 32px;
        }

        .calib-card__desc {
          font-size: 15px;
          line-height: 1.7;
          color: var(--text);
          margin: 0 0 16px;
        }

        .calib-card__steps {
          font-size: 14px;
          line-height: 1.8;
          color: var(--text);
          margin: 0 0 16px;
          padding-left: 20px;
        }

        .calib-card__steps li {
          margin-bottom: 4px;
        }

        .calib-card__note {
          font-size: 13px;
          color: var(--text-muted);
          margin: 0;
          font-style: italic;
        }

        .calib-card__status {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 13px;
          color: var(--text-muted);
          padding: 12px 32px;
          border-top: 1px solid var(--border);
          background: var(--bg);
        }

        .calib-card__status--error {
          color: var(--error, #ef4444);
        }

        .calib-card__status--info {
          color: var(--accent);
        }

        .calib-card__actions {
          padding: 20px 32px 28px;
          border-top: 1px solid var(--border);
          display: flex;
          justify-content: flex-end;
        }

        .btn--lg {
          padding: 10px 28px;
          font-size: 15px;
          font-weight: 600;
          display: inline-flex;
          align-items: center;
          gap: 8px;
        }

        .btn--accent:disabled {
          opacity: 0.45;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
