/**
 * UserStudyPage.tsx
 * ─────────────────
 * Full user study examination pipeline. Participants complete both conditions
 * in a fixed order: Baseline → Break → Adaptive.
 *
 * Steps:
 *   1   Welcome & study overview
 *   2   Informed consent
 *   3   Researcher setup (participant ID, document selection, condition order)
 *   4   Demographics questionnaire
 *   5   Pre-study attitudes questionnaire
 *   6   Calibration check (must be calibrated before proceeding)
 *   7   Baseline task instructions
 *   8   Baseline session active (reader opens; page waits for return)
 *   9   Post-baseline NASA-TLX
 *  10   Scheduled break (timer)
 *  11   Adaptive task instructions
 *  12   Adaptive session active
 *  13   Post-adaptive NASA-TLX
 *  14   System Usability Scale (SUS)
 *  15   Post-experiment questionnaire
 *  16   Completion + data export
 *
 * EDITING QUESTIONS: Edit apps/desktop/src/study/StudyConfig.ts — no need to
 * touch this file.
 */

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "../contexts/AuthContext";
import { authService } from "../services/authService";
import { calibrationService } from "../services/calibrationService";
import { sessionService } from "../services/sessionService";
import { apiRequest } from "../services/apiClient";

// localStorage for the participant token — shared across tabs (calibration tab needs it).
// sessionStorage for credentials — same-tab only, fine for re-login on refresh.
const STUDY_TOKEN_KEY = "study_participant_token";
const STUDY_CREDS_KEY = "study_participant_creds";

import {
  STUDY_PARAMS,
  DEMOGRAPHICS_QUESTIONS,
  PRE_STUDY_QUESTIONS,
  POST_EXPERIMENT_QUESTIONS,
  SUS_QUESTIONS_LIST,
  NASA_TLX_DIMENSIONS,
  computeRawTlx,
  computeSus,
  susGrade,
} from "../study/StudyConfig";

import {
  QuestionRenderer,
  defaultNasaValues,
  getMissingRequired,
  type Answers,
} from "../study/QuestionRenderer";

import {
  buildStudyExport,
  downloadStudyFiles,
  saveExportToServer,
} from "../study/StudyDataExport";

// ── Types ─────────────────────────────────────────────────────────────────────

type Step = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16;

const STEP_LABELS: string[] = [
  "Welcome", "Consent", "Setup", "Demographics", "Pre-Survey",
  "Calibration", "Baseline Setup", "Baseline Session", "Post-Baseline",
  "Break", "Adaptive Setup", "Adaptive Session", "Post-Adaptive",
  "Usability", "Post-Survey", "Complete",
];

const TOTAL_STEPS = 16;

// ── Layout primitives ─────────────────────────────────────────────────────────

function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{
      background: "var(--bg-surface)", border: "1px solid var(--border)",
      borderRadius: "var(--radius-lg)", padding: "32px 36px", ...style,
    }}>
      {children}
    </div>
  );
}

function StepNav({ step, total = TOTAL_STEPS }: { step: Step; total?: number }) {
  const pct = Math.round(((step - 1) / (total - 1)) * 100);
  return (
    <div style={{ marginBottom: 28 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 12 }}>
        <span style={{ color: "var(--text-muted)" }}>Step {step} of {total}</span>
        <span style={{ color: "var(--accent)", fontWeight: 600 }}>{STEP_LABELS[step - 1]}</span>
      </div>
      <div style={{ height: 4, background: "var(--bg-hover)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: "var(--accent)", borderRadius: 2, transition: "width 0.3s" }} />
      </div>
    </div>
  );
}

function NavButtons({
  onBack, onNext, nextLabel = "Continue →", nextDisabled = false, backHidden = false,
}: {
  onBack: () => void; onNext: () => void;
  nextLabel?: string; nextDisabled?: boolean; backHidden?: boolean;
}) {
  return (
    <div style={{ display: "flex", gap: 12, marginTop: 28 }}>
      {!backHidden && (
        <button className="btn btn--ghost" onClick={onBack} style={{ flex: 1 }}>← Back</button>
      )}
      <button className="btn btn--primary" onClick={onNext}
        disabled={nextDisabled}
        style={{ flex: backHidden ? 1 : 2, opacity: nextDisabled ? 0.4 : 1 }}>
        {nextLabel}
      </button>
    </div>
  );
}

function PhaseBadge({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      fontSize: 11, fontWeight: 700, padding: "3px 8px", borderRadius: 4,
      background: color + "22", color, border: `1px solid ${color}44`,
      textTransform: "uppercase", letterSpacing: "0.05em",
    }}>
      {label}
    </span>
  );
}

// ── Step screens ──────────────────────────────────────────────────────────────

// STEP 1
function StepWelcome({ onNext }: { onNext: () => void }) {
  return (
    <Card>
      <div style={{ textAlign: "center", marginBottom: 28 }}>
        <div style={{ fontSize: 48, marginBottom: 10 }}>🔒</div>
        <h1 style={{ fontSize: 26, fontWeight: 800, marginBottom: 8 }}>{STUDY_PARAMS.studyTitle}</h1>
        <p style={{ color: "var(--text-muted)", fontSize: 14, maxWidth: 520, margin: "0 auto", lineHeight: 1.7 }}>
          This study investigates whether AI-generated, attentionally-adaptive interventions improve
          reading engagement and reduce cognitive load during digital reading, with a focus on
          participants with ADHD or attention difficulties.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 24 }}>
        {[
          { icon: "⏱", label: "Total duration", value: `~${STUDY_PARAMS.totalDurationMinutes} minutes` },
          { icon: "📖", label: "Reading tasks",  value: "Two 15-minute sessions" },
          { icon: "📋", label: "Surveys",        value: "Short before and after each task" },
          { icon: "🔒", label: "Privacy",        value: "All data stays on this device" },
        ].map(({ icon, label, value }) => (
          <div key={label} style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "14px 16px", border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 20, marginBottom: 6 }}>{icon}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>{label}</div>
            <div style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>{value}</div>
          </div>
        ))}
      </div>

      <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", padding: "18px 20px", marginBottom: 24 }}>
        <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>You will complete the following in order:</p>
        <ol style={{ paddingLeft: 20, fontSize: 13, lineHeight: 2, margin: 0, color: "var(--text-muted)" }}>
          <li>Read and sign the informed consent form.</li>
          <li>Complete a short demographics and pre-study questionnaire.</li>
          <li>Read an assigned article for 15 minutes without any AI support <PhaseBadge label="Baseline" color="#888" /></li>
          <li>Take a 5-minute break.</li>
          <li>Read the same (or comparable) article for 15 minutes with the adaptive AI system active <PhaseBadge label="Adaptive" color="#6366f1" /></li>
          <li>Complete post-task surveys after each session and a final questionnaire.</li>
        </ol>
      </div>

      <button className="btn btn--primary btn--full" onClick={onNext} style={{ fontSize: 15 }}>
        Begin →
      </button>
    </Card>
  );
}

// STEP 2
function StepConsent({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const [agreed, setAgreed] = useState(false);
  return (
    <Card>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 16 }}>Informed Consent</h2>

      <div style={{
        background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)",
        padding: "20px 24px", maxHeight: 360, overflowY: "auto", fontSize: 13, lineHeight: 1.8,
        color: "var(--text-muted)", marginBottom: 20,
      }}>
        <p><strong style={{ color: "var(--text)" }}>Study Title:</strong> {STUDY_PARAMS.studyTitle}</p>
        <p><strong style={{ color: "var(--text)" }}>Purpose:</strong> This study evaluates whether a system that monitors reading attention and delivers personalised, just-in-time interventions can improve reading engagement and reduce cognitive load compared to unassisted reading, with a focus on individuals with ADHD or attention difficulties.</p>
        <p><strong style={{ color: "var(--text)" }}>Procedure:</strong> You will complete two reading sessions on the same text: one without AI support (baseline) and one with the AI intervention system active (adaptive). Questionnaires are administered before and after each session.</p>
        <p><strong style={{ color: "var(--text)" }}>What the system measures:</strong> Eye-gaze proxy signals derived from mouse movement, scroll behaviour, and keyboard interaction. No video, audio, or identifiable biometric data is captured. All data is processed and stored locally on this device.</p>
        <p><strong style={{ color: "var(--text)" }}>Data use:</strong> Anonymised session data (attentional state classifications, drift trajectories, intervention events, questionnaire responses) may be used in the researcher's thesis and any resulting publications. Your participant ID will be used rather than your name.</p>
        <p><strong style={{ color: "var(--text)" }}>Voluntary participation:</strong> Participation is entirely voluntary. You may withdraw at any time without penalty.</p>
        <p><strong style={{ color: "var(--text)" }}>Researcher:</strong> {STUDY_PARAMS.researcherName}, {STUDY_PARAMS.degreeProgram}.</p>
        <p><strong style={{ color: "var(--text)" }}>Questions:</strong> Please ask the researcher before proceeding if you have any questions.</p>
      </div>

      <label style={{ display: "flex", gap: 12, alignItems: "flex-start", cursor: "pointer", marginBottom: 20 }}>
        <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)}
          style={{ width: 18, height: 18, accentColor: "var(--accent)", marginTop: 3, flexShrink: 0 }}
        />
        <span style={{ fontSize: 14, lineHeight: 1.6 }}>
          I have read and understood the information above. I voluntarily agree to participate and consent to the described use of my anonymised data.
        </span>
      </label>

      <NavButtons onBack={onBack} onNext={onNext} nextDisabled={!agreed} nextLabel="I Agree & Continue →" />
    </Card>
  );
}

// STEP 3
function StepResearcherSetup({
  participantId, setParticipantId,
  registerStatus, registerError, seedStatus,
  onNext, onBack,
}: {
  participantId: string; setParticipantId: (s: string) => void;
  registerStatus: "idle" | "loading" | "done" | "error";
  registerError: string;
  seedStatus: "idle" | "loading" | "done" | "error";
  onNext: () => void; onBack: () => void;
}) {
  const valid = participantId.trim().length >= 2;
  const busy  = registerStatus === "loading" || seedStatus === "loading";
  const allDone = registerStatus === "done" && seedStatus === "done";

  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>Researcher Setup</h2>
        <span style={{ fontSize: 12, background: "var(--bg-elevated)", border: "1px solid var(--border)", padding: "3px 8px", borderRadius: 4, color: "var(--text-muted)" }}>
          For researcher only — not shown to participant
        </span>
      </div>

      <div className="form-group" style={{ marginBottom: 20 }}>
        <label htmlFor="pid">Participant ID (assign sequentially, e.g. P01, P02…)</label>
        <input id="pid" type="text" value={participantId} placeholder="e.g. P01"
          onChange={(e) => setParticipantId(e.target.value)}
          disabled={registerStatus === "done"}
        />
        <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
          A fresh user account will be created for this participant so calibration
          always starts from scratch. Both study texts (baseline and adaptive) are
          automatically loaded — no manual document upload required.
        </p>
      </div>

      {/* Study texts info */}
      <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "14px 16px", marginBottom: 20, fontSize: 13 }}>
        <p style={{ fontWeight: 600, marginBottom: 8 }}>Study texts (auto-loaded)</p>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text-muted)" }}>
            <PhaseBadge label="Baseline" color="#888" />
            <span>Descartes — <em>Discourse on the Method</em> (Parts I &amp; II)</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text-muted)" }}>
            <PhaseBadge label="Adaptive" color="#6366f1" />
            <span>Russell — <em>The Problems of Philosophy</em> (Chapters I &amp; II)</span>
          </div>
        </div>
      </div>

      {/* Status messages */}
      {registerStatus === "done" && seedStatus === "idle" && (
        <div style={{ background: "rgba(74,222,128,0.08)", border: "1px solid rgba(74,222,128,0.3)", borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, color: "#4ade80", marginBottom: 16 }}>
          ✓ Participant account created. Loading study texts…
        </div>
      )}
      {seedStatus === "loading" && (
        <div style={{ background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.3)", borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, color: "#818cf8", marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
          <span className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} /> Preparing study documents…
        </div>
      )}
      {allDone && (
        <div style={{ background: "rgba(74,222,128,0.08)", border: "1px solid rgba(74,222,128,0.3)", borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, color: "#4ade80", marginBottom: 16 }}>
          ✓ Participant account created and study texts loaded. Ready to proceed.
        </div>
      )}
      {registerStatus === "error" && (
        <div style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, color: "#f87171", marginBottom: 16 }}>
          ✗ {registerError}
        </div>
      )}
      {seedStatus === "error" && (
        <div style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, color: "#f87171", marginBottom: 16 }}>
          ✗ Failed to load study texts. Check that the server has access to UserStudy/ConditionTexts/.
        </div>
      )}

      <NavButtons onBack={onBack} onNext={onNext}
        nextDisabled={!valid || busy || (!allDone && registerStatus !== "idle")}
        nextLabel={busy ? "Setting up…" : allDone ? "Continue →" : "Set Up & Continue →"}
      />
    </Card>
  );
}

// STEP 4 — Demographics
function StepQuestionnaire({
  title, subtitle, questions, answers, onChange,
  onNext, onBack, nextLabel,
}: {
  title: string; subtitle?: string;
  questions: typeof DEMOGRAPHICS_QUESTIONS;
  answers: Answers; onChange: (id: string, v: Answers[string]) => void;
  onNext: () => void; onBack: () => void;
  nextLabel?: string;
}) {
  const missing = getMissingRequired(questions, answers);
  return (
    <Card>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>{title}</h2>
      {subtitle && <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 20 }}>{subtitle}</p>}
      <QuestionRenderer questions={questions} answers={answers} onChange={onChange} />
      {missing.length > 0 && (
        <p style={{ fontSize: 12, color: "#f87171", marginTop: 8 }}>
          {missing.length} required question{missing.length > 1 ? "s" : ""} unanswered.
        </p>
      )}
      <NavButtons onBack={onBack} onNext={onNext}
        nextDisabled={missing.length > 0}
        nextLabel={nextLabel ?? "Continue →"}
      />
    </Card>
  );
}

// STEP 6 — Calibration gate
function StepCalibrationCheck({
  isCalibrated, onCheckAgain, onNext, onBack,
}: {
  isCalibrated: boolean | null;
  onCheckAgain: () => void;
  onNext: () => void;
  onBack: () => void;
}) {
  // Open calibration in a new tab so this study page stays intact.
  // Pass ?study_mode=true so CalibrationPage uses the participant token.
  function openCalibration() {
    window.open("/calibration?study_mode=true", "_blank", "noopener");
  }

  return (
    <Card>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 16 }}>Calibration</h2>

      {isCalibrated === null && (
        <p style={{ color: "var(--text-muted)" }}>Checking calibration status…</p>
      )}

      {isCalibrated === true && (
        <>
          <div style={{ background: "rgba(74,222,128,0.08)", border: "1px solid rgba(74,222,128,0.3)", borderRadius: "var(--radius-sm)", padding: "14px 18px", fontSize: 14, color: "#4ade80", marginBottom: 20 }}>
            ✓ Calibration complete. The system has a reading baseline for this participant.
          </div>
          <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 20 }}>
            You may now proceed to the first reading task.
          </p>
          <NavButtons onBack={onBack} onNext={onNext} nextLabel="Proceed to Baseline Session →" />
        </>
      )}

      {isCalibrated === false && (
        <>
          <div style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "var(--radius-sm)", padding: "14px 18px", fontSize: 14, color: "#f87171", marginBottom: 16 }}>
            ⚠ No calibration baseline found for this participant.
          </div>

          <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "16px 18px", marginBottom: 20, fontSize: 13, lineHeight: 1.7 }}>
            <p style={{ fontWeight: 600, marginBottom: 8 }}>Instructions for researcher:</p>
            <ol style={{ paddingLeft: 18, margin: 0, color: "var(--text-muted)" }}>
              <li>Click <strong style={{ color: "var(--text)" }}>Open Calibration</strong> — it opens in a new tab.</li>
              <li>Hand control to the participant and ask them to read the calibration text at their natural pace.</li>
              <li>When the participant clicks <strong style={{ color: "var(--text)" }}>Done</strong>, the results are shown and they click <strong style={{ color: "var(--text)" }}>Close tab &amp; return to study</strong>.</li>
              <li>Return here and click <strong style={{ color: "var(--text)" }}>Check Calibration Status</strong>. Once confirmed, click Continue.</li>
            </ol>
          </div>

          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn btn--ghost" onClick={onBack} style={{ flex: 1 }}>← Back</button>
            <button className="btn btn--ghost" onClick={onCheckAgain} style={{ flex: 1 }}>
              ↻ Check Status
            </button>
            <button className="btn btn--primary" onClick={openCalibration} style={{ flex: 2 }}>
              Open Calibration →
            </button>
          </div>
        </>
      )}
    </Card>
  );
}

// STEP 7/11 — Task instructions
function StepTaskInstructions({
  phase, documentTitle, onNext, onBack,
}: {
  phase: "baseline" | "adaptive"; documentTitle: string;
  onNext: () => void; onBack: () => void;
}) {
  const isAdaptive = phase === "adaptive";
  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>Reading Task Instructions</h2>
        <PhaseBadge label={phase} color={isAdaptive ? "#6366f1" : "#888"} />
      </div>

      <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", padding: "18px 20px", marginBottom: 20 }}>
        <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Please read carefully before starting:</p>
        <ul style={{ paddingLeft: 20, fontSize: 14, lineHeight: 2, margin: 0, color: "var(--text-muted)" }}>
          <li>You will read <strong style={{ color: "var(--text)" }}>"{documentTitle}"</strong> for approximately <strong style={{ color: "var(--text)" }}>{STUDY_PARAMS.targetReadingMinutes} minutes</strong>.</li>
          <li>Read at your <strong style={{ color: "var(--text)" }}>natural, normal pace</strong> — do not rush.</li>
          {isAdaptive ? (
            <>
              <li>The <strong style={{ color: "var(--accent)" }}>adaptive system is active</strong>. You may receive occasional prompts, cues, or sounds. Engage with them naturally — you are not obligated to act on each one.</li>
              <li>Interventions may appear as on-screen prompts, a brief focus chime, or background soundscapes.</li>
            </>
          ) : (
            <>
              <li>This is the <strong style={{ color: "var(--text)" }}>standard reading condition</strong>. The system will monitor your reading silently in the background but will not show any interventions.</li>
            </>
          )}
          <li>If you need to pause, use the <strong style={{ color: "var(--text)" }}>Pause</strong> button in the reader toolbar.</li>
          <li>When you have finished reading, click <strong style={{ color: "var(--text)" }}>End Session</strong> in the reader tab, then return to <strong style={{ color: "var(--text)" }}>this page</strong> and click <strong style={{ color: "var(--text)" }}>I Have Finished Reading</strong>.</li>
        </ul>
      </div>

      <div style={{ background: isAdaptive ? "var(--accent-glow)" : "var(--bg-elevated)", border: `1px solid ${isAdaptive ? "var(--accent)" : "var(--border)"}`, borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, marginBottom: 20 }}>
        <strong>Article:</strong> {documentTitle} &nbsp;·&nbsp;
        <strong>Mode:</strong> {isAdaptive ? "Adaptive (AI interventions enabled)" : "Baseline (no interventions)"}
      </div>

      <NavButtons onBack={onBack} onNext={onNext} nextLabel={`Start ${phase === "adaptive" ? "Adaptive" : "Baseline"} Session →`} />
    </Card>
  );
}

// STEP 8/12 — Session active
function StepSessionActive({
  phase, sessionId, onPostTaskClick,
}: {
  phase: "baseline" | "adaptive"; sessionId: number | null; onPostTaskClick: () => void;
}) {
  const isAdaptive = phase === "adaptive";
  return (
    <Card style={{ textAlign: "center" }}>
      <div style={{ fontSize: 48, marginBottom: 14 }}>📖</div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 10 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700 }}>Reading Session Active</h2>
        <PhaseBadge label={phase} color={isAdaptive ? "#6366f1" : "#888"} />
      </div>

      <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 24, maxWidth: 480, margin: "0 auto 24px" }}>
        The reading session is running in the reader window.
        When you have finished reading, click <strong>End Session</strong> in the reader
        and then return here and click the button below to continue.
      </p>

      {sessionId && (
        <>
          <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, color: "var(--text-muted)", marginBottom: 20, display: "inline-block" }}>
            Session ID: <strong style={{ color: "var(--text)" }}>#{sessionId}</strong>
          </div>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", marginBottom: 24 }}>
            <a href={`/sessions/${sessionId}/reader`} target="_blank" rel="noopener noreferrer"
              className="btn btn--ghost" style={{ textDecoration: "none" }}>
              Open Reader ↗
            </a>
          </div>
        </>
      )}

      <button className="btn btn--primary" onClick={onPostTaskClick} style={{ fontSize: 15, minWidth: 260 }}>
        I Have Finished Reading — Continue →
      </button>
    </Card>
  );
}

// STEP 9/13 — NASA-TLX
function StepNasaTlx({
  phase, nasaAnswers, onChange, onNext, onBack,
}: {
  phase: "baseline" | "adaptive";
  nasaAnswers: Record<string, number>;
  onChange: (dims: Record<string, number>) => void;
  onNext: () => void; onBack: () => void;
}) {
  const score = computeRawTlx(nasaAnswers);
  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>Post-Task Workload Rating</h2>
        <PhaseBadge label={phase} color={phase === "adaptive" ? "#6366f1" : "#888"} />
      </div>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 4 }}>
        NASA Task Load Index (Raw TLX) — Hart & Staveland (1988).
      </p>
      <p style={{ color: "var(--text-muted)", fontSize: 12, marginBottom: 20 }}>
        Rate your experience <strong style={{ color: "var(--text)" }}>during the {phase} reading session</strong> you just completed.
        Move each slider to reflect how you felt. 0 = Very Low / Perfect, 100 = Very High / Failure.
      </p>

      {NASA_TLX_DIMENSIONS.map((dim) => (
        <div key={dim.key} style={{ display: "grid", gridTemplateColumns: "160px 1fr 52px", alignItems: "center", gap: 16, marginBottom: 18 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700 }}>{dim.label}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2, lineHeight: 1.4 }}>{dim.hint}</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 10, color: "var(--text-faint)", minWidth: 40, textAlign: "right" }}>{dim.lowLabel}</span>
            <input type="range" min={0} max={100} step={5}
              value={nasaAnswers[dim.key] ?? 50}
              onChange={(e) => onChange({ ...nasaAnswers, [dim.key]: Number(e.target.value) })}
              style={{ flex: 1, accentColor: "var(--accent)" }}
            />
            <span style={{ fontSize: 10, color: "var(--text-faint)", minWidth: 40 }}>{dim.highLabel}</span>
          </div>
          <span style={{ fontSize: 15, fontWeight: 700, color: "var(--accent)", textAlign: "right" }}>
            {nasaAnswers[dim.key] ?? 50}
          </span>
        </div>
      ))}

      <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "14px 18px", marginTop: 4, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Raw TLX Composite Score</span>
        <span style={{ fontSize: 24, fontWeight: 800, color: score > 65 ? "#f87171" : score > 40 ? "#facc15" : "#4ade80" }}>
          {score} <span style={{ fontSize: 13, fontWeight: 400, color: "var(--text-muted)" }}>/ 100</span>
        </span>
      </div>

      <NavButtons onBack={onBack} onNext={onNext} nextLabel="Submit & Continue →" />
    </Card>
  );
}

// STEP 10 — Break
function StepBreak({ onDone }: { onDone: () => void }) {
  const totalSecs  = STUDY_PARAMS.breakMinutes * 60;
  const [secs, setSecs] = useState(totalSecs);
  const done = secs <= 0;

  useEffect(() => {
    if (done) return;
    const id = setInterval(() => setSecs((s) => Math.max(0, s - 1)), 1000);
    return () => clearInterval(id);
  }, [done]);

  const mm = String(Math.floor(secs / 60)).padStart(2, "0");
  const ss = String(secs % 60).padStart(2, "0");
  const pct = Math.round(((totalSecs - secs) / totalSecs) * 100);

  return (
    <Card style={{ textAlign: "center" }}>
      <div style={{ fontSize: 48, marginBottom: 12 }}>☕</div>
      <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: 8 }}>Scheduled Break</h2>
      <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 28 }}>
        Take a {STUDY_PARAMS.breakMinutes}-minute break before the adaptive reading session. Stand up, stretch, and relax.
      </p>

      <div style={{ fontSize: 64, fontWeight: 900, fontVariantNumeric: "tabular-nums", color: done ? "#4ade80" : "var(--text)", letterSpacing: "-0.03em", marginBottom: 20 }}>
        {done ? "Done!" : `${mm}:${ss}`}
      </div>

      <div style={{ height: 8, background: "var(--bg-hover)", borderRadius: 4, overflow: "hidden", maxWidth: 360, margin: "0 auto 24px" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: done ? "#4ade80" : "var(--accent)", borderRadius: 4, transition: "width 1s linear" }} />
      </div>

      <button className="btn btn--primary" onClick={onDone} disabled={!done}
        style={{ fontSize: 15, opacity: done ? 1 : 0.4 }}>
        {done ? "Continue to Adaptive Session →" : "Waiting for break to end…"}
      </button>

      {!done && (
        <button
          type="button"
          onClick={() => setSecs(0)}
          style={{
            marginTop: 12, background: "none", border: "none",
            color: "var(--text-muted)", fontSize: 12, cursor: "pointer",
            textDecoration: "underline", opacity: 0.6,
          }}
        >
          Skip break
        </button>
      )}
    </Card>
  );
}

// STEP 14 — SUS
function StepSus({
  susAnswers, onChange, onNext, onBack,
}: {
  susAnswers: (number | null)[];
  onChange: (idx: number, v: number) => void;
  onNext: () => void; onBack: () => void;
}) {
  const filled = susAnswers.every((v) => v !== null);
  const score  = computeSus(susAnswers);

  return (
    <Card>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>System Usability Scale</h2>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 4 }}>
        System Usability Scale (Brooke, 1996) — 10 items, 5-point agreement scale.
      </p>
      <p style={{ color: "var(--text-muted)", fontSize: 12, marginBottom: 20 }}>
        For each statement below, indicate how much you agree or disagree about the <strong style={{ color: "var(--text)" }}>Lock-in adaptive reading system</strong>.
      </p>

      {SUS_QUESTIONS_LIST.map((q, i) => {
        const v = susAnswers[i];
        return (
          <div key={i} style={{ marginBottom: 18, padding: "16px", background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", border: `1px solid ${v !== null ? "var(--border-focus)" : "var(--border)"}` }}>
            <p style={{ fontSize: 13, marginBottom: 12 }}>
              <span style={{ color: "var(--text-muted)", marginRight: 8, fontWeight: 700 }}>{i + 1}.</span>{q}
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              {([1, 2, 3, 4, 5] as const).map((rating) => (
                <button key={rating} type="button" onClick={() => onChange(i, rating)}
                  style={{
                    flex: 1, padding: "8px 4px", borderRadius: "var(--radius-sm)",
                    border: v === rating ? "2px solid var(--accent)" : "1px solid var(--border)",
                    background: v === rating ? "var(--accent-glow)" : "var(--bg-hover)",
                    color: v === rating ? "var(--accent)" : "var(--text-muted)",
                    fontSize: 10, cursor: "pointer", lineHeight: 1.3, textAlign: "center",
                    fontWeight: v === rating ? 700 : 400, whiteSpace: "pre-line",
                  }}>
                  {["Strongly\nDisagree", "Disagree", "Neutral", "Agree", "Strongly\nAgree"][rating - 1]}
                </button>
              ))}
            </div>
          </div>
        );
      })}

      {filled && (
        <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "14px 18px", marginBottom: 4, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>SUS Score</span>
          <div style={{ textAlign: "right" }}>
            <span style={{ fontSize: 26, fontWeight: 800, color: score >= 70 ? "#4ade80" : score >= 50 ? "#facc15" : "#f87171" }}>{score}</span>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}> / 100 — {susGrade(score)}</span>
          </div>
        </div>
      )}

      <NavButtons onBack={onBack} onNext={onNext} nextDisabled={!filled} nextLabel="Continue →" />
    </Card>
  );
}

// STEP 16 — Complete
function StepComplete({
  participantId, baselineNasa, adaptiveNasa, susAnswers,
  onExport, exporting, exportDone,
}: {
  participantId: string;
  baselineNasa:  Record<string, number>;
  adaptiveNasa:  Record<string, number>;
  susAnswers:    (number | null)[];
  onExport: () => void;
  exporting: boolean;
  exportDone: boolean;
}) {
  const bScore = computeRawTlx(baselineNasa);
  const aScore = computeRawTlx(adaptiveNasa);
  const delta  = aScore - bScore;
  const sus    = computeSus(susAnswers);

  // ── Multi-participant aggregation ────────────────────────────────────────
  const [aggStatus, setAggStatus] = useState<"idle" | "loading" | "done">("idle");
  const [aggSummary, setAggSummary] = useState<Record<string, unknown> | null>(null);

  async function handleAggregateFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setAggStatus("loading");
    try {
      const { StudyAggregator } = await import("../study/StudyAggregator");
      const agg = await StudyAggregator.fromFiles(files);
      setAggSummary(agg.summary() as unknown as Record<string, unknown>);
      agg.downloadCsv();
      agg.downloadSummary();
      setAggStatus("done");
    } catch {
      setAggStatus("idle");
    }
    e.target.value = "";
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Card style={{ textAlign: "center" }}>
        <div style={{ fontSize: 52, marginBottom: 10 }}>🎉</div>
        <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: 8 }}>Study Complete</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 28 }}>
          Thank you, <strong style={{ color: "var(--text)" }}>{participantId}</strong>. All tasks are complete.
          Please let the researcher know you have finished.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14, marginBottom: 24 }}>
          {[
            { label: "Baseline Workload",  value: bScore, color: "var(--text-muted)" },
            { label: "Adaptive Workload",  value: aScore, color: delta < 0 ? "#4ade80" : delta > 10 ? "#f87171" : "var(--text)" },
            { label: "SUS Score",          value: sus,    color: sus >= 70 ? "#4ade80" : sus >= 50 ? "#facc15" : "#f87171" },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "16px 10px", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: 30, fontWeight: 800, color }}>{value}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>{label}</div>
            </div>
          ))}
        </div>

        {delta !== 0 && (
          <div style={{
            background: delta < 0 ? "rgba(74,222,128,0.08)" : "rgba(248,113,113,0.08)",
            border: `1px solid ${delta < 0 ? "rgba(74,222,128,0.3)" : "rgba(248,113,113,0.3)"}`,
            borderRadius: "var(--radius-sm)", padding: "12px 16px",
            fontSize: 13, color: delta < 0 ? "#4ade80" : "#f87171", marginBottom: 24,
          }}>
            {delta < 0
              ? `Workload was ${Math.abs(delta)} points lower in the adaptive condition.`
              : `Workload was ${delta} points higher in the adaptive condition.`}
          </div>
        )}

        <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
          {!exportDone ? (
            <button className="btn btn--primary" onClick={onExport} disabled={exporting}
              style={{ minWidth: 220 }}>
              {exporting
                ? <><span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Exporting…</>
                : "↓ Export This Participant's Data"}
            </button>
          ) : (
            <div style={{ fontSize: 13, color: "#4ade80", padding: "10px 20px", border: "1px solid rgba(74,222,128,0.3)", borderRadius: "var(--radius-sm)" }}>
              ✓ 3 files exported — save to experimentresults/
            </div>
          )}
          <a href="/" className="btn btn--ghost" style={{ textDecoration: "none" }}>
            Return to Dashboard
          </a>
        </div>
      </Card>

      {/* Multi-participant aggregator */}
      <Card>
        <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}>Aggregate All Participants</h3>
        <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 16, lineHeight: 1.6 }}>
          Once all participants have completed the study and their data files are saved in
          <code style={{ background: "var(--bg-hover)", padding: "2px 6px", borderRadius: 3, marginLeft: 4 }}>experimentresults/</code>,
          load all <code style={{ background: "var(--bg-hover)", padding: "2px 6px", borderRadius: 3 }}>lockin_P*_study.json</code> files
          here to generate a combined CSV and summary JSON ready for analysis.
        </p>

        <label style={{
          display: "block", border: "2px dashed var(--border)", borderRadius: "var(--radius-sm)",
          padding: "24px", textAlign: "center", cursor: "pointer",
          background: "var(--bg-elevated)", marginBottom: 16,
        }}>
          <input type="file" accept=".json" multiple onChange={handleAggregateFiles}
            style={{ display: "none" }} />
          <span style={{ fontSize: 24 }}>📂</span>
          <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 8 }}>
            Click to select all participant JSON files (lockin_P*_study.json)
          </p>
          {aggStatus === "loading" && <p style={{ color: "var(--accent)", fontSize: 13 }}>Aggregating…</p>}
          {aggStatus === "done"    && <p style={{ color: "#4ade80", fontSize: 13 }}>✓ Aggregated — CSV and summary JSON downloaded</p>}
        </label>

        {aggSummary && (
          <div style={{ background: "var(--bg-hover)", borderRadius: "var(--radius-sm)", padding: "14px 16px" }}>
            <p style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
              Aggregation Preview
            </p>
            <pre style={{ fontSize: 11, color: "var(--text-muted)", margin: 0, overflowX: "auto", whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
              {JSON.stringify(aggSummary, null, 2)}
            </pre>
          </div>
        )}
      </Card>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function UserStudyPage() {
  const { token: researcherToken } = useAuth();

  // ── Navigation ──────────────────────────────────────────────────────────────
  const [step, setStep]     = useState<Step>(1);
  const next = () => setStep((s) => Math.min(s + 1, TOTAL_STEPS) as Step);
  const back = () => setStep((s) => Math.max(s - 1, 1) as Step);

  // ── Participant account ─────────────────────────────────────────────────────
  // Each participant gets their own user account so calibration is fresh per person.
  // The token is stored in sessionStorage so the ReaderPage can use it when
  // opened in a new tab with ?study_mode=baseline|adaptive.
  const [participantToken,     setParticipantToken]     = useState<string | null>(null);
  const [registerStatus,       setRegisterStatus]       = useState<"idle" | "loading" | "done" | "error">("idle");
  const [registerError,        setRegisterError]        = useState<string>("");

  // ── Study document seeding ───────────────────────────────────────────────────
  // After participant registration, the two condition documents are auto-seeded
  // under the participant's account. Subsequent calls are idempotent.
  const [seedStatus,      setSeedStatus]      = useState<"idle" | "loading" | "done" | "error">("idle");
  const [baselineDocId,   setBaselineDocId]   = useState<number | null>(null);
  const [adaptiveDocId,   setAdaptiveDocId]   = useState<number | null>(null);

  const token = participantToken ?? researcherToken;

  // ── Researcher setup ────────────────────────────────────────────────────────
  const [participantId,    setParticipantId]    = useState("");
  const [isCalibrated,     setIsCalibrated]     = useState<boolean | null>(null);

  // ── Sessions ────────────────────────────────────────────────────────────────
  const [baselineSessionId, setBaselineSessionId] = useState<number | null>(null);
  const [adaptiveSessionId, setAdaptiveSessionId] = useState<number | null>(null);

  // ── Questionnaire state ─────────────────────────────────────────────────────
  const [demoAnswers,    setDemoAnswers]    = useState<Answers>({});
  const [preAnswers,     setPreAnswers]     = useState<Answers>({});
  const [baselineNasa,   setBaselineNasa]  = useState<Record<string, number>>(defaultNasaValues());
  const [adaptiveNasa,   setAdaptiveNasa]  = useState<Record<string, number>>(defaultNasaValues());
  const [susAnswers,     setSusAnswers]    = useState<(number | null)[]>(SUS_QUESTIONS_LIST.map(() => null));
  const [postAnswers,    setPostAnswers]   = useState<Answers>({});

  // ── Export state ────────────────────────────────────────────────────────────
  const [exporting,   setExporting]   = useState(false);
  const [exportDone,  setExportDone]  = useState(false);

  // ── Auto-register participant account ──────────────────────────────────────
  // Each study run gets a UNIQUE internal account, guaranteeing fresh calibration
  // and isolated data every time regardless of what's in the database already.
  //
  // Internal account: pid_<unix_ms>@study.lockin (unique per run)
  // Researcher label: participantId (e.g. "P01") — used in all exports
  //
  // Credentials are persisted in sessionStorage so a page refresh mid-study
  // can re-authenticate without creating another account.
  async function registerParticipant(): Promise<string | null> {
    if (!participantId.trim()) return null;
    setRegisterStatus("loading");
    setRegisterError("");

    // Check if this setup run already has credentials stored (page was refreshed)
    const stored = sessionStorage.getItem(STUDY_CREDS_KEY);
    if (stored) {
      try {
        const { email, pass } = JSON.parse(stored) as { email: string; pass: string };
        const log = await authService.login(email, pass);
        const tok = log.access_token;
        localStorage.setItem(STUDY_TOKEN_KEY, tok);
        setParticipantToken(tok);
        setRegisterStatus("done");
        return tok;
      } catch {
        // Stored creds are stale — fall through to create a new account below
        sessionStorage.removeItem(STUDY_CREDS_KEY);
        localStorage.removeItem(STUDY_TOKEN_KEY);
      }
    }

    // Generate a unique internal account for this study run
    const pid   = participantId.trim().toLowerCase().replace(/[^a-z0-9_-]/g, "_");
    const uid   = `${pid}_${Date.now()}`;
    const email = `${uid}@study.lockin`;
    const pass  = `study_${uid}`;

    try {
      const reg = await authService.register(uid, email, pass);
      const tok = reg.access_token;
      // Persist credentials so mid-study refresh can re-login
      sessionStorage.setItem(STUDY_CREDS_KEY, JSON.stringify({ email, pass }));
      localStorage.setItem(STUDY_TOKEN_KEY, tok);
      setParticipantToken(tok);
      setRegisterStatus("done");
      return tok;
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Registration failed";
      setRegisterError(msg);
      setRegisterStatus("error");
      return null;
    }
  }

  // ── Seed study documents for the participant ───────────────────────────────
  // After registration, call the backend to insert the two condition text files
  // as Document + DocumentChunk records under the participant's account.
  async function seedStudyDocuments(tok: string): Promise<boolean> {
    setSeedStatus("loading");
    try {
      const result = await apiRequest<{ baseline_doc_id: number; adaptive_doc_id: number }>(
        "/study/seed-documents",
        { method: "POST", token: tok },
      );
      setBaselineDocId(result.baseline_doc_id);
      setAdaptiveDocId(result.adaptive_doc_id);
      setSeedStatus("done");
      return true;
    } catch {
      setSeedStatus("error");
      return false;
    }
  }

  // ── Check + auto-poll calibration on step 6 ────────────────────────────────
  // Polls every 5 s while uncalibrated so the researcher just closes the
  // calibration tab and this page automatically detects it.
  const checkCalibration = useCallback(async () => {
    if (!participantToken) return;
    try {
      const s = await calibrationService.getStatus(participantToken);
      setIsCalibrated(s.has_baseline);
    } catch {
      setIsCalibrated(false);
    }
  }, [participantToken]);

  useEffect(() => {
    if (step !== 6) return;
    checkCalibration();
  }, [step, checkCalibration]);

  useEffect(() => {
    if (step !== 6 || isCalibrated !== false) return;
    const id = setInterval(checkCalibration, 5000);
    return () => clearInterval(id);
  }, [step, isCalibrated, checkCalibration]);

  // ── Start a session and open reader in a new tab ──────────────────────────
  // The study page stays open. When the participant finishes reading and ends
  // the session, they close the reader tab and click "Continue" here.
  // Reader URL includes ?study_mode so ReaderPage enters participant-facing mode.
  async function startSession(mode: "baseline" | "adaptive") {
    const tok   = participantToken;
    const docId = mode === "baseline" ? baselineDocId : adaptiveDocId;
    if (!tok || docId === null) {
      alert("Study documents not ready. Please complete researcher setup first.");
      return;
    }
    const name = `Study ${mode.toUpperCase()} — ${participantId} — ${new Date().toLocaleDateString()}`;
    try {
      const session = await sessionService.start(tok, docId, name, mode);
      if (mode === "baseline") setBaselineSessionId(session.id);
      else                      setAdaptiveSessionId(session.id);
      // study_mode tells ReaderPage to use participant token + hide researcher UI
      window.open(`/sessions/${session.id}/reader?study_mode=${mode}`, "_blank", "noopener");
    } catch {
      alert("Failed to start session. Please check your connection and try again.");
    }
  }

  // ── Export ────────────────────────────────────────────────────────────────
  async function handleExport() {
    if (!token) return;
    setExporting(true);
    try {
      const result = await buildStudyExport({
        token,
        participantId,
        age:              String(demoAnswers["age_group"] ?? ""),
        readingFrequency: String(demoAnswers["reading_frequency"] ?? ""),
        educationLevel:   String(demoAnswers["education"] ?? ""),
        baselineSessionId,
        adaptiveSessionId,
        documentTitle:    docTitle,
        demographicAnswers: demoAnswers,
        preSurveyAnswers:   preAnswers,
        baselineNasaAnswers: baselineNasa,
        adaptiveNasaAnswers: adaptiveNasa,
        susAnswers,
        postAnswers,
        interventionLog: [],  // populated by future enhancement
      });
      // Save server-side into experimentresults/ (primary) then browser download as fallback
      const serverResult = await saveExportToServer(token, participantId, result);
      if (!serverResult) {
        console.warn("[export] Server save failed or offline — falling back to browser download only.");
      }
      downloadStudyFiles(participantId, result);
      setExportDone(true);
    } catch (err) {
      alert("Export failed. Check console for details.");
      console.error(err);
    } finally {
      setExporting(false);
    }
  }

  const docTitle = "Philosophical Texts (Descartes / Russell)";

  return (
    <div style={{
      minHeight: "100vh", background: "var(--bg-base)",
      display: "flex", flexDirection: "column", alignItems: "center",
      padding: "40px 20px 80px",
    }}>
      <div style={{ width: "100%", maxWidth: 780 }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 20 }}>🔒</span>
            <span style={{ fontWeight: 800, fontSize: 16 }}>Lock-in</span>
            <span style={{ color: "var(--text-faint)", fontSize: 13 }}>/ User Study</span>
          </div>
          <a href="/" style={{ fontSize: 12, color: "var(--text-muted)", textDecoration: "none" }}>← Dashboard</a>
        </div>

        <StepNav step={step} />

        {step === 1  && <StepWelcome onNext={next} />}
        {step === 2  && <StepConsent onNext={next} onBack={back} />}
        {step === 3  && (
          <StepResearcherSetup
            participantId={participantId} setParticipantId={setParticipantId}
            registerStatus={registerStatus} registerError={registerError}
            seedStatus={seedStatus}
            onNext={async () => {
              // Already fully set up — just proceed
              if (registerStatus === "done" && seedStatus === "done") { next(); return; }
              // Clear any stale credentials from a previous participant on this machine
              if (registerStatus === "idle") {
                sessionStorage.removeItem(STUDY_CREDS_KEY);
                localStorage.removeItem(STUDY_TOKEN_KEY);
              }
              // Register a fresh account, then seed documents under it
              const tok = await registerParticipant();
              if (!tok) return;
              const seeded = await seedStudyDocuments(tok);
              if (seeded) next();
            }}
            onBack={back}
          />
        )}
        {step === 4  && (
          <StepQuestionnaire
            title="Demographics" subtitle="The following questions help us describe the study sample."
            questions={DEMOGRAPHICS_QUESTIONS} answers={demoAnswers}
            onChange={(id, v) => setDemoAnswers((p) => ({ ...p, [id]: v }))}
            onNext={next} onBack={back}
          />
        )}
        {step === 5  && (
          <StepQuestionnaire
            title="Pre-Study Attitudes"
            subtitle="These questions measure your attitudes toward AI and reading tools before the study begins."
            questions={PRE_STUDY_QUESTIONS} answers={preAnswers}
            onChange={(id, v) => setPreAnswers((p) => ({ ...p, [id]: v }))}
            onNext={next} onBack={back}
          />
        )}
        {step === 6  && (
          <StepCalibrationCheck
            isCalibrated={isCalibrated}
            onCheckAgain={checkCalibration}
            onNext={next}
            onBack={back}
          />
        )}
        {step === 7  && (
          <StepTaskInstructions phase="baseline"
            documentTitle="Discourse on the Method — René Descartes (Parts I & II)"
            onNext={() => { startSession("baseline"); setStep(8); }} onBack={back}
          />
        )}
        {step === 8  && (
          <StepSessionActive phase="baseline" sessionId={baselineSessionId}
            onPostTaskClick={next}
          />
        )}
        {step === 9  && (
          <StepNasaTlx phase="baseline" nasaAnswers={baselineNasa}
            onChange={setBaselineNasa} onNext={next} onBack={back}
          />
        )}
        {step === 10 && <StepBreak onDone={next} />}
        {step === 11 && (
          <StepTaskInstructions phase="adaptive"
            documentTitle="The Problems of Philosophy — Bertrand Russell (Chapters I & II)"
            onNext={() => { startSession("adaptive"); setStep(12); }} onBack={back}
          />
        )}
        {step === 12 && (
          <StepSessionActive phase="adaptive" sessionId={adaptiveSessionId}
            onPostTaskClick={next}
          />
        )}
        {step === 13 && (
          <StepNasaTlx phase="adaptive" nasaAnswers={adaptiveNasa}
            onChange={setAdaptiveNasa} onNext={next} onBack={back}
          />
        )}
        {step === 14 && (
          <StepSus susAnswers={susAnswers}
            onChange={(idx, v) => setSusAnswers((p) => p.map((x, i) => i === idx ? v : x))}
            onNext={next} onBack={back}
          />
        )}
        {step === 15 && (
          <StepQuestionnaire
            title="Post-Experiment Questionnaire"
            subtitle="These questions ask about your overall experience with the Lock-in system. There are no right or wrong answers."
            questions={POST_EXPERIMENT_QUESTIONS} answers={postAnswers}
            onChange={(id, v) => setPostAnswers((p) => ({ ...p, [id]: v }))}
            onNext={next} onBack={back} nextLabel="Submit & Complete →"
          />
        )}
        {step === 16 && (
          <StepComplete
            participantId={participantId}
            baselineNasa={baselineNasa} adaptiveNasa={adaptiveNasa}
            susAnswers={susAnswers}
            onExport={handleExport} exporting={exporting} exportDone={exportDone}
          />
        )}
      </div>
    </div>
  );
}
