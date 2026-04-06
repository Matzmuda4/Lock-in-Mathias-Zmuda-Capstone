/**
 * UserStudyPage.tsx
 * ─────────────────
 * Guided multi-step examination page for the Lock-in user study.
 *
 * Steps:
 *   1  Welcome & study information
 *   2  Informed consent
 *   3  Participant details & session setup
 *   4  Pre-task questionnaire  (cognitive state + NASA-TLX baseline)
 *   5  Reading task instructions
 *   6  Reading session  (redirects to ReaderPage; returns here on completion)
 *   7  Post-task questionnaire (NASA-TLX + SUS + open feedback)
 *   8  Completion screen  (exportable summary for researcher)
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { documentService, type Document } from "../services/documentService";
import { sessionService, type SessionMode } from "../services/sessionService";

// ── Types ─────────────────────────────────────────────────────────────────────

type Step = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;

interface ParticipantInfo {
  participantId: string;
  group: "adaptive" | "baseline";
  age: string;
  readingFrequency: string;
}

interface NasaTlx {
  mental:    number; // Mental Demand
  physical:  number; // Physical Demand
  temporal:  number; // Temporal Demand
  effort:    number; // Effort
  frustration: number; // Frustration
  performance: number; // Performance (lower = better in NASA-TLX, we display as-is)
}

interface SusItem {
  q: string;
  value: number | null; // 1–5
}

const INITIAL_NASA: NasaTlx = {
  mental: 50, physical: 50, temporal: 50, effort: 50, frustration: 50, performance: 50,
};

const SUS_QUESTIONS: string[] = [
  "I think that I would like to use this system frequently.",
  "I found the system unnecessarily complex.",
  "I thought the system was easy to use.",
  "I think that I would need the support of a technical person to be able to use this system.",
  "I found the various functions in this system were well integrated.",
  "I thought there was too much inconsistency in this system.",
  "I would imagine that most people would learn to use this system very quickly.",
  "I found the system very cumbersome to use.",
  "I felt very confident using the system.",
  "I needed to learn a lot of things before I could get going with this system.",
];

function computeSus(items: SusItem[]): number {
  // Standard SUS formula: odd items −1, even items 5−value; sum × 2.5
  const filled = items.filter((i) => i.value !== null);
  if (filled.length < 10) return 0;
  let sum = 0;
  items.forEach((item, idx) => {
    const v = item.value ?? 3;
    sum += idx % 2 === 0 ? v - 1 : 5 - v;
  });
  return Math.round(sum * 2.5);
}

function computeNasaWl(tlx: NasaTlx): number {
  const vals = Object.values(tlx);
  return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StepIndicator({ current, total = 8 }: { current: Step; total?: number }) {
  const STEP_LABELS = ["Welcome", "Consent", "Setup", "Pre-Task", "Instructions", "Session", "Post-Task", "Complete"];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 0, marginBottom: 32 }}>
      {STEP_LABELS.map((label, i) => {
        const n = (i + 1) as Step;
        const done    = n < current;
        const active  = n === current;
        const future  = n > current;
        return (
          <div key={n} style={{ display: "flex", alignItems: "center", flex: n < total ? 1 : undefined }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div style={{
                width: 28, height: 28, borderRadius: "50%",
                background: done ? "var(--accent)" : active ? "var(--accent)" : "var(--bg-hover)",
                border: active ? "2px solid var(--accent)" : done ? "none" : "2px solid var(--border)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 12, fontWeight: 700, color: future ? "var(--text-faint)" : "#fff",
                opacity: future ? 0.45 : 1,
                flexShrink: 0,
              }}>
                {done ? "✓" : n}
              </div>
              <span style={{ fontSize: 10, color: active ? "var(--accent)" : "var(--text-faint)", whiteSpace: "nowrap", fontWeight: active ? 600 : 400 }}>
                {label}
              </span>
            </div>
            {n < total && (
              <div style={{
                flex: 1, height: 2, background: done ? "var(--accent)" : "var(--border)",
                margin: "0 4px", marginBottom: 18, minWidth: 12,
              }} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{
      background: "var(--bg-surface)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius-lg)",
      padding: "32px 36px",
      ...style,
    }}>
      {children}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 28 }}>
      <h3 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 14 }}>
        {title}
      </h3>
      {children}
    </div>
  );
}

function SliderRow({ label, hint, value, onChange }: {
  label: string; hint: string; value: number; onChange: (v: number) => void;
}) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "160px 1fr 48px", alignItems: "center", gap: 16, marginBottom: 14 }}>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600 }}>{label}</div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{hint}</div>
      </div>
      <input type="range" min={0} max={100} step={5} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ accentColor: "var(--accent)", width: "100%" }}
      />
      <span style={{ fontSize: 14, fontWeight: 700, color: "var(--accent)", textAlign: "right" }}>{value}</span>
    </div>
  );
}

function LikertRow({ question, value, onChange, idx }: {
  question: string; value: number | null; onChange: (v: number) => void; idx: number;
}) {
  const labels = ["Strongly\nDisagree", "Disagree", "Neutral", "Agree", "Strongly\nAgree"];
  return (
    <div style={{ marginBottom: 20, padding: "16px", background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)" }}>
      <p style={{ fontSize: 13, marginBottom: 12 }}>
        <span style={{ color: "var(--text-muted)", marginRight: 8, fontWeight: 600 }}>{idx + 1}.</span>
        {question}
      </p>
      <div style={{ display: "flex", gap: 8 }}>
        {[1, 2, 3, 4, 5].map((v) => (
          <button key={v} type="button" onClick={() => onChange(v)}
            style={{
              flex: 1, padding: "8px 4px", borderRadius: "var(--radius-sm)",
              border: value === v ? "2px solid var(--accent)" : "1px solid var(--border)",
              background: value === v ? "var(--accent-glow)" : "var(--bg-hover)",
              color: value === v ? "var(--accent)" : "var(--text-muted)",
              fontSize: 10, cursor: "pointer", lineHeight: 1.3, textAlign: "center",
              fontWeight: value === v ? 700 : 400, whiteSpace: "pre-line",
            }}>
            {labels[v - 1]}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Step screens ──────────────────────────────────────────────────────────────

function StepWelcome({ onNext }: { onNext: () => void }) {
  return (
    <Card>
      <div style={{ textAlign: "center", marginBottom: 32 }}>
        <div style={{ fontSize: 48, marginBottom: 12 }}>🔒</div>
        <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 8 }}>Lock-in User Study</h1>
        <p style={{ color: "var(--text-muted)", fontSize: 15, maxWidth: 520, margin: "0 auto" }}>
          Thank you for participating. This study investigates whether AI-generated adaptive interventions
          can support attentional regulation during digital reading.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 28 }}>
        {[
          { icon: "⏱", label: "Duration", value: "~25–35 minutes" },
          { icon: "📖", label: "Task", value: "Read an assigned article" },
          { icon: "🧠", label: "System", value: "Attentional state monitoring" },
          { icon: "🔒", label: "Privacy", value: "All data stays on this device" },
        ].map(({ icon, label, value }) => (
          <div key={label} style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "14px 16px", border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 20, marginBottom: 6 }}>{icon}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>{label}</div>
            <div style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>{value}</div>
          </div>
        ))}
      </div>

      <Section title="What will happen">
        <ol style={{ paddingLeft: 20, color: "var(--text-muted)", fontSize: 14, lineHeight: 1.8, margin: 0 }}>
          <li>Read and sign the informed consent form.</li>
          <li>Complete a short pre-task questionnaire about your current mental state.</li>
          <li>Read an assigned article for approximately 15 minutes. The system will monitor your reading and may offer subtle support prompts.</li>
          <li>Complete a post-task questionnaire about your experience.</li>
        </ol>
      </Section>

      <button className="btn btn--primary btn--full" onClick={onNext} style={{ fontSize: 15 }}>
        Begin →
      </button>
    </Card>
  );
}

function StepConsent({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const [agreed, setAgreed] = useState(false);
  return (
    <Card>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 20 }}>Informed Consent</h2>

      <div style={{
        background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)",
        padding: "20px 24px", maxHeight: 340, overflowY: "auto", fontSize: 13, lineHeight: 1.8,
        color: "var(--text-muted)", marginBottom: 24,
      }}>
        <p><strong style={{ color: "var(--text)" }}>Study Title:</strong> Lock-in — AI-Driven Attentional Monitoring and Adaptive Intervention in Digital Reading</p>
        <p><strong style={{ color: "var(--text)" }}>Purpose:</strong> This study evaluates whether a system that monitors reading attention and delivers personalised, just-in-time interventions can improve reading engagement and reduce cognitive load compared to unassisted reading.</p>
        <p><strong style={{ color: "var(--text)" }}>What the system collects:</strong> Eye-gaze proxy signals (mouse/keyboard patterns and scroll behaviour), which are used locally to classify your attentional state. No video, audio, or identifiable biometric data is collected. All data remains on this device and is never transmitted externally.</p>
        <p><strong style={{ color: "var(--text)" }}>Voluntary participation:</strong> Participation is entirely voluntary. You may withdraw at any time by closing the application without penalty.</p>
        <p><strong style={{ color: "var(--text)" }}>Data use:</strong> Anonymised session data (attentional state logs, intervention events, and questionnaire responses) may be used in the thesis and any resulting publications. Your participant ID will be used instead of your name.</p>
        <p><strong style={{ color: "var(--text)" }}>Researcher:</strong> Mathias Zmuda — Computing Science BSc, thesis submitted in partial fulfilment of the degree requirements.</p>
        <p><strong style={{ color: "var(--text)" }}>Questions:</strong> Please ask the researcher if you have any questions before agreeing.</p>
      </div>

      <label style={{ display: "flex", gap: 12, alignItems: "flex-start", cursor: "pointer", marginBottom: 24 }}>
        <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)}
          style={{ width: 18, height: 18, accentColor: "var(--accent)", marginTop: 2, flexShrink: 0 }}
        />
        <span style={{ fontSize: 14 }}>
          I have read and understood the information above. I voluntarily agree to participate in this study and consent to the use of my anonymised data as described.
        </span>
      </label>

      <div style={{ display: "flex", gap: 12 }}>
        <button className="btn btn--ghost" onClick={onBack} style={{ flex: 1 }}>← Back</button>
        <button className="btn btn--primary" onClick={onNext} disabled={!agreed} style={{ flex: 2, opacity: agreed ? 1 : 0.4 }}>
          I Agree & Continue →
        </button>
      </div>
    </Card>
  );
}

function StepSetup({
  info, onChange, documents, loadingDocs, onNext, onBack,
}: {
  info: ParticipantInfo;
  onChange: (p: ParticipantInfo) => void;
  documents: Document[];
  loadingDocs: boolean;
  selectedDoc: Document | null;
  onSelectDoc: (d: Document) => void;
  onNext: () => void;
  onBack: () => void;
}) {
  const valid = info.participantId.trim().length >= 2 && info.age.trim() !== "";
  return (
    <Card>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Participant Details</h2>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 24 }}>
        This information is used to organise study records. All entries are anonymised.
      </p>

      <Section title="Identification">
        <div className="form-group" style={{ marginBottom: 16 }}>
          <label htmlFor="pid">Participant ID (assigned by researcher)</label>
          <input id="pid" type="text" value={info.participantId} placeholder="e.g. P01"
            onChange={(e) => onChange({ ...info, participantId: e.target.value })}
          />
        </div>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label htmlFor="age">Age</label>
          <input id="age" type="number" min={16} max={99} value={info.age} placeholder="e.g. 22"
            onChange={(e) => onChange({ ...info, age: e.target.value })}
          />
        </div>
      </Section>

      <Section title="Background">
        <div className="form-group">
          <label>How often do you read academic or technical texts?</label>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 6 }}>
            {[
              ["rarely", "Rarely (less than once a week)"],
              ["sometimes", "Sometimes (1–3 times a week)"],
              ["often", "Often (most days)"],
              ["daily", "Daily (core part of my work/study)"],
            ].map(([val, label]) => (
              <label key={val} style={{ display: "flex", gap: 10, cursor: "pointer", alignItems: "center" }}>
                <input type="radio" name="freq" value={val} checked={info.readingFrequency === val}
                  onChange={() => onChange({ ...info, readingFrequency: val })}
                  style={{ accentColor: "var(--accent)" }}
                />
                <span style={{ fontSize: 13 }}>{label}</span>
              </label>
            ))}
          </div>
        </div>
      </Section>

      <Section title="Condition (researcher assigns)">
        <div style={{ display: "flex", gap: 12 }}>
          {(["adaptive", "baseline"] as const).map((g) => (
            <button key={g} type="button"
              className={`mode-btn ${info.group === g ? "mode-btn--active" : ""}`}
              onClick={() => onChange({ ...info, group: g })}
              style={{
                flex: 1, padding: "14px", borderRadius: "var(--radius-sm)", textAlign: "left", cursor: "pointer",
                background: "var(--bg-elevated)",
                border: info.group === g ? "2px solid var(--accent)" : "1px solid var(--border)",
              }}>
              <span style={{ display: "block", fontWeight: 700, fontSize: 14, textTransform: "capitalize", color: info.group === g ? "var(--accent)" : "var(--text)" }}>{g}</span>
              <span style={{ display: "block", fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                {g === "adaptive" ? "AI interventions enabled" : "Log only — no interventions"}
              </span>
            </button>
          ))}
        </div>
      </Section>

      {!loadingDocs && documents.length === 0 && (
        <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, color: "#f87171", marginBottom: 16 }}>
          No documents found. Please upload the study article first via the main dashboard.
        </div>
      )}

      <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
        <button className="btn btn--ghost" onClick={onBack} style={{ flex: 1 }}>← Back</button>
        <button className="btn btn--primary" onClick={onNext}
          disabled={!valid || documents.length === 0}
          style={{ flex: 2, opacity: valid && documents.length > 0 ? 1 : 0.4 }}>
          Continue →
        </button>
      </div>
    </Card>
  );
}

function StepPreSurvey({
  tlx, onChange, onNext, onBack,
}: {
  tlx: NasaTlx; onChange: (t: NasaTlx) => void; onNext: () => void; onBack: () => void;
}) {
  return (
    <Card>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Pre-Task Questionnaire</h2>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 4 }}>
        Rate your <em>current</em> state <strong>before</strong> the reading task. Move each slider to reflect how you feel right now.
      </p>
      <p style={{ color: "var(--text-muted)", fontSize: 12, marginBottom: 24 }}>
        Scale: 0 = very low / not at all &nbsp;·&nbsp; 100 = extremely high
      </p>

      <Section title="NASA Task Load Index (Baseline)">
        {([
          ["mental",      "Mental Demand",     "How mentally demanding did you find your current activity?"],
          ["physical",    "Physical Demand",   "How physically demanding was your current activity?"],
          ["temporal",    "Temporal Demand",   "How hurried or rushed do you feel?"],
          ["effort",      "Effort",            "How hard did you have to work to accomplish your current level of performance?"],
          ["frustration", "Frustration",       "How insecure, discouraged, irritated, stressed, or annoyed do you currently feel?"],
          ["performance", "Performance",       "How successful were you in accomplishing what you were asked to do so far today?"],
        ] as [keyof NasaTlx, string, string][]).map(([key, label, hint]) => (
          <SliderRow key={key} label={label} hint={hint} value={tlx[key]}
            onChange={(v) => onChange({ ...tlx, [key]: v })}
          />
        ))}
      </Section>

      <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "12px 16px", marginBottom: 24, fontSize: 13 }}>
        <span style={{ color: "var(--text-muted)" }}>Baseline workload score: </span>
        <strong style={{ color: "var(--accent)", fontSize: 16 }}>{computeNasaWl(tlx)}</strong>
        <span style={{ color: "var(--text-muted)" }}> / 100</span>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <button className="btn btn--ghost" onClick={onBack} style={{ flex: 1 }}>← Back</button>
        <button className="btn btn--primary" onClick={onNext} style={{ flex: 2 }}>Continue →</button>
      </div>
    </Card>
  );
}

function StepInstructions({
  info, documents, selectedDocIdx, onSelectDoc, onNext, onBack,
}: {
  info: ParticipantInfo;
  documents: Document[];
  selectedDocIdx: number;
  onSelectDoc: (idx: number) => void;
  onNext: () => void;
  onBack: () => void;
}) {
  return (
    <Card>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Reading Task Instructions</h2>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 24 }}>
        Please read the following carefully before starting.
      </p>

      <Section title="Instructions for the participant">
        <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", padding: "18px 20px" }}>
          <ul style={{ paddingLeft: 18, fontSize: 14, lineHeight: 2, margin: 0, color: "var(--text-muted)" }}>
            <li>Read the assigned article <strong style={{ color: "var(--text)" }}>at your normal reading pace</strong> — do not rush.</li>
            <li>The system will monitor your reading in the background. <strong style={{ color: "var(--text)" }}>Do not close the application</strong> during the task.</li>
            {info.group === "adaptive" && (
              <li>You may receive occasional prompts or suggestions from the system. <strong style={{ color: "var(--text)" }}>Engage with them naturally</strong> — you are not required to act on every one.</li>
            )}
            <li>If you need to stop, use the pause button in the top bar. To end the session early, use the stop button.</li>
            <li>Read for <strong style={{ color: "var(--text)" }}>approximately 15 minutes</strong>. The session will not end automatically — please complete it when you feel you have read enough.</li>
            <li>When you are done, click <strong style={{ color: "var(--text)" }}>End Session</strong> and you will be returned here to complete the final questionnaire.</li>
          </ul>
        </div>
      </Section>

      {documents.length > 1 && (
        <Section title="Select article">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {documents.map((d, i) => (
              <button key={d.id} type="button" onClick={() => onSelectDoc(i)}
                style={{
                  padding: "12px 14px", borderRadius: "var(--radius-sm)", textAlign: "left", cursor: "pointer",
                  background: selectedDocIdx === i ? "var(--accent-glow)" : "var(--bg-elevated)",
                  border: selectedDocIdx === i ? "2px solid var(--accent)" : "1px solid var(--border)",
                  color: selectedDocIdx === i ? "var(--accent)" : "var(--text)",
                  fontWeight: selectedDocIdx === i ? 600 : 400, fontSize: 13,
                }}>
                {d.title}
              </button>
            ))}
          </div>
        </Section>
      )}

      {documents.length === 1 && (
        <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", padding: "12px 16px", marginBottom: 24, fontSize: 13 }}>
          <span style={{ color: "var(--text-muted)" }}>Article: </span>
          <strong>{documents[0].title}</strong>
        </div>
      )}

      <div style={{ display: "flex", gap: 12 }}>
        <button className="btn btn--ghost" onClick={onBack} style={{ flex: 1 }}>← Back</button>
        <button className="btn btn--primary" onClick={onNext} disabled={documents.length === 0} style={{ flex: 2 }}>
          Start Reading Session →
        </button>
      </div>
    </Card>
  );
}

function StepSession({ sessionId }: { sessionId: number | null }) {
  return (
    <Card style={{ textAlign: "center" }}>
      <div style={{ fontSize: 48, marginBottom: 16 }}>📖</div>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 10 }}>Reading Session Active</h2>
      {sessionId ? (
        <>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 24 }}>
            Your session is running in the reader. When you are finished, click <strong>End Session</strong> inside the reader — you will be returned here automatically.
          </p>
          <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, color: "var(--text-muted)", marginBottom: 24 }}>
            Session ID: <strong style={{ color: "var(--text)" }}>#{sessionId}</strong>
          </div>
          <a href={`/sessions/${sessionId}/reader?study=1`} className="btn btn--primary" style={{ fontSize: 15, textDecoration: "none", display: "inline-block" }}>
            Open Reader →
          </a>
        </>
      ) : (
        <p style={{ color: "var(--text-muted)" }}>Starting session…</p>
      )}
    </Card>
  );
}

function StepPostSurvey({
  tlx, preTlx, onTlxChange,
  susItems, onSusChange,
  openFeedback, onOpenFeedback,
  onNext, onBack,
}: {
  tlx: NasaTlx; preTlx: NasaTlx; onTlxChange: (t: NasaTlx) => void;
  susItems: SusItem[]; onSusChange: (idx: number, v: number) => void;
  openFeedback: { helpful: string; distract: string; improve: string };
  onOpenFeedback: (k: keyof typeof openFeedback, v: string) => void;
  onNext: () => void; onBack: () => void;
}) {
  const susScore = computeSus(susItems);
  const susComplete = susItems.every((i) => i.value !== null);

  return (
    <Card>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Post-Task Questionnaire</h2>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 24 }}>
        Please answer the following questions about your experience <em>during</em> the reading task just completed.
      </p>

      <Section title="NASA Task Load Index (Post-Task)">
        {([
          ["mental",      "Mental Demand",     "How mentally demanding was the reading task?"],
          ["physical",    "Physical Demand",   "How physically demanding was the task?"],
          ["temporal",    "Temporal Demand",   "How hurried or rushed did you feel during the task?"],
          ["effort",      "Effort",            "How hard did you have to work to achieve your level of performance?"],
          ["frustration", "Frustration",       "How insecure, discouraged, irritated, or stressed did you feel?"],
          ["performance", "Performance",       "How successful do you feel you were in understanding what you read?"],
        ] as [keyof NasaTlx, string, string][]).map(([key, label, hint]) => (
          <SliderRow key={key} label={label} hint={hint} value={tlx[key]}
            onChange={(v) => onTlxChange({ ...tlx, [key]: v })}
          />
        ))}
        <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13, marginTop: 4, marginBottom: 8 }}>
          <span style={{ color: "var(--text-muted)" }}>Pre-task workload: </span>
          <strong style={{ color: "var(--text-muted)" }}>{computeNasaWl(preTlx)}</strong>
          <span style={{ color: "var(--text-muted)" }}> &nbsp;→&nbsp; Post-task: </span>
          <strong style={{ color: "var(--accent)", fontSize: 16 }}>{computeNasaWl(tlx)}</strong>
          <span style={{ color: "var(--text-muted)" }}> / 100</span>
          {computeNasaWl(tlx) < computeNasaWl(preTlx) && (
            <span style={{ marginLeft: 12, color: "#4ade80", fontSize: 12 }}>↓ Workload reduced</span>
          )}
        </div>
      </Section>

      <Section title="System Usability Scale (SUS)">
        <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 14 }}>
          For each statement, indicate how much you agree or disagree with it regarding the Lock-in system.
        </p>
        {susItems.map((item, i) => (
          <LikertRow key={i} idx={i} question={item.q} value={item.value}
            onChange={(v) => onSusChange(i, v)}
          />
        ))}
        {susComplete && (
          <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "14px 18px", fontSize: 14, marginTop: 8 }}>
            <span style={{ color: "var(--text-muted)" }}>SUS Score: </span>
            <strong style={{ color: susScore >= 70 ? "#4ade80" : susScore >= 50 ? "#facc15" : "#f87171", fontSize: 22 }}>{susScore}</strong>
            <span style={{ color: "var(--text-muted)", fontSize: 12, marginLeft: 8 }}>
              / 100 &nbsp;·&nbsp; {susScore >= 85 ? "Excellent" : susScore >= 70 ? "Good" : susScore >= 50 ? "Acceptable" : "Poor"}
            </span>
          </div>
        )}
      </Section>

      <Section title="Open Feedback">
        {([
          ["helpful",  "What aspects of the system (if any) did you find helpful during reading?"],
          ["distract", "Did any interventions feel disruptive or poorly timed? If so, describe."],
          ["improve",  "What would you change or improve about the system?"],
        ] as [keyof typeof openFeedback, string][]).map(([key, q]) => (
          <div key={key} className="form-group" style={{ marginBottom: 16 }}>
            <label>{q}</label>
            <textarea rows={3} value={openFeedback[key]}
              onChange={(e) => onOpenFeedback(key, e.target.value)}
              style={{ resize: "vertical", minHeight: 72 }}
            />
          </div>
        ))}
      </Section>

      <div style={{ display: "flex", gap: 12 }}>
        <button className="btn btn--ghost" onClick={onBack} style={{ flex: 1 }}>← Back</button>
        <button className="btn btn--primary" onClick={onNext} disabled={!susComplete} style={{ flex: 2, opacity: susComplete ? 1 : 0.4 }}>
          Submit →
        </button>
      </div>
    </Card>
  );
}

function StepComplete({
  info, preTlx, postTlx, susItems, openFeedback, sessionId,
}: {
  info: ParticipantInfo;
  preTlx: NasaTlx; postTlx: NasaTlx;
  susItems: SusItem[];
  openFeedback: { helpful: string; distract: string; improve: string };
  sessionId: number | null;
}) {
  const susScore = computeSus(susItems);
  const preWl = computeNasaWl(preTlx);
  const postWl = computeNasaWl(postTlx);
  const delta = postWl - preWl;

  function exportData() {
    const data = {
      participantId: info.participantId,
      group: info.group,
      age: info.age,
      readingFrequency: info.readingFrequency,
      sessionId,
      timestamp: new Date().toISOString(),
      preTlx,
      postTlx,
      preWorkload: preWl,
      postWorkload: postWl,
      workloadDelta: delta,
      susScore,
      susItems: susItems.map((i, idx) => ({ q: idx + 1, response: i.value })),
      openFeedback,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lockin_study_${info.participantId}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <Card style={{ textAlign: "center" }}>
      <div style={{ fontSize: 52, marginBottom: 12 }}>🎉</div>
      <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: 8 }}>Study Complete</h2>
      <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 32 }}>
        Thank you, <strong style={{ color: "var(--text)" }}>{info.participantId}</strong>. Your responses have been recorded.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14, marginBottom: 28 }}>
        {[
          { label: "Pre-Task Workload", value: preWl, unit: "/ 100", color: "var(--text-muted)" },
          { label: "Post-Task Workload", value: postWl, unit: "/ 100", color: delta < 0 ? "#4ade80" : delta > 10 ? "#f87171" : "var(--text)" },
          { label: "SUS Score", value: susScore, unit: "/ 100", color: susScore >= 70 ? "#4ade80" : susScore >= 50 ? "#facc15" : "#f87171" },
        ].map(({ label, value, unit, color }) => (
          <div key={label} style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "16px 12px", border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 28, fontWeight: 800, color }}>{value}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{unit}</div>
            <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 4 }}>{label}</div>
          </div>
        ))}
      </div>

      {delta !== 0 && (
        <div style={{
          background: delta < 0 ? "rgba(74,222,128,0.08)" : "rgba(248,113,113,0.08)",
          border: `1px solid ${delta < 0 ? "rgba(74,222,128,0.25)" : "rgba(248,113,113,0.25)"}`,
          borderRadius: "var(--radius-sm)", padding: "12px 16px", fontSize: 13,
          color: delta < 0 ? "#4ade80" : "#f87171", marginBottom: 24,
        }}>
          {delta < 0
            ? `Workload decreased by ${Math.abs(delta)} points after the session.`
            : `Workload increased by ${delta} points after the session.`}
        </div>
      )}

      <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 24 }}>
        Please let the researcher know you have finished. They may ask a few follow-up questions.
      </p>

      <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
        <button className="btn btn--ghost" onClick={exportData} style={{ gap: 8 }}>
          ↓ Export Results (JSON)
        </button>
        <a href="/" className="btn btn--primary" style={{ textDecoration: "none" }}>
          Return to Dashboard
        </a>
      </div>
    </Card>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function UserStudyPage() {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [step, setStep] = useState<Step>(1);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [selectedDocIdx, setSelectedDocIdx] = useState(0);
  const [sessionId, setSessionId] = useState<number | null>(null);

  const [participantInfo, setParticipantInfo] = useState<ParticipantInfo>({
    participantId: "", group: "adaptive", age: "", readingFrequency: "",
  });

  const [preTlx, setPreTlx] = useState<NasaTlx>(INITIAL_NASA);
  const [postTlx, setPostTlx] = useState<NasaTlx>(INITIAL_NASA);

  const [susItems, setSusItems] = useState<SusItem[]>(
    SUS_QUESTIONS.map((q) => ({ q, value: null }))
  );

  const [openFeedback, setOpenFeedback] = useState({ helpful: "", distract: "", improve: "" });

  // Return from reader after session ends
  useEffect(() => {
    const returnedFrom = searchParams.get("study_return");
    const sid = searchParams.get("session_id");
    if (returnedFrom === "1" && sid) {
      setSessionId(Number(sid));
      setStep(7);
    }
  }, [searchParams]);

  useEffect(() => {
    if (!token) return;
    documentService.list(token).then((r) => {
      setDocuments(r.documents);
    }).catch(() => {}).finally(() => setLoadingDocs(false));
  }, [token]);

  const startSession = useCallback(async () => {
    if (!token || documents.length === 0) return;
    const doc = documents[selectedDocIdx];
    const mode: SessionMode = participantInfo.group === "adaptive" ? "adaptive" : "baseline";
    const name = `Study — ${participantInfo.participantId} — ${doc.title}`;
    try {
      const session = await sessionService.start(token, doc.id, name, mode);
      setSessionId(session.id);
      setStep(6);
      navigate(`/sessions/${session.id}/reader?study=1`);
    } catch {
      alert("Failed to start session. Please check your connection and try again.");
    }
  }, [token, documents, selectedDocIdx, participantInfo, navigate]);

  function handleSusChange(idx: number, v: number) {
    setSusItems((prev) => prev.map((item, i) => i === idx ? { ...item, value: v } : item));
  }

  function handleOpenFeedback(k: keyof typeof openFeedback, v: string) {
    setOpenFeedback((prev) => ({ ...prev, [k]: v }));
  }

  const next = () => setStep((s) => Math.min(s + 1, 8) as Step);
  const back = () => setStep((s) => Math.max(s - 1, 1) as Step);

  return (
    <div style={{
      minHeight: "100vh", background: "var(--bg-base)",
      display: "flex", flexDirection: "column", alignItems: "center",
      padding: "40px 20px 80px",
    }}>
      <div style={{ width: "100%", maxWidth: 760 }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 32 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 20 }}>🔒</span>
            <span style={{ fontWeight: 800, fontSize: 16, letterSpacing: "-0.02em" }}>Lock-in</span>
            <span style={{ color: "var(--text-faint)", fontSize: 13 }}>/ User Study</span>
          </div>
          <a href="/" style={{ fontSize: 12, color: "var(--text-muted)", textDecoration: "none" }}>
            ← Dashboard
          </a>
        </div>

        <StepIndicator current={step} />

        {step === 1 && <StepWelcome onNext={next} />}
        {step === 2 && <StepConsent onNext={next} onBack={back} />}
        {step === 3 && (
          <StepSetup
            info={participantInfo} onChange={setParticipantInfo}
            documents={documents} loadingDocs={loadingDocs}
            selectedDoc={documents[selectedDocIdx] ?? null} onSelectDoc={setSelectedDocIdx}
            onNext={next} onBack={back}
          />
        )}
        {step === 4 && <StepPreSurvey tlx={preTlx} onChange={setPreTlx} onNext={next} onBack={back} />}
        {step === 5 && (
          <StepInstructions
            info={participantInfo} documents={documents}
            selectedDocIdx={selectedDocIdx} onSelectDoc={setSelectedDocIdx}
            onNext={startSession} onBack={back}
          />
        )}
        {step === 6 && <StepSession sessionId={sessionId} />}
        {step === 7 && (
          <StepPostSurvey
            tlx={postTlx} preTlx={preTlx} onTlxChange={setPostTlx}
            susItems={susItems} onSusChange={handleSusChange}
            openFeedback={openFeedback} onOpenFeedback={handleOpenFeedback}
            onNext={next} onBack={back}
          />
        )}
        {step === 8 && (
          <StepComplete
            info={participantInfo} preTlx={preTlx} postTlx={postTlx}
            susItems={susItems} openFeedback={openFeedback} sessionId={sessionId}
          />
        )}
      </div>
    </div>
  );
}
