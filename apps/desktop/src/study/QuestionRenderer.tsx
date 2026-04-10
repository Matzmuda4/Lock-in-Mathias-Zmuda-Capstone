/**
 * QuestionRenderer.tsx
 * ────────────────────
 * Renders any question defined in StudyConfig.ts.
 * All styling uses the existing design-system variables.
 *
 * Props:
 *   questions  — array of Question objects from StudyConfig
 *   answers    — Record<questionId, value> controlled by parent
 *   onChange   — called when any answer changes
 *   readonly   — show filled-in answers without allowing edits (for review)
 */

import type { Question, NasaDimension } from "./StudyConfig";
import { NASA_TLX_DIMENSIONS } from "./StudyConfig";

export type Answers = Record<string, string | number | string[] | Record<string, number> | null>;

interface Props {
  questions: Question[];
  answers:   Answers;
  onChange:  (id: string, value: Answers[string]) => void;
  readonly?: boolean;
}

// ── Internal sub-components ───────────────────────────────────────────────────

function SectionLabel({ text }: { text: string }) {
  return (
    <h3 style={{
      fontSize: 12, fontWeight: 700, color: "var(--text-muted)",
      textTransform: "uppercase", letterSpacing: "0.07em",
      margin: "28px 0 14px", paddingBottom: 8,
      borderBottom: "1px solid var(--border)",
    }}>
      {text}
    </h3>
  );
}

function InfoBlock({ text }: { text: string }) {
  return (
    <div style={{
      background: "var(--bg-elevated)", border: "1px solid var(--border)",
      borderRadius: "var(--radius-sm)", padding: "14px 18px",
      fontSize: 13, color: "var(--text-muted)", lineHeight: 1.7,
      marginBottom: 8,
    }}>
      {text}
    </div>
  );
}

function LikertItem({
  text, scaleLeft, scaleRight, value, onChange, readonly,
}: {
  id?: string; text: string; scaleLeft: string; scaleRight: string;
  value: number | null; onChange: (v: number) => void; readonly?: boolean;
}) {
  return (
    <div style={{ marginBottom: 20 }}>
      <p style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 12 }}>{text}</p>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 11, color: "var(--text-muted)", minWidth: 110, textAlign: "right" }}>{scaleLeft}</span>
        <div style={{ display: "flex", gap: 6, flex: 1, justifyContent: "center" }}>
          {[1, 2, 3, 4, 5].map((v) => (
            <button key={v} type="button" disabled={readonly}
              onClick={() => !readonly && onChange(v)}
              style={{
                width: 40, height: 40, borderRadius: "50%",
                border: value === v ? "2px solid var(--accent)" : "1px solid var(--border)",
                background: value === v ? "var(--accent)" : "var(--bg-elevated)",
                color: value === v ? "#fff" : "var(--text-muted)",
                fontWeight: value === v ? 700 : 400, fontSize: 14, cursor: readonly ? "default" : "pointer",
                flexShrink: 0,
              }}>
              {v}
            </button>
          ))}
        </div>
        <span style={{ fontSize: 11, color: "var(--text-muted)", minWidth: 110 }}>{scaleRight}</span>
      </div>
    </div>
  );
}

function McSingleItem({
  id, text, options, value, onChange, readonly,
}: {
  id?: string; text: string; options: { value: string; label: string }[];
  value: string | null; onChange: (v: string) => void; readonly?: boolean;
}) {
  return (
    <div style={{ marginBottom: 20 }}>
      <p style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 12 }}>{text}</p>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {options.map((opt) => (
          <label key={opt.value} style={{ display: "flex", gap: 12, alignItems: "center", cursor: readonly ? "default" : "pointer" }}>
            <input type="radio" name={id} value={opt.value} checked={value === opt.value}
              disabled={readonly}
              onChange={() => !readonly && onChange(opt.value)}
              style={{ accentColor: "var(--accent)", width: 16, height: 16, flexShrink: 0 }}
            />
            <span style={{ fontSize: 13, lineHeight: 1.5 }}>{opt.label}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

function McMultiItem({
  text, options, value, onChange, readonly,
}: {
  id?: string; text: string; options: { value: string; label: string }[];
  value: string[]; onChange: (v: string[]) => void; readonly?: boolean;
}) {
  function toggle(v: string) {
    if (readonly) return;
    onChange(value.includes(v) ? value.filter((x) => x !== v) : [...value, v]);
  }
  return (
    <div style={{ marginBottom: 20 }}>
      <p style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 4 }}>{text}</p>
      <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12 }}>Select all that apply.</p>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {options.map((opt) => (
          <label key={opt.value} style={{ display: "flex", gap: 12, alignItems: "center", cursor: readonly ? "default" : "pointer" }}>
            <input type="checkbox" value={opt.value} checked={value.includes(opt.value)}
              disabled={readonly}
              onChange={() => toggle(opt.value)}
              style={{ accentColor: "var(--accent)", width: 16, height: 16, flexShrink: 0 }}
            />
            <span style={{ fontSize: 13, lineHeight: 1.5 }}>{opt.label}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

function NasaTlxItem({
  condition, value, onChange, readonly,
}: {
  condition: string;
  value: Record<string, number>;
  onChange: (v: Record<string, number>) => void;
  readonly?: boolean;
}) {
  function setDim(key: string, v: number) {
    if (!readonly) onChange({ ...value, [key]: v });
  }
  const rawScore = Math.round(
    NASA_TLX_DIMENSIONS.reduce((s, d) => s + (value[d.key] ?? 50), 0) / NASA_TLX_DIMENSIONS.length
  );

  return (
    <div>
      <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "10px 14px", fontSize: 12, color: "var(--text-muted)", marginBottom: 20, border: "1px solid var(--border)" }}>
        Rate your experience during the <strong style={{ color: "var(--text)" }}>{condition}</strong> reading session. Move each slider to reflect how you felt.
        <br />Scale: 0 = Very Low / Perfect &nbsp;·&nbsp; 100 = Very High / Failure
      </div>
      {NASA_TLX_DIMENSIONS.map((dim: NasaDimension) => (
        <div key={dim.key} style={{ display: "grid", gridTemplateColumns: "160px 1fr 52px", alignItems: "center", gap: 16, marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600 }}>{dim.label}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2, lineHeight: 1.4 }}>{dim.hint}</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 10, color: "var(--text-faint)", minWidth: 40, textAlign: "right" }}>{dim.lowLabel}</span>
            <input type="range" min={0} max={100} step={5}
              value={value[dim.key] ?? 50}
              disabled={readonly}
              onChange={(e) => setDim(dim.key, Number(e.target.value))}
              style={{ flex: 1, accentColor: "var(--accent)" }}
            />
            <span style={{ fontSize: 10, color: "var(--text-faint)", minWidth: 40 }}>{dim.highLabel}</span>
          </div>
          <span style={{ fontSize: 15, fontWeight: 700, color: "var(--accent)", textAlign: "right" }}>
            {value[dim.key] ?? 50}
          </span>
        </div>
      ))}
      <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "12px 16px", marginTop: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Raw TLX Workload Score</span>
        <span style={{ fontSize: 22, fontWeight: 800, color: rawScore > 65 ? "#f87171" : rawScore > 40 ? "#facc15" : "#4ade80" }}>
          {rawScore} <span style={{ fontSize: 13, fontWeight: 400, color: "var(--text-muted)" }}>/ 100</span>
        </span>
      </div>
    </div>
  );
}

function OpenItem({
  text, rows, value, onChange, readonly,
}: {
  text: string; rows: number;
  value: string; onChange: (v: string) => void; readonly?: boolean;
}) {
  return (
    <div style={{ marginBottom: 20 }}>
      <p style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 10 }}>{text}</p>
      <textarea rows={rows} value={value} disabled={readonly}
        onChange={(e) => !readonly && onChange(e.target.value)}
        style={{
          width: "100%", resize: "vertical", minHeight: rows * 24,
          background: "var(--bg-elevated)", border: "1px solid var(--border)",
          borderRadius: "var(--radius-sm)", color: "var(--text)", padding: "10px 12px",
          fontSize: 13, boxSizing: "border-box",
        }}
        placeholder="Optional…"
      />
    </div>
  );
}

// ── Default NASA-TLX values ───────────────────────────────────────────────────
export function defaultNasaValues(): Record<string, number> {
  return Object.fromEntries(NASA_TLX_DIMENSIONS.map((d) => [d.key, 50]));
}

// ── Main renderer ─────────────────────────────────────────────────────────────

export function QuestionRenderer({ questions, answers, onChange, readonly }: Props) {
  return (
    <div>
      {questions.map((q) => {
        const label = q.label;
        return (
          <div key={q.id}>
            {label && <SectionLabel text={label} />}

            {q.type === "info" && <InfoBlock text={q.text} />}

            {q.type === "likert" && (
              <LikertItem
                id={q.id} text={q.text}
                scaleLeft={q.scaleLeft} scaleRight={q.scaleRight}
                value={(answers[q.id] as number) ?? null}
                onChange={(v) => onChange(q.id, v)}
                readonly={readonly}
              />
            )}

            {q.type === "mc_single" && (
              <McSingleItem
                id={q.id} text={q.text} options={q.options}
                value={(answers[q.id] as string) ?? null}
                onChange={(v) => onChange(q.id, v)}
                readonly={readonly}
              />
            )}

            {q.type === "mc_multi" && (
              <McMultiItem
                id={q.id} text={q.text} options={q.options}
                value={(answers[q.id] as string[]) ?? []}
                onChange={(v) => onChange(q.id, v)}
                readonly={readonly}
              />
            )}

            {q.type === "nasa_tlx" && (
              <NasaTlxItem
                condition={q.condition}
                value={(answers[q.id] as Record<string, number>) ?? defaultNasaValues()}
                onChange={(v) => onChange(q.id, v)}
                readonly={readonly}
              />
            )}

            {q.type === "open" && (
              <OpenItem
                text={q.text} rows={q.rows ?? 3}
                value={(answers[q.id] as string) ?? ""}
                onChange={(v) => onChange(q.id, v)}
                readonly={readonly}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Validation helper ─────────────────────────────────────────────────────────
// Returns a list of IDs that are required but unanswered.

export function getMissingRequired(questions: Question[], answers: Answers): string[] {
  return questions
    .filter((q) => {
      if (!q.required || q.type === "info") return false;
      const v = answers[q.id];
      if (q.type === "nasa_tlx") return false; // always has defaults
      if (q.type === "mc_multi")  return false; // optional
      if (q.type === "open")      return false; // optional
      return v === undefined || v === null || v === "";
    })
    .map((q) => q.id);
}
