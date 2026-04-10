/**
 * StudyDataExport.ts
 * ──────────────────
 * Fetches all session data from the backend and exports it as structured JSON
 * files that can be loaded into Python/R for analysis and plot generation.
 *
 * Output files per participant (all downloaded to the browser):
 *
 *   lockin_P01_study.json          ← master file: all questionnaires + scores
 *   lockin_P01_baseline_telemetry.csv  ← raw telemetry (scroll, mouse, idle…)
 *   lockin_P01_adaptive_telemetry.csv  ← same for adaptive condition
 *
 * The master JSON is self-contained and can be read without the CSV files.
 * The CSVs are useful for time-series plots of attentional state and drift.
 *
 * All files are named with the participant ID so multiple participants' files
 * can be placed in the same experimentresults/ directory without collision.
 */

import { apiRequest } from "../services/apiClient";
import { computeRawTlx, computeSus, type NasaDimension } from "./StudyConfig";
import { NASA_TLX_DIMENSIONS } from "./StudyConfig";
import type { Answers } from "./QuestionRenderer";

// ── API response types ─────────────────────────────────────────────────────────

interface AttentionalStateRecord {
  session_id:     number;
  created_at:     string;
  packet_seq:     number;
  primary_state:  string;
  confidence:     number;
  distribution:   Record<string, number>;
  drift_level:    number;
  drift_ema:      number;
  ambiguous:      boolean;
  intervention_context: Record<string, unknown>;
  latency_ms:     number;
  parse_ok:       boolean;
}

interface AttentionalStateHistoryResponse {
  session_id:      number;
  total_records:   number;
  records:         AttentionalStateRecord[];
  state_counts:    Record<string, number>;
  sustained:       boolean;
  sustained_state: string | null;
}

interface SessionDetail {
  id:         number;
  name:       string;
  mode:       string;
  status:     string;
  started_at: string;
  ended_at:   string | null;
  document_id: number;
}

interface InterventionHistoryItem {
  id:         number;
  type:       string;
  intensity:  string;
  created_at: string;
  content:    Record<string, unknown> | null;
}

// ── Fetch helpers ──────────────────────────────────────────────────────────────

async function fetchSessionDetail(token: string, sessionId: number): Promise<SessionDetail | null> {
  try {
    return await apiRequest<SessionDetail>(`/sessions/${sessionId}`, { token });
  } catch {
    return null;
  }
}

async function fetchAttentionalHistory(
  token: string,
  sessionId: number,
): Promise<AttentionalStateRecord[]> {
  try {
    const r = await apiRequest<AttentionalStateHistoryResponse>(
      `/sessions/${sessionId}/attentional-state/history?limit=2000`,
      { token }
    );
    // API returns newest-first; reverse to get chronological order
    return [...r.records].reverse();
  } catch {
    return [];
  }
}

async function fetchInterventionHistory(
  token: string,
  sessionId: number,
): Promise<InterventionHistoryItem[]> {
  try {
    return await apiRequest<InterventionHistoryItem[]>(
      `/sessions/${sessionId}/interventions/history`,
      { token }
    );
  } catch {
    return [];
  }
}

async function fetchTelemetryCSV(token: string, sessionId: number): Promise<string> {
  try {
    const res = await fetch(`http://localhost:8000/sessions/${sessionId}/export.csv`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) return "";
    return await res.text();
  } catch {
    return "";
  }
}

// ── Master export builder ──────────────────────────────────────────────────────

export interface StudyExportInput {
  token: string;
  participantId:       string;
  age:                 string;
  readingFrequency:    string;
  educationLevel:      string;
  baselineSessionId:   number | null;
  adaptiveSessionId:   number | null;
  documentTitle:       string;

  demographicAnswers:  Answers;
  preSurveyAnswers:    Answers;

  baselineNasaAnswers: Record<string, number>;
  adaptiveNasaAnswers: Record<string, number>;

  susAnswers:          (number | null)[];

  postAnswers:         Answers;
}

export interface StudyExportResult {
  masterJson:        string;   // master JSON with all questionnaires + full attentional/intervention data
  baselineCsvData:   string;   // raw telemetry CSV (scroll, mouse, idle…)
  adaptiveCsvData:   string;   // raw telemetry CSV for adaptive condition
  timelineCsvData:   string;   // combined attentional-state + intervention timeline (both conditions)
  summary: {
    baselineNasaScore:  number;
    adaptiveNasaScore:  number;
    nasaDelta:          number;
    susScore:           number;
    susGrade:           string;
    totalInterventions: number;
  };
}

export async function buildStudyExport(input: StudyExportInput): Promise<StudyExportResult> {
  const {
    token,
    participantId,
    baselineSessionId,
    adaptiveSessionId,
  } = input;

  // ── Fetch all session data in parallel ────────────────────────────────────
  const [
    baselineDetail,
    adaptiveDetail,
    baselineHistory,
    adaptiveHistory,
    adaptiveInterventions,
    baselineCsvData,
    adaptiveCsvData,
  ] = await Promise.all([
    baselineSessionId ? fetchSessionDetail(token, baselineSessionId)         : Promise.resolve(null),
    adaptiveSessionId ? fetchSessionDetail(token, adaptiveSessionId)         : Promise.resolve(null),
    baselineSessionId ? fetchAttentionalHistory(token, baselineSessionId)    : Promise.resolve([]),
    adaptiveSessionId ? fetchAttentionalHistory(token, adaptiveSessionId)    : Promise.resolve([]),
    adaptiveSessionId ? fetchInterventionHistory(token, adaptiveSessionId)   : Promise.resolve([]),
    baselineSessionId ? fetchTelemetryCSV(token, baselineSessionId)          : Promise.resolve(""),
    adaptiveSessionId ? fetchTelemetryCSV(token, adaptiveSessionId)          : Promise.resolve(""),
  ]);

  // ── Compute scores ─────────────────────────────────────────────────────────
  const baselineNasaScore = computeRawTlx(input.baselineNasaAnswers);
  const adaptiveNasaScore = computeRawTlx(input.adaptiveNasaAnswers);
  const nasaDelta         = adaptiveNasaScore - baselineNasaScore;
  const susScore          = computeSus(input.susAnswers);
  const susGrade          = susScore >= 85 ? "Excellent" : susScore >= 70 ? "Good" : susScore >= 50 ? "Acceptable" : "Poor";

  // ── Attentional state distribution helpers ─────────────────────────────────
  function stateCounts(history: AttentionalStateRecord[]) {
    const counts: Record<string, number> = {};
    for (const r of history) {
      counts[r.primary_state] = (counts[r.primary_state] ?? 0) + 1;
    }
    return counts;
  }

  function statePercent(history: AttentionalStateRecord[]) {
    const n = history.length;
    if (n === 0) return {};
    const counts = stateCounts(history);
    return Object.fromEntries(Object.entries(counts).map(([k, v]) => [k, Math.round((v / n) * 100)]));
  }

  function avgDrift(history: AttentionalStateRecord[]) {
    const vals = history.map((r) => r.drift_ema).filter((v) => v !== null && !isNaN(v));
    if (vals.length === 0) return null;
    return Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 1000) / 1000;
  }

  // ── Build full attentional timeline (all probability columns) ─────────────
  function buildTimeline(history: AttentionalStateRecord[]) {
    return history.map((r) => ({
      timestamp:              r.created_at,
      packet_seq:             r.packet_seq,
      primary_state:          r.primary_state,
      confidence:             r.confidence,
      prob_focused:           r.distribution["focused"]           ?? 0,
      prob_drifting:          r.distribution["drifting"]          ?? 0,
      prob_hyperfocused:      r.distribution["hyperfocused"]      ?? 0,
      prob_cognitive_overload:r.distribution["cognitive_overload"] ?? 0,
      drift_level:            r.drift_level,
      drift_ema:              r.drift_ema,
      ambiguous:              r.ambiguous,
      latency_ms:             r.latency_ms,
    }));
  }

  const totalInterventions = adaptiveInterventions.length;

  // ── Combined timeline CSV ──────────────────────────────────────────────────
  // One row per ~10-second classification point for both conditions.
  // Adaptive rows also show any intervention that fired within that window.
  // This is the primary CSV for generating the attentional-state-over-time plot.
  function buildTimelineCsv(
    bHistory: AttentionalStateRecord[],
    aHistory: AttentionalStateRecord[],
    interventions: InterventionHistoryItem[],
    bDetail: SessionDetail | null,
    aDetail: SessionDetail | null,
  ): string {
    const header = [
      "condition", "timestamp", "elapsed_seconds",
      "primary_state", "confidence",
      "prob_focused", "prob_drifting", "prob_hyperfocused", "prob_cognitive_overload",
      "drift_level", "drift_ema",
      "intervention_fired", "intervention_type", "intervention_intensity",
    ].join(",");

    const rows: string[] = [header];

    const bStart = bDetail?.started_at ? new Date(bDetail.started_at).getTime() : 0;
    const aStart = aDetail?.started_at ? new Date(aDetail.started_at).getTime() : 0;

    for (const r of bHistory) {
      const elapsed = bStart > 0 ? Math.round((new Date(r.created_at).getTime() - bStart) / 1000) : "";
      rows.push([
        "baseline", r.created_at, elapsed,
        r.primary_state, r.confidence.toFixed(4),
        (r.distribution["focused"] ?? 0).toFixed(4),
        (r.distribution["drifting"] ?? 0).toFixed(4),
        (r.distribution["hyperfocused"] ?? 0).toFixed(4),
        (r.distribution["cognitive_overload"] ?? 0).toFixed(4),
        r.drift_level.toFixed(6), r.drift_ema.toFixed(6),
        "false", "", "",
      ].join(","));
    }

    // Build a lookup: for each adaptive classification row, find any intervention
    // that fired within ±15 seconds of that timestamp.
    const intByTime = new Map<number, InterventionHistoryItem>();
    for (const iv of interventions) {
      intByTime.set(new Date(iv.created_at).getTime(), iv);
    }

    for (const r of aHistory) {
      const elapsed = aStart > 0 ? Math.round((new Date(r.created_at).getTime() - aStart) / 1000) : "";
      const rowTs = new Date(r.created_at).getTime();

      // Find the nearest intervention within a 15-second window of this row
      let nearestIv: InterventionHistoryItem | null = null;
      let nearestDist = Infinity;
      for (const [ivTs, iv] of intByTime) {
        const dist = Math.abs(ivTs - rowTs);
        if (dist <= 15_000 && dist < nearestDist) {
          nearestDist = dist;
          nearestIv = iv;
        }
      }

      rows.push([
        "adaptive", r.created_at, elapsed,
        r.primary_state, r.confidence.toFixed(4),
        (r.distribution["focused"] ?? 0).toFixed(4),
        (r.distribution["drifting"] ?? 0).toFixed(4),
        (r.distribution["hyperfocused"] ?? 0).toFixed(4),
        (r.distribution["cognitive_overload"] ?? 0).toFixed(4),
        r.drift_level.toFixed(6), r.drift_ema.toFixed(6),
        nearestIv ? "true" : "false",
        nearestIv ? nearestIv.type : "",
        nearestIv ? nearestIv.intensity : "",
      ].join(","));
    }

    return rows.join("\n");
  }

  const timelineCsvData = buildTimelineCsv(
    baselineHistory, adaptiveHistory, adaptiveInterventions, baselineDetail, adaptiveDetail,
  );

  // ── Assemble NASA-TLX summary rows ────────────────────────────────────────
  const nasaSummary = NASA_TLX_DIMENSIONS.map((d: NasaDimension) => ({
    dimension: d.label,
    baseline:  input.baselineNasaAnswers[d.key] ?? 50,
    adaptive:  input.adaptiveNasaAnswers[d.key] ?? 50,
    delta:     (input.adaptiveNasaAnswers[d.key] ?? 50) - (input.baselineNasaAnswers[d.key] ?? 50),
  }));

  // ── Master JSON ────────────────────────────────────────────────────────────
  const masterJson = JSON.stringify({
    meta: {
      participant_id:   participantId,
      export_timestamp: new Date().toISOString(),
      study_version:    "1.1",
      document_title:   input.documentTitle,
    },

    demographics: {
      age:               input.age,
      reading_frequency: input.readingFrequency,
      education_level:   input.educationLevel,
      ...flattenAnswers(input.demographicAnswers),
    },

    pre_study_survey: flattenAnswers(input.preSurveyAnswers),

    baseline_session: {
      session_id:       baselineSessionId,
      session_name:     baselineDetail?.name ?? null,
      started_at:       baselineDetail?.started_at ?? null,
      ended_at:         baselineDetail?.ended_at ?? null,
      nasa_tlx: {
        raw_scores:      input.baselineNasaAnswers,
        dimensions:      nasaSummary.map((r) => ({ dimension: r.dimension, score: r.baseline })),
        composite_score: baselineNasaScore,
      },
      attentional_summary: {
        total_classifications: baselineHistory.length,
        state_counts:    stateCounts(baselineHistory),
        state_percent:   statePercent(baselineHistory),
        avg_drift_ema:   avgDrift(baselineHistory),
      },
      attentional_timeline: buildTimeline(baselineHistory),
    },

    adaptive_session: {
      session_id:       adaptiveSessionId,
      session_name:     adaptiveDetail?.name ?? null,
      started_at:       adaptiveDetail?.started_at ?? null,
      ended_at:         adaptiveDetail?.ended_at ?? null,
      nasa_tlx: {
        raw_scores:      input.adaptiveNasaAnswers,
        dimensions:      nasaSummary.map((r) => ({ dimension: r.dimension, score: r.adaptive })),
        composite_score: adaptiveNasaScore,
      },
      attentional_summary: {
        total_classifications: adaptiveHistory.length,
        state_counts:    stateCounts(adaptiveHistory),
        state_percent:   statePercent(adaptiveHistory),
        avg_drift_ema:   avgDrift(adaptiveHistory),
      },
      attentional_timeline: buildTimeline(adaptiveHistory),
      interventions: {
        total_fired:  totalInterventions,
        log: adaptiveInterventions.map((i) => ({
          id:          i.id,
          type:        i.type,
          intensity:   i.intensity,
          fired_at:    i.created_at,
          content:     i.content,
        })),
        type_counts: countBy(
          adaptiveInterventions.map((i) => ({ type: i.type } as Record<string, unknown>)),
          "type",
        ),
      },
    },

    comparative: {
      nasa_tlx_comparison: nasaSummary,
      nasa_composite: {
        baseline:       baselineNasaScore,
        adaptive:       adaptiveNasaScore,
        delta:          nasaDelta,
        interpretation: nasaDelta < -5
          ? "Adaptive condition produced meaningfully lower workload"
          : nasaDelta > 5
          ? "Adaptive condition produced higher workload"
          : "No meaningful difference in workload between conditions",
      },
      attentional_state_comparison: {
        baseline: statePercent(baselineHistory),
        adaptive: statePercent(adaptiveHistory),
      },
    },

    system_usability_scale: {
      item_responses:  input.susAnswers,
      composite_score: susScore,
      grade:           susGrade,
    },

    post_experiment_survey: flattenAnswers(input.postAnswers),

  }, null, 2);

  return {
    masterJson,
    baselineCsvData,
    adaptiveCsvData,
    timelineCsvData,
    summary: {
      baselineNasaScore,
      adaptiveNasaScore,
      nasaDelta,
      susScore,
      susGrade,
      totalInterventions,
    },
  };
}

// ── Download helpers ───────────────────────────────────────────────────────────

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function downloadStudyFiles(
  participantId: string,
  result: StudyExportResult,
) {
  const pid = participantId.replace(/[^a-zA-Z0-9_-]/g, "_");
  downloadFile(result.masterJson,       `lockin_${pid}_study.json`,               "application/json");
  if (result.baselineCsvData) {
    downloadFile(result.baselineCsvData, `lockin_${pid}_baseline_telemetry.csv`,  "text/csv");
  }
  if (result.adaptiveCsvData) {
    downloadFile(result.adaptiveCsvData, `lockin_${pid}_adaptive_telemetry.csv`,  "text/csv");
  }
  if (result.timelineCsvData) {
    downloadFile(result.timelineCsvData, `lockin_${pid}_timeline.csv`,            "text/csv");
  }
}

/**
 * POST the three export files to the backend so they are automatically saved
 * in experimentresults/ on the server, independent of the browser download location.
 */
export async function saveExportToServer(
  token: string,
  participantId: string,
  result: StudyExportResult,
): Promise<{ saved_files: string[]; directory: string } | null> {
  try {
    const pid = participantId.replace(/[^a-zA-Z0-9_-]/g, "_");
    const response = await fetch("http://localhost:8000/study/save-export", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
      body: JSON.stringify({
        participant_id: pid,
        master_json:    result.masterJson,
        baseline_csv:   result.baselineCsvData ?? null,
        adaptive_csv:   result.adaptiveCsvData ?? null,
        timeline_csv:   result.timelineCsvData ?? null,
      }),
    });
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

// ── Utility ────────────────────────────────────────────────────────────────────

function flattenAnswers(answers: Answers): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(answers).map(([k, v]) => [k, v])
  );
}

function countBy<T extends Record<string, unknown>>(arr: T[], key: keyof T): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const item of arr) {
    const v = String(item[key]);
    counts[v] = (counts[v] ?? 0) + 1;
  }
  return counts;
}
