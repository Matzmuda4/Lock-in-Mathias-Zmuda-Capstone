/**
 * StudyAggregator.ts
 * ──────────────────
 * Loads multiple participant JSON exports from experimentresults/ and produces
 * aggregated statistics ready for plotting or reporting.
 *
 * Usage in the browser (drag-and-drop multiple lockin_P*_study.json files):
 *
 *   const agg = await StudyAggregator.fromFiles(fileList);
 *   console.log(agg.summary());
 *   const csv = agg.toCsv();   // one row per participant
 *
 * The aggregated CSV can be imported directly into Python (pandas) or R for
 * statistical analysis and plotting.
 *
 * Key comparisons supported:
 *   - NASA-TLX composite score: baseline vs adaptive (paired t-test input)
 *   - NASA-TLX per-dimension breakdown
 *   - SUS score distribution
 *   - Attentional state % (focused, drifting, overload, hyperfocused)
 *   - Intervention type frequency
 *   - Post-experiment questionnaire response distributions
 */

// ── Interfaces matching the master JSON output ────────────────────────────────

interface NasaComposite {
  baseline: number;
  adaptive: number;
  delta:    number;
}

interface AttentionalStatePercent {
  focused?:           number;
  drifting?:          number;
  hyperfocused?:      number;
  cognitive_overload?: number;
  [state: string]:    number | undefined;
}

interface AttentionalComparison {
  baseline: AttentionalStatePercent;
  adaptive: AttentionalStatePercent;
}

interface InterventionCounts {
  total_fired:  number;
  type_counts:  Record<string, number>;
}

interface ParticipantRecord {
  participant_id: string;
  nasa_composite: NasaComposite;
  sus_score: number;
  sus_grade: string;
  attentional_comparison: AttentionalComparison;
  interventions: InterventionCounts;
  demographics: Record<string, unknown>;
  post_survey:  Record<string, unknown>;
  // Raw for custom analysis
  raw: Record<string, unknown>;
}

// ── Aggregator class ──────────────────────────────────────────────────────────

export class StudyAggregator {
  private participants: ParticipantRecord[] = [];

  private constructor(records: ParticipantRecord[]) {
    this.participants = records;
  }

  // ── Load from browser File objects (drag-and-drop) ──────────────────────────
  static async fromFiles(files: FileList | File[]): Promise<StudyAggregator> {
    const records: ParticipantRecord[] = [];
    for (const file of Array.from(files)) {
      try {
        const text  = await file.text();
        const json  = JSON.parse(text);
        const rec   = StudyAggregator.parseRecord(json);
        if (rec) records.push(rec);
      } catch {
        console.warn(`StudyAggregator: could not parse ${file.name}`);
      }
    }
    return new StudyAggregator(records);
  }

  private static parseRecord(json: Record<string, unknown>): ParticipantRecord | null {
    try {
      const meta     = json.meta as Record<string, unknown>;
      const comp     = json.comparative as Record<string, unknown>;
      const sus      = json.system_usability_scale as Record<string, unknown>;
      const adapt    = json.adaptive_session as Record<string, unknown>;

      const nasa = comp.nasa_composite as NasaComposite;
      const attComp = comp.attentional_state_comparison as AttentionalComparison;
      const interventions = (adapt?.interventions ?? { total_fired: 0, type_counts: {} }) as InterventionCounts;

      return {
        participant_id:          String(meta.participant_id),
        nasa_composite:          nasa,
        sus_score:               Number(sus.composite_score),
        sus_grade:               String(sus.grade),
        attentional_comparison:  attComp,
        interventions,
        demographics:            (json.demographics ?? {}) as Record<string, unknown>,
        post_survey:             (json.post_experiment_survey ?? {}) as Record<string, unknown>,
        raw:                     json,
      };
    } catch {
      return null;
    }
  }

  // ── Number of loaded participants ───────────────────────────────────────────
  get n(): number { return this.participants.length; }

  // ── Summary statistics ──────────────────────────────────────────────────────
  summary() {
    const n = this.n;
    if (n === 0) return { n: 0, message: "No participants loaded" };

    const baselineNasa = this.participants.map((p) => p.nasa_composite.baseline);
    const adaptiveNasa = this.participants.map((p) => p.nasa_composite.adaptive);
    const deltas       = this.participants.map((p) => p.nasa_composite.delta);
    const susScores    = this.participants.map((p) => p.sus_score);

    return {
      n,
      nasa: {
        baseline: { mean: mean(baselineNasa), sd: sd(baselineNasa), min: Math.min(...baselineNasa), max: Math.max(...baselineNasa) },
        adaptive: { mean: mean(adaptiveNasa), sd: sd(adaptiveNasa), min: Math.min(...adaptiveNasa), max: Math.max(...adaptiveNasa) },
        delta:    { mean: mean(deltas),       sd: sd(deltas),       interpretation: mean(deltas) < -5 ? "Adaptive produces lower workload" : mean(deltas) > 5 ? "Adaptive produces higher workload" : "No meaningful difference" },
      },
      sus: {
        mean:   mean(susScores),
        sd:     sd(susScores),
        grades: countValues(this.participants.map((p) => p.sus_grade)),
      },
      interventions: {
        total_across_all: this.participants.reduce((s, p) => s + p.interventions.total_fired, 0),
        avg_per_session:  mean(this.participants.map((p) => p.interventions.total_fired)),
        type_totals:      mergeCountMaps(this.participants.map((p) => p.interventions.type_counts)),
      },
    };
  }

  // ── Per-participant rows ────────────────────────────────────────────────────
  rows(): Record<string, unknown>[] {
    return this.participants.map((p) => ({
      participant_id:          p.participant_id,
      baseline_nasa:           p.nasa_composite.baseline,
      adaptive_nasa:           p.nasa_composite.adaptive,
      nasa_delta:              p.nasa_composite.delta,
      sus_score:               p.sus_score,
      sus_grade:               p.sus_grade,
      interventions_fired:     p.interventions.total_fired,
      baseline_focused_pct:    p.attentional_comparison.baseline?.focused ?? null,
      baseline_drifting_pct:   p.attentional_comparison.baseline?.drifting ?? null,
      baseline_overload_pct:   p.attentional_comparison.baseline?.cognitive_overload ?? null,
      adaptive_focused_pct:    p.attentional_comparison.adaptive?.focused ?? null,
      adaptive_drifting_pct:   p.attentional_comparison.adaptive?.drifting ?? null,
      adaptive_overload_pct:   p.attentional_comparison.adaptive?.cognitive_overload ?? null,
      // Post-survey items (flattened)
      ...Object.fromEntries(
        Object.entries(p.post_survey).map(([k, v]) => [`post_${k}`, v])
      ),
      // Demographics
      age_group:         p.demographics.age_group ?? null,
      education:         p.demographics.education ?? null,
      reading_frequency: p.demographics.reading_frequency ?? null,
    }));
  }

  // ── Export as CSV ───────────────────────────────────────────────────────────
  toCsv(): string {
    const rowObjs = this.rows();
    if (rowObjs.length === 0) return "";
    const headers = Object.keys(rowObjs[0]);
    const lines   = [
      headers.join(","),
      ...rowObjs.map((row) =>
        headers.map((h) => {
          const v = row[h];
          if (v === null || v === undefined) return "";
          const s = String(v);
          return s.includes(",") || s.includes('"') || s.includes("\n")
            ? `"${s.replace(/"/g, '""')}"` : s;
        }).join(",")
      ),
    ];
    return lines.join("\n");
  }

  // ── Post-survey frequency tables (for bar charts) ───────────────────────────
  // Returns { questionId: { optionValue: count } }
  postSurveyFrequencies(): Record<string, Record<string, number>> {
    const result: Record<string, Record<string, number>> = {};
    for (const p of this.participants) {
      for (const [k, v] of Object.entries(p.post_survey)) {
        if (!result[k]) result[k] = {};
        const key = Array.isArray(v) ? v.join("|") : String(v ?? "");
        result[k][key] = (result[k][key] ?? 0) + 1;
      }
    }
    return result;
  }

  // ── Download aggregated CSV ─────────────────────────────────────────────────
  downloadCsv(filename = "lockin_study_aggregated.csv") {
    const csv  = this.toCsv();
    const blob = new Blob([csv], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ── Download summary JSON ───────────────────────────────────────────────────
  downloadSummary(filename = "lockin_study_summary.json") {
    const json = JSON.stringify({ summary: this.summary(), rows: this.rows() }, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}

// ── Utility ────────────────────────────────────────────────────────────────────

function mean(arr: number[]): number {
  if (!arr.length) return 0;
  return Math.round((arr.reduce((a, b) => a + b, 0) / arr.length) * 10) / 10;
}

function sd(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m  = mean(arr);
  const v  = arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.round(Math.sqrt(v) * 10) / 10;
}

function countValues<T extends string>(arr: T[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const v of arr) { counts[v] = (counts[v] ?? 0) + 1; }
  return counts;
}

function mergeCountMaps(maps: Record<string, number>[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of maps) {
    for (const [k, v] of Object.entries(m)) {
      out[k] = (out[k] ?? 0) + v;
    }
  }
  return out;
}
