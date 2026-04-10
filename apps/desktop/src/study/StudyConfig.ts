/**
 * StudyConfig.ts
 * ──────────────
 * Single source of truth for every questionnaire item in the user study.
 * Edit this file to change questions, scales, labels, or study parameters
 * without touching any React component.
 *
 * Question types:
 *   likert    — 5-point agree/disagree scale  (produces ordinal data for plots)
 *   mc_single — single-choice radio           (produces categorical data)
 *   mc_multi  — multi-select checkboxes       (produces frequency data)
 *   nasa_tlx  — six NASA-TLX sliders 0-100   (produces workload score)
 *   open      — free-text textarea            (qualitative)
 *   info      — static text block, no input   (instructions / section header)
 */

// ── Study parameters ──────────────────────────────────────────────────────────
// Edit these without touching any component.

export const STUDY_PARAMS = {
  studyTitle:              "Lock-in: AI-Driven Attentional Monitoring in Digital Reading",
  researcherName:          "Mathias Zmuda",
  degreeProgram:           "Computing Science BSc",
  targetReadingMinutes:    15,      // per condition
  breakMinutes:            5,
  totalDurationMinutes:    40,
  conditionOrder:          ["baseline", "adaptive"] as const,   // all participants do both
} as const;

// ── Question types ────────────────────────────────────────────────────────────

export type QuestionType =
  | "likert"
  | "mc_single"
  | "mc_multi"
  | "nasa_tlx"
  | "open"
  | "info";

export interface BaseQuestion {
  id:       string;
  type:     QuestionType;
  label?:   string;   // section heading shown above the question
  required: boolean;
}

export interface LikertQuestion extends BaseQuestion {
  type:     "likert";
  text:     string;
  scaleLeft:  string;   // label for 1
  scaleRight: string;   // label for 5
}

export interface McSingleQuestion extends BaseQuestion {
  type:    "mc_single";
  text:    string;
  options: { value: string; label: string }[];
}

export interface McMultiQuestion extends BaseQuestion {
  type:    "mc_multi";
  text:    string;
  options: { value: string; label: string }[];
}

export interface NasaTlxQuestion extends BaseQuestion {
  type:      "nasa_tlx";
  condition: string;   // "baseline" | "adaptive" | custom label
}

export interface OpenQuestion extends BaseQuestion {
  type:  "open";
  text:  string;
  rows?: number;
}

export interface InfoBlock extends BaseQuestion {
  type:  "info";
  text:  string;
}

export type Question =
  | LikertQuestion
  | McSingleQuestion
  | McMultiQuestion
  | NasaTlxQuestion
  | OpenQuestion
  | InfoBlock;

// ── Helpers ───────────────────────────────────────────────────────────────────

const likert = (
  id: string,
  text: string,
  opts?: { label?: string; scaleLeft?: string; scaleRight?: string }
): LikertQuestion => ({
  id,
  type: "likert",
  text,
  label:      opts?.label,
  scaleLeft:  opts?.scaleLeft  ?? "Strongly disagree",
  scaleRight: opts?.scaleRight ?? "Strongly agree",
  required:   true,
});

const mc = (
  id: string,
  text: string,
  options: { value: string; label: string }[],
  opts?: { label?: string }
): McSingleQuestion => ({
  id, type: "mc_single", text, options,
  label: opts?.label, required: true,
});

const mcMulti = (
  id: string,
  text: string,
  options: { value: string; label: string }[],
  opts?: { label?: string }
): McMultiQuestion => ({
  id, type: "mc_multi", text, options,
  label: opts?.label, required: false,
});

const open = (id: string, text: string, rows?: number): OpenQuestion => ({
  id, type: "open", text, rows: rows ?? 3, label: undefined, required: false,
});

const info = (id: string, text: string): InfoBlock => ({
  id, type: "info", text, label: undefined, required: false,
});

// ── SECTION 1: Demographics & background ─────────────────────────────────────
// Administered once at the start. Used for descriptive statistics.

export const DEMOGRAPHICS_QUESTIONS: Question[] = [
  mc("age_group", "What is your age range?", [
    { value: "under_18",  label: "Under 18" },
    { value: "18_24",     label: "18–24" },
    { value: "25_34",     label: "25–34" },
    { value: "35_44",     label: "35–44" },
    { value: "45_plus",   label: "45 or older" },
  ], { label: "About you" }),

  mc("education", "What is your highest level of education completed or currently pursuing?", [
    { value: "secondary",    label: "Secondary / High School" },
    { value: "undergraduate","label": "Undergraduate degree" },
    { value: "postgraduate", label: "Postgraduate degree" },
    { value: "other",        label: "Other" },
  ]),

  mc("reading_frequency", "How often do you read academic or technical texts (articles, textbooks)?", [
    { value: "rarely",    label: "Rarely (less than once a week)" },
    { value: "sometimes", label: "Sometimes (1–3 times a week)" },
    { value: "often",     label: "Often (most days)" },
    { value: "daily",     label: "Daily (core part of my studies or work)" },
  ]),

  mc("digital_reading_habit", "How do you primarily read long-form academic content?", [
    { value: "print",         label: "Printed paper" },
    { value: "screen_laptop", label: "On screen (laptop or desktop)" },
    { value: "screen_tablet", label: "On screen (tablet)" },
    { value: "mixed",         label: "A mix of both" },
  ]),

  mc("attention_difficulty", "How often do you experience difficulty staying focused while reading?", [
    { value: "rarely",     label: "Rarely" },
    { value: "sometimes",  label: "Sometimes" },
    { value: "often",      label: "Often" },
    { value: "very_often", label: "Very often" },
  ]),

  mc("adhd_diagnosis", "Have you been diagnosed with any type or severity of ADHD (Attention Deficit Hyperactivity Disorder)?", [
    { value: "yes_medicated",   label: "Yes — diagnosed and currently medicated" },
    { value: "yes_unmedicated", label: "Yes — diagnosed but not currently medicated" },
    { value: "suspected",       label: "Not formally diagnosed, but suspected / under assessment" },
    { value: "no",              label: "No" },
    { value: "prefer_not",      label: "Prefer not to say" },
  ], { label: "Attention & ADHD" }),
];

// ── SECTION 2: Pre-study attitudes ───────────────────────────────────────────
// Measures prior attitudes toward AI tools and reading technology.
// Useful as a covariate in analysis to control for pre-existing biases.

export const PRE_STUDY_QUESTIONS: Question[] = [
  info("pre_info", "The following questions ask about your general attitudes toward AI-assisted learning tools. Please answer based on your experiences before this study."),

  likert("prior_ai_positive", "I generally have a positive attitude toward using AI tools for learning.", { label: "Prior attitudes" }),
  likert("prior_tools_helpful", "Digital reading tools (highlights, annotations, etc.) generally help me read more effectively."),
  likert("prior_monitoring_comfort", "I am comfortable with software that tracks my behaviour to provide personalised feedback."),

  mc("prior_ai_tool_use", "Have you previously used any AI-powered learning or productivity tools (e.g. ChatGPT for studying, Grammarly, adaptive learning platforms)?", [
    { value: "never",      label: "Never" },
    { value: "rarely",     label: "Rarely (1–2 times)" },
    { value: "sometimes",  label: "Sometimes" },
    { value: "regularly",  label: "Regularly" },
  ], { label: "Prior experience" }),
];

// ── SECTION 3: NASA Task Load Index (administered after each reading session) ─
// Raw TLX — the unweighted version (Sauro & Dumas, 2009).
// Each dimension is rated 0–100 in steps of 5.
// Final score = mean of all 6 dimensions.
//
// EDITING: Change the labels/hints in the NASA_TLX_DIMENSIONS array below.

export interface NasaDimension {
  key:   string;
  label: string;
  hint:  string;
  lowLabel:  string;  // label for 0
  highLabel: string;  // label for 100
}

export const NASA_TLX_DIMENSIONS: NasaDimension[] = [
  {
    key:       "mental",
    label:     "Mental Demand",
    hint:      "How much mental and perceptual activity was required? Was the task easy or demanding, simple or complex?",
    lowLabel:  "Very Low",
    highLabel: "Very High",
  },
  {
    key:       "physical",
    label:     "Physical Demand",
    hint:      "How much physical activity was required? Were you mostly stationary or actively navigating?",
    lowLabel:  "Very Low",
    highLabel: "Very High",
  },
  {
    key:       "temporal",
    label:     "Temporal Demand",
    hint:      "How hurried or rushed was the pace of the task? Was there time pressure?",
    lowLabel:  "Very Low",
    highLabel: "Very High",
  },
  {
    key:       "performance",
    label:     "Performance",
    hint:      "How successful were you in accomplishing what you were asked to do? How satisfied are you with your performance?",
    lowLabel:  "Perfect",
    highLabel: "Failure",
  },
  {
    key:       "effort",
    label:     "Effort",
    hint:      "How hard did you have to work (mentally and physically) to achieve your level of performance?",
    lowLabel:  "Very Low",
    highLabel: "Very High",
  },
  {
    key:       "frustration",
    label:     "Frustration Level",
    hint:      "How insecure, discouraged, irritated, stressed, and annoyed were you during the task?",
    lowLabel:  "Very Low",
    highLabel: "Very High",
  },
];

// Compute Raw TLX score (0–100)
export function computeRawTlx(scores: Record<string, number>): number {
  const vals = NASA_TLX_DIMENSIONS.map((d) => scores[d.key] ?? 50);
  return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
}

// ── SECTION 4: System Usability Scale (SUS) ───────────────────────────────────
// Brooke (1996). 10-item alternating-valence Likert scale. Produces 0–100 score.
// Score interpretation: ≥85 Excellent, ≥70 Good, ≥50 Acceptable, <50 Poor.

export const SUS_QUESTIONS_LIST: string[] = [
  "I think that I would like to use this system frequently.",
  "I found the system unnecessarily complex.",
  "I thought the system was easy to use.",
  "I think that I would need the support of a technical person to use this system.",
  "I found the various functions in this system were well integrated.",
  "I thought there was too much inconsistency in this system.",
  "I would imagine that most people would learn to use this system very quickly.",
  "I found the system very cumbersome to use.",
  "I felt very confident using the system.",
  "I needed to learn a lot of things before I could get going with this system.",
];

// Standard SUS formula: odd indices (1,3,5,7,9): score-1; even indices (2,4,6,8,10): 5-score; sum × 2.5
export function computeSus(responses: (number | null)[]): number {
  if (responses.some((r) => r === null)) return 0;
  let sum = 0;
  responses.forEach((v, i) => {
    const score = v ?? 3;
    sum += i % 2 === 0 ? score - 1 : 5 - score;
  });
  return Math.round(sum * 2.5);
}

export function susGrade(score: number): string {
  if (score >= 85) return "Excellent";
  if (score >= 70) return "Good";
  if (score >= 50) return "Acceptable";
  return "Poor";
}

// ── SECTION 5: Post-experiment questionnaire ──────────────────────────────────
// Measures attitudes toward the system after experiencing both conditions.
// All Likert items produce ordinal data suitable for violin/bar plots.
// All mc_single items produce categorical/frequency data.
//
// EDITING: Add, remove, or reorder questions here freely.
// IDs become keys in the exported JSON — change them carefully if you have
// data already collected.

export const POST_EXPERIMENT_QUESTIONS: Question[] = [
  // ── Perceived usefulness (TAM, Davis 1989) ───────────────────────────────
  info("pu_header", "The following questions ask about your experience using the Lock-in reading assistant during the study."),

  likert("pu_focus",       "The AI-assisted reading session helped me maintain focus on the text.",   { label: "Perceived Usefulness" }),
  likert("pu_engagement",  "The system made me more engaged with what I was reading."),
  likert("pu_understand",  "I felt I understood the material better with the adaptive system active."),
  likert("pu_future",      "I would find this system useful in my regular academic reading."),

  // ── Intervention experience ───────────────────────────────────────────────
  likert("int_timely",     "The reading prompts appeared at appropriate moments during my reading.",  { label: "Intervention Experience" }),
  likert("int_relevant",   "The prompts and cues were relevant to what I was reading."),
  likert("int_disrupt",    "The interventions disrupted my reading flow.",
    { scaleLeft: "Strongly disagree (not disruptive)", scaleRight: "Strongly agree (very disruptive)" }),
  likert("int_helpful",    "The interventions helped me get back on track when I lost focus."),
  likert("int_subtle",     "The prompts felt subtle enough that I did not find them intrusive."),

  mc("int_most_helpful", "Which type of support, if any, did you find most helpful during the adaptive session?", [
    { value: "focus_point",          label: "Focus point prompts (curiosity hooks)" },
    { value: "comprehension_check",  label: "Comprehension checks (true/false questions)" },
    { value: "section_summary",      label: "Section summaries" },
    { value: "re_engagement",        label: "Re-engagement messages" },
    { value: "ambient_sound",        label: "Background soundscapes" },
    { value: "chime",                label: "Focus chime" },
    { value: "gamification",         label: "The Walk of Words journey" },
    { value: "none",                 label: "None — I did not find any particularly helpful" },
  ], { label: "Intervention types" }),

  mcMulti("int_least_helpful", "Which types of support, if any, were least helpful or most disruptive? (select all that apply)", [
    { value: "focus_point",          label: "Focus point prompts" },
    { value: "comprehension_check",  label: "Comprehension checks" },
    { value: "section_summary",      label: "Section summaries" },
    { value: "re_engagement",        label: "Re-engagement messages" },
    { value: "ambient_sound",        label: "Background soundscapes" },
    { value: "chime",                label: "Focus chime" },
    { value: "gamification",         label: "The Walk of Words journey" },
    { value: "none",                 label: "None — all were acceptable" },
  ]),

  // ── Trust & transparency ─────────────────────────────────────────────────
  likert("trust_accurate",  "I felt the system accurately detected when I was losing focus.",         { label: "Trust & Transparency" }),
  likert("trust_trust",     "I trusted the system to make appropriate decisions about when to intervene."),
  likert("trust_aware",     "Being aware that the system was monitoring my attention was distracting."),

  // ── Cognitive load perception ─────────────────────────────────────────────
  likert("cl_easy",         "Reading with the adaptive system felt cognitively manageable.",          { label: "Cognitive Load" }),
  likert("cl_compare",      "The adaptive session felt less mentally demanding than the standard session.",
    { scaleLeft: "Strongly disagree", scaleRight: "Strongly agree" }),

  // ── Preference & willingness to use ──────────────────────────────────────
  mc("pref_session", "Overall, which reading session did you prefer?", [
    { value: "baseline", label: "The standard session (no AI interventions)" },
    { value: "adaptive", label: "The adaptive session (with AI interventions)" },
    { value: "no_pref",  label: "No strong preference" },
  ], { label: "Overall preference" }),

  mc("pref_frequency", "How often would you want AI reading support if you used this system regularly?", [
    { value: "never",      label: "Never — I prefer reading without any AI support" },
    { value: "heavy_drift","label": "Only when I am clearly distracted" },
    { value: "occasional", label: "Occasionally, as background support" },
    { value: "always",     label: "Always — throughout every reading session" },
  ]),

  likert("adopt_recommend", "I would recommend this system to other students.",                      { label: "Adoption" }),
  likert("adopt_deploy",    "I would use this system if it were freely available."),

  // ── Open feedback ─────────────────────────────────────────────────────────
  open("open_best",      "What did you find most useful or positive about the system? (optional)", 3),
  open("open_improve",   "What would you change or improve about the system? (optional)", 3),
  open("open_other",     "Any other comments or observations? (optional)", 2),
];
