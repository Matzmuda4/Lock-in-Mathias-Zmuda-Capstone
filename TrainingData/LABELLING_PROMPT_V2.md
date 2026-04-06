# Lock-in Intervention Dataset — Labelling Prompt (V2)

You are labelling training examples for a fine-tuned LLM called the **Lock-in Intervention Engine**.
This LLM is embedded inside a reading assistant. It receives real-time signals about a student's attentional state and the text they are reading, and it must decide whether to intervene and generate the content of that intervention.

---

## ⚠️ CRITICAL — READ THIS BEFORE ANYTHING ELSE

**`content` is ALWAYS a JSON object, never a plain string.**

Every intervention type has its own required object shape. If you output a plain string like `"Quick check: what does this mean?"` you are doing it wrong. Every `content` field must be a structured JSON object with the exact keys listed in the schema below.

| Type | content shape (keys required) |
|---|---|
| `focus_point` | `{"headline": "...", "body": "...", "cta": "..."}` |
| `section_summary` | `{"title": "...", "summary": "...", "key_point": "..."}` |
| `re_engagement` | `{"headline": "...", "body": "...", "cta": "..."}` |
| `comprehension_check` | `{"type": "true_false"\|"highlight", "question": "...", "answer": true\|false\|"phrase", "explanation": "..."}` |
| `break_suggestion` | `{"headline": "...", "message": "...", "duration_minutes": 5}` |
| `gamification` | `{"event": "journey_start"\|"milestone"\|"badge_earned"\|"xp_boost", "message": "..."}` |
| `none` | `null` |

**Wrong — do NOT do this:**
```json
{"id": "s177_w67", "intervene": true, "tier": "moderate", "content": "Quick check: what is the main reason..."}
```

**Correct — always do this:**
```json
{"id": "s177_w67", "intervene": true, "tier": "moderate", "content": {"type": "true_false", "question": "The study found that single-modality attention tracking is sufficient for real-time classification.", "answer": false, "explanation": "The section argues that combining EEG and eye-tracking significantly improves classification accuracy over either alone."}}
```

---

You will be given batches of JSON objects. Each object represents one moment in a reading session. Your job is to fill in three fields:

- `intervene` — always `true` for these examples
- `tier` — already filled; do not change it unless it is clearly wrong
- `content` — **a JSON object** following the exact schema for the `output.type` (see table above and full schemas below)

Return **only** a JSON array of objects, one per input, each containing exactly:
```json
{
  "id": "<same id as the input>",
  "intervene": true,
  "tier": "<same tier as input, or corrected>",
  "content": { <structured object — NOT a string> }
}
```

Do not include any other text, explanation, or markdown outside the JSON array.

---

## Context You Will Receive (Input Fields)

Each input contains:

- `output.type` — the intervention type already decided by the system (do NOT change this)
- `output.tier` — subtle / moderate / strong / special
- `output.rationale` — the signal-based reason this intervention was selected
- `input.reading_context.text_window` — array of 1–3 paragraphs the student is currently reading. **This is the most important field.** Your content must be grounded in this text.
- `input.attentional_state_window` — last 3 attentional state classifications (focused / drifting / hyperfocused / cognitive_overload)
- `input.drift_progression.drift_ema` — exponentially-smoothed drift score (0.0 = fully engaged, 1.0 = completely disengaged)
- `input.session_context.session_stage` — early / mid / late

---

## Content Schemas (by type)

### `focus_point`
A curiosity-spark prompt that makes the student want to keep reading.

> ⚠️ `content` must be a JSON **object** with keys `headline`, `body`, `cta`. **Never a plain string.**

```json
{
  "headline": "string — a question or intriguing hook phrased from the text (max 12 words)",
  "body": "string — 1–2 sentences that raise an interesting angle or implication from the text_window (max 40 words)",
  "cta": "string — action label, e.g. 'Find out' | 'Keep reading' | 'Dig deeper'"
}
```

**Rules:**
- The `headline` MUST reference a specific concept, entity, or idea from `text_window`. Never write "Did you know…" or "Want to find out more?" generically.
- Do NOT ask "what happens next?" — instead, highlight an unexpected or counterintuitive implication.
- Vary style: some can be rhetorical questions, some can be declarative provocations ("There is a reason why…"), some can be incomplete thoughts ("What if the opposite were true?")
- The `body` must add context beyond what the headline says.

**Bad example (do NOT do this):**
```json
{"headline": "Keep going, you're close!", "body": "This section is doing important setup.", "cta": "I'm back"}
```

**Good example:**
```json
{
  "headline": "Why does System 2 slow down when reading gets hard?",
  "body": "The dual-process theory suggests our fast, intuitive System 1 is what we rely on — until the text forces deliberate processing. What does that switching cost look like neurologically?",
  "cta": "Keep reading"
}
```

---

### `section_summary`
A synthesised, paraphrased summary of what the text_window covers. This is NOT a copy of the text.

> ⚠️ `content` must be a JSON **object** with keys `title`, `summary`, `key_point`. **Never a plain string.**

```json
{
  "title": "string — 3–5 word title capturing the theme (not just extracted nouns)",
  "summary": "string — 2–3 sentence original paraphrase that synthesises the main idea. Must NOT reproduce any full sentence from the text.",
  "key_point": "string — one-sentence distillation of the single most important takeaway"
}
```

**Rules:**
- Read the `text_window` carefully. Then write the `summary` in your own words as if explaining to a peer — do not lift any phrase longer than 3 words from the source.
- The `title` should reflect the theme, not be a list of keywords. "Memory and Attention" is better than "Working Memory Capacity Study".
- The `key_point` should be actionable or insightful, not just descriptive.

**Bad example (verbatim — do NOT do this):**
```json
{
  "title": "Cognitive Load Theory Proposes",
  "summary": "Cognitive load theory proposes three distinct categories of intrinsic, extraneous, and germane load.",
  "key_point": "The text states that working memory capacity has been studied extensively."
}
```

**Good example:**
```json
{
  "title": "How Mental Load Compounds",
  "summary": "Our working memory operates under strict capacity limits — the way a task is designed either burns that capacity on irrelevant details or channels it into actual learning. Understanding which is happening helps explain why some texts feel exhausting even when short.",
  "key_point": "Cognitive load is a design problem as much as a learner problem — structure shapes how hard your brain works."
}
```

---

### `re_engagement`
A personalised prompt to pull a drifting reader back. It must feel like it was written for this exact text, not a generic nudge.

> ⚠️ `content` must be a JSON **object** with keys `headline`, `body`, `cta`. **Never a plain string.**

```json
{
  "headline": "string — a short (max 8 words) hook that names or alludes to something specific in the text_window",
  "body": "string — 1–2 sentences that create urgency or curiosity about returning to the text (max 35 words)",
  "cta": "string — e.g. 'Re-read the last line' | 'Start here' | 'Jump back in'"
}
```

**Rules:**
- The `headline` MUST contain a specific word or concept from `text_window`. If the text is about procrastination in graduate students, a valid headline is "Graduate students and the procrastination trap" — NOT "Still with us?" or "Focus check".
- Never use: "Still with us?", "Take a moment", "Focus check", "Quick check", "Staying with us?", "Hey reader", or any variant.
- The `body` should name one specific thing from the text that the student is about to miss.
- Vary tone: some can be urgent, some warm, some intriguing.

**Bad example:**
```json
{"headline": "Still with us?", "body": "This section is doing important setup. Take one breath, revisit the last line, and jump back in.", "cta": "I'm back"}
```

**Good example (text was about procrastination):**
```json
{
  "headline": "The 70–95% stat is harder to ignore than you think.",
  "body": "Steel's meta-analysis found that nearly all undergrads procrastinate. The next paragraph explains exactly why graduate students do it differently — and it's not what you'd expect.",
  "cta": "Keep reading"
}
```

---

### `comprehension_check`
A short quiz generated directly from a specific claim in the `text_window`. Either True/False or a highlight-to-answer question.

> ⚠️ `content` must be a JSON **object** with keys `type`, `question`, `answer`, `explanation`. **Never a plain string question.**

Choose `true_false` when a clear factual claim exists in the text. Choose `highlight` when you want the student to locate evidence.

```json
{
  "type": "true_false" | "highlight",
  "question": "string — a specific testable question about the text_window",
  "answer": true | false | "string of the answer phrase",
  "explanation": "string — why this answer is correct, referencing the text (max 30 words)"
}
```

**Rules for `true_false`:**
- The question must be answerable from the `text_window` alone — not general knowledge.
- Do NOT use: "You can clearly recall the main idea introduced in the last paragraph." — this is a self-reflection question, not a comprehension check.
- Mix true and false answers. False questions test active comprehension.
- Vary difficulty: some should test literal recall, some should test inference.

**Bad example:**
```json
{"type": "true_false", "question": "You can clearly recall the main idea introduced in the last paragraph you read.", "answer": true, "explanation": "A self-anchoring check."}
```

**Good example (text about SAM model):**
```json
{
  "type": "true_false",
  "question": "The Self-Assessment Manikin (SAM) measures emotional response on a single dimension.",
  "answer": false,
  "explanation": "SAM measures multiple dimensions — valence, arousal, and dominance — not a single scale."
}
```

**Rules for `highlight`:**
- The `question` should ask the student to find a specific phrase or sentence.
- The `answer` should be a short verbatim phrase from the text (3–10 words).
- Only use this format when the text actually contains a clearly quotable answer.

---

### `break_suggestion`
A suggestion that the student takes a short break. Content should be calm, not alarming.

> ⚠️ `content` must be a JSON **object** with keys `headline`, `message`, `duration_minutes`. **Never a plain string.**

```json
{
  "headline": "string — a calm, non-alarming title for the break (max 8 words)",
  "message": "string — 1–2 sentences explaining why a break is being suggested, referencing the reading context (max 40 words)",
  "duration_minutes": 5
}
```

**Rules:**
- Vary the `headline` — avoid repeating the same phrasing. Use: "Time to step back", "A moment away from the page", "Your brain earned a pause", "A five-minute reset", "The text will wait", "Brief recharge — then back".
- The `message` should acknowledge what the student has been doing (reading long, high cognitive load, struggling to focus) without being alarming.
- `duration_minutes` is always 5.

---

### `gamification`
Signals the system to trigger or update the gamification journey. The LLM selects the event type and writes a brief motivational message.

> ⚠️ `content` must be a JSON **object** with keys `event` and `message`. **Never a plain string.**

```json
{
  "event": "journey_start" | "milestone" | "badge_earned" | "xp_boost",
  "message": "string — short motivational message (max 20 words), specific to the reading context"
}
```

**Rules:**
- Use `journey_start` at early session (stage = "early") when the student is focused.
- Use `milestone` when a checkpoint is near (mid/late session, sustained focus).
- Use `badge_earned` when the signal indicates a notable streak (check `rationale`).
- Use `xp_boost` for moderate-tier gamification during drifting recovery.
- The `message` should be specific: if the text is about cognitive load theory, say "You're navigating some dense theory — your focus is paying off." Not "Keep it up!"

---

### `none`
No intervention warranted. Student is on track.

```json
null
```

(Return `content: null` for these rows.)

---

### `ambient_sound` (if any appear as pending)
```json
{
  "track": "nature" | "pink_noise" | "brown_noise",
  "fade_in_seconds": 3 | 5 | 8
}
```
- `nature` for mild drifting; `pink_noise` for moderate; `brown_noise` for overload.
- Faster fade for stronger tiers.

---

## Critical Rules (Apply to All Types)

1. **Never copy a full sentence from `text_window` into any content field.** Paraphrase, synthesise, or ask questions about it.
2. **Every text-generative content must be grounded in `text_window`.** If the text is about ADHD behavioural inhibition, your `re_engagement` headline should mention ADHD, inhibition, or Douglas — not be generic.
3. **Vary structure within each type.** Across a batch, no two `focus_point` headlines should have the same phrasing pattern. Vary question types for `comprehension_check`.
4. **For `none` entries:** just return `content: null`. These teach the model restraint.
5. **Match intensity to tier:** subtle content is gentle, warm, low-pressure. Strong content is more urgent and direct.

---

## How to Process a Batch

You will be given a JSON array of skeleton rows. For each:
1. Read the `text_window` carefully — this is your primary source.
2. Check `output.type`, `output.tier`, and `output.rationale`.
3. Generate the appropriate `content` object following the schema and rules above.
4. Return exactly the fields: `id`, `intervene: true`, `tier`, `content`.

---

## Output Format

Return a valid JSON array only. No markdown formatting, no explanations.

```json
[
  {
    "id": "s210_w5",
    "intervene": true,
    "tier": "subtle",
    "content": { ... }
  },
  ...
]
```

PLEASE ENSURE THE OUTPUT IS IN A JSON BLOCK THAT I CAN COPY EASILY AND PASTE DIRECTLY AT NO POINT WILL YOU JUST OUTPUT TEXT, YOU'RE ONLY OUTPUT IS A JSON CELL WITH THE ADEQUATE LABELS INSIDE IT

If you are unsure about any row, still label it — do your best based on the `text_window` and the rules above.
