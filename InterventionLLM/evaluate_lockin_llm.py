"""
evaluate_lockin_llm.py
──────────────────────
Evaluates the Lock-in intervention LLM on the held-out eval set.
Uses the same Ollama / GGUF Q4_K_M model that runs in production —
making this evaluation directly representative of real system behaviour.

Run from the repo root (Ollama must be running):
  ollama serve &   # if not already running
  python InterventionLLM/evaluate_lockin_llm.py

Requirements: pip install requests
Time: ~15-25 min on Apple Silicon
"""

from __future__ import annotations
import json, re, time, pathlib
import requests
from collections import Counter, defaultdict

HERE        = pathlib.Path(__file__).parent
EVAL_PATH   = HERE.parent / "TrainingData" / "intervention_eval_v2.jsonl"
RESULTS_OUT = HERE / "evaluation_results.json"

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "lockin-intervention"

# ── System prompt — must match ENRICHED_INSTRUCTION in format_for_training.py ─
SYSTEM_PROMPT = """\
You are an adaptive reading assistant engine embedded in a digital reading tool called \
Lock-in. Every 10 seconds you receive a 30-second window of signals about a student's \
attentional state, drift trajectory, and the text they are currently reading. Your task \
is to decide:
  (1) whether to intervene and which type + tier is most appropriate;
  (2) generate the exact content for that intervention based on what the student is reading.

COOLDOWN RULE: if session_context.cooldown_status is "cooling", you MUST set \
intervene: false. Still output the type and content you would have fired so the \
system can schedule it — but intervene must be false.

Output a single JSON object with exactly these fields:
  intervene  : true | false
  type       : one of [focus_point, section_summary, comprehension_check, re_engagement,
               ambient_sound, chime, text_reformat, break_suggestion, gamification, none]
  tier       : subtle | moderate | strong | special | none
  content    : object — exact required shape per type:
    chime               : {"sound": "gentle_bell"|"double_tap", "message": "<2-4 word prompt>"}
    ambient_sound       : {"track": "pink_noise"|"brown_noise"|"nature", "fade_in_seconds": <4-10>}
    text_reformat       : {"line_spacing": <1.5|1.7|2.0>, "chunk_size": <1|2|3>}
    gamification        : {"event": "journey_start"|"milestone"|"xp_boost", "message": "<specific to reading>"}
    focus_point         : {"headline": "...", "body": "...", "cta": "..."}
    re_engagement       : {"headline": "...", "body": "...", "cta": "..."}
    section_summary     : {"title": "...", "summary": "...", "key_point": "..."}
    comprehension_check : {"type": "true_false", "question": "...", "answer": true|false, "explanation": "..."}
    break_suggestion    : {"headline": "...", "message": "...", "duration_minutes": 5}
    none                : null

Type guide (one line per type — brief signal hints only):
  none              : student is focused or hyperfocused with no anomalies; no action needed
  chime             : any early or brief attention lapse; lightest nudge, no text required; fires before re_engagement when drift first appears
  focus_point       : attention beginning to waver; curiosity hook grounded in the text
  gamification      : focused progress milestone; reward the student; do not fire when drift is rising
  re_engagement     : sustained drifting across multiple packets; direct text pull-back needed
  ambient_sound     : mild sustained drift; background audio without interrupting reading
  comprehension_check : focused or hyperfocused for a sustained period; verify encoding with a true/false question
  section_summary   : rising drift over a dense passage; synthesised recap helps re-orient
  text_reformat     : severe cognitive overload with very high drift; layout relief (spacing/chunking) needed, not a text prompt
  break_suggestion  : persistent cognitive overload that text changes alone cannot address; full break required

Always ground text-generative content in text_window. Never copy sentences verbatim.\
"""

# ── Schemas & valid values ─────────────────────────────────────────────────────
VALID_TYPES = {
    "focus_point", "section_summary", "comprehension_check", "re_engagement",
    "ambient_sound", "chime", "text_reformat", "break_suggestion", "gamification", "none",
}
VALID_TIERS = {"subtle", "moderate", "strong", "special", "none"}
REQUIRED_FIELDS = {"intervene", "type", "tier", "content"}
CONTENT_SCHEMA = {
    "chime":               {"sound", "message"},
    "ambient_sound":       {"track", "fade_in_seconds"},
    "text_reformat":       {"line_spacing", "chunk_size"},
    "gamification":        {"event", "message"},
    "focus_point":         {"headline", "body", "cta"},
    "re_engagement":       {"headline", "body", "cta"},
    "section_summary":     {"title", "summary", "key_point"},
    "comprehension_check": {"type", "question", "answer", "explanation"},
    "break_suggestion":    {"headline", "message", "duration_minutes"},
    "none":                None,
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_json_safe(raw: str):
    """Find the first balanced JSON object in raw output, tolerating preamble/trailing tokens."""
    raw = raw.strip()
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass
    for start in range(len(raw)):
        if raw[start] != '{':
            continue
        depth, in_str, escape = 0, False, False
        for end in range(start, len(raw)):
            ch = raw[end]
            if escape:            escape = False; continue
            if ch == '\\' and in_str: escape = True; continue
            if ch == '"':         in_str = not in_str; continue
            if in_str:            continue
            if ch == '{':         depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:    return json.loads(raw[start:end + 1]), None
                    except: break
    return None, "no valid JSON object found"


def gt_type_from_text(text: str) -> str:
    asst = text.split("<|im_start|>assistant")[-1]
    m = re.search(r'"type"\s*:\s*"([^"]+)"', asst)
    return m.group(1) if m else "unknown"


def gt_cooling(text: str) -> bool:
    user_part = text.split("<|im_start|>user")[-1].split("<|im_start|>assistant")[0]
    return '"cooling"' in user_part


# ── Verify Ollama is reachable ─────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print(f"  Lock-in Intervention LLM — Evaluation")
print(f"{'─' * 60}")
print(f"  Model  : {OLLAMA_MODEL}  (Ollama / GGUF Q4_K_M)")
print(f"  Eval   : {EVAL_PATH.name}  (80 examples, 8 per type)")
print(f"{'─' * 60}\n")

try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    if not any(OLLAMA_MODEL in m for m in models):
        raise RuntimeError(f"Model '{OLLAMA_MODEL}' not found in Ollama. Run: bash InterventionLLM/convert_to_gguf.sh")
    print(f"Ollama reachable — model '{OLLAMA_MODEL}' ready\n")
except requests.exceptions.ConnectionError:
    raise RuntimeError("Ollama is not running. Start it with:  ollama serve")

# ── Load eval set ──────────────────────────────────────────────────────────────
eval_examples: list[dict] = []
with open(EVAL_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            eval_examples.append(json.loads(line))
print(f"Loaded {len(eval_examples)} eval examples\n")

# ── Inference via Ollama API ───────────────────────────────────────────────────
FORCED_PREFIX = '{"intervene":'

results:        list[dict] = []
parse_failures: list[dict] = []
t_start = time.time()

print(f"{'─' * 60}")
print(f"  {'#':>3}  {'GT type':<24}  {'Predicted type':<24}  {'ETA':>8}")
print(f"{'─' * 60}")

for i, ex in enumerate(eval_examples):
    user_msg = ex["text"].split("<|im_start|>user")[-1].split("<|im_end|>")[0].strip()

    payload = {
        "model":  OLLAMA_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": 300,   # max ~300 tokens — all outputs fit well within this
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
            # Seed the assistant turn with the forced prefix to prevent user-input echo
            {"role": "assistant", "content": FORCED_PREFIX},
        ],
    }

    response  = requests.post(OLLAMA_URL, json=payload, timeout=300)  # 300s for slow CPU
    resp_json = response.json()
    continuation = resp_json.get("message", {}).get("content", "")
    raw_output   = FORCED_PREFIX + continuation.strip()

    gt_type    = gt_type_from_text(ex["text"])
    is_cooling = gt_cooling(ex["text"])
    parsed, err = extract_json_safe(raw_output)

    pred_type = parsed.get("type", "FAIL") if parsed else "FAIL"
    match_str = "✓" if pred_type == gt_type else " "

    elapsed = time.time() - t_start
    eta_min = elapsed / (i + 1) * (len(eval_examples) - i - 1) / 60
    print(f"  {i+1:>3}  {gt_type:<24}  {pred_type:<24}  {eta_min:>6.1f}min  {match_str}")

    results.append({
        "index":      i,
        "raw":        raw_output,
        "parsed":     parsed,
        "parse_err":  err,
        "gt_type":    gt_type,
        "is_cooling": is_cooling,
    })
    if parsed is None:
        parse_failures.append({"gt": gt_type, "raw": raw_output[:150]})

# ── Score ──────────────────────────────────────────────────────────────────────
n = len(results)
scores = {k: 0 for k in [
    "json_valid", "fields_complete", "tier_valid", "type_valid",
    "type_accuracy", "cooldown_logic", "content_schema",
]}
pred_types   = Counter()
gt_types     = Counter()
schema_fails = defaultdict(list)

for r in results:
    p, gt = r["parsed"], r["gt_type"]
    gt_types[gt] += 1
    if p is None:
        continue

    scores["json_valid"] += 1

    if not REQUIRED_FIELDS.issubset(p.keys()):
        continue

    pred_type = p.get("type", "")
    pred_tier = p.get("tier", "")
    pred_types[pred_type] += 1
    scores["fields_complete"] += 1

    if pred_tier in VALID_TIERS:  scores["tier_valid"]    += 1
    if pred_type in VALID_TYPES:  scores["type_valid"]    += 1
    if pred_type == gt:           scores["type_accuracy"] += 1

    if r["is_cooling"]:
        if p.get("intervene") is False: scores["cooldown_logic"] += 1
    else:
        if p.get("intervene") is True:  scores["cooldown_logic"] += 1

    expected = CONTENT_SCHEMA.get(pred_type)
    content  = p.get("content")
    if expected is None:
        if content is None:
            scores["content_schema"] += 1
    elif isinstance(content, dict) and expected.issubset(content.keys()):
        scores["content_schema"] += 1
    else:
        schema_fails[pred_type].append(r["raw"][:80])

# ── Print results ──────────────────────────────────────────────────────────────
total_time = time.time() - t_start
print(f"\n{'══' * 35}")
print(f"  EVALUATION RESULTS  ({n} examples)  —  merged model, local inference")
print(f"{'══' * 35}")

metric_labels = [
    ("1. JSON valid",          "json_valid"),
    ("2. Fields complete",     "fields_complete"),
    ("3. Tier valid",          "tier_valid"),
    ("4. Type valid",          "type_valid"),
    ("5. Type accuracy (=GT)", "type_accuracy"),
    ("6. Cooldown logic",      "cooldown_logic"),
    ("7. Content schema",      "content_schema"),
]
for label, key in metric_labels:
    v = scores[key]
    bar = "█" * int(20 * v / n)
    print(f"  {label:<26}: {v:>2}/{n}  ({100*v/n:5.1f}%)  {bar}")

print(f"\n{'──' * 35}")
print(f"  Predicted type distribution vs. ground truth")
print(f"{'──' * 35}")
all_types = sorted(VALID_TYPES | set(pred_types.keys()))
print(f"  {'Type':<28}  {'Pred':>6}  {'GT':>6}  {'Match%':>7}")
print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*7}")
for t in all_types:
    pred_c = pred_types.get(t, 0)
    gt_c   = gt_types.get(t, 0)
    correct = sum(
        1 for r in results
        if r["parsed"] and r["parsed"].get("type") == t and r["gt_type"] == t
    )
    ms = f"{100 * correct // max(gt_c, 1)}%" if gt_c > 0 else "—"
    print(f"  {t:<28}  {pred_c:>6}  {gt_c:>6}  {ms:>7}")

unexpected = {t for t in pred_types if t not in VALID_TYPES}
if unexpected:
    print(f"\n  ⚠  Hallucinated types (tokenizer artifacts): {unexpected}")

if schema_fails:
    print(f"\n{'──' * 35}")
    print("  Content schema failures by type")
    print(f"{'──' * 35}")
    for t, samples in schema_fails.items():
        print(f"  {t}: {len(samples)} failure(s)  — sample: {samples[0][:80]}")

if parse_failures:
    print(f"\n{'──' * 35}")
    print(f"  JSON parse failures ({len(parse_failures)} total)")
    print(f"{'──' * 35}")
    for pf in parse_failures[:3]:
        print(f"  GT: {pf['gt']:<22} | {repr(pf['raw'][:100])}")

print(f"\n  Total eval time: {total_time/60:.1f} min  ({total_time/n:.1f}s per example)")

# ── Save full results JSON for thesis reporting ────────────────────────────────
output_data = {
    "model":       str(MERGED_PATH),
    "eval_file":   str(EVAL_PATH),
    "n_examples":  n,
    "eval_time_min": round(total_time / 60, 1),
    "scores": {
        k: {"count": v, "pct": round(100 * v / n, 1)}
        for k, v in scores.items()
    },
    "pred_distribution": dict(pred_types),
    "gt_distribution":   dict(gt_types),
    "per_type_accuracy": {
        t: {
            "pred": pred_types.get(t, 0),
            "gt":   gt_types.get(t, 0),
            "correct": sum(
                1 for r in results
                if r["parsed"] and r["parsed"].get("type") == t and r["gt_type"] == t
            ),
        }
        for t in sorted(VALID_TYPES)
    },
    "parse_failures": parse_failures,
    "per_example": [
        {
            "index":      r["index"],
            "gt_type":    r["gt_type"],
            "pred_type":  r["parsed"].get("type") if r["parsed"] else None,
            "json_valid": r["parsed"] is not None,
            "fields_ok":  r["parsed"] is not None and REQUIRED_FIELDS.issubset(r["parsed"].keys()),
            "correct":    r["parsed"] is not None and r["parsed"].get("type") == r["gt_type"],
            "is_cooling": r["is_cooling"],
            "raw_truncated": r["raw"][:200],
        }
        for r in results
    ],
}

with open(RESULTS_OUT, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n  Results saved → {RESULTS_OUT}")
print("  (Full per-example breakdown available for thesis appendix)\n")
