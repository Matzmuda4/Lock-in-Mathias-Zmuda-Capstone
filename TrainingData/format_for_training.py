"""
format_for_training.py
──────────────────────
Converts intervention_training_v2_labelled.jsonl (Alpaca-style) into two files
ready for QLoRA fine-tuning on Qwen 2.5 7B:

  intervention_train_v2.jsonl   (~90%)
  intervention_eval_v2.jsonl    (~10%)

Each line contains a single key "text" whose value is a fully-formatted
Qwen ChatML string that SFTTrainer / Unsloth can consume directly.

Format:
  <|im_start|>system
  {instruction}<|im_end|>
  <|im_start|>user
  {json(input)}<|im_end|>
  <|im_start|>assistant
  {json(output)}<|im_end|>

Run:
  python TrainingData/format_for_training.py

Outputs land in the same TrainingData/ directory.
"""

from __future__ import annotations
import json
import pathlib
import random

random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = pathlib.Path(__file__).parent
SRC         = HERE / "intervention_training_v2_labelled.jsonl"
TRAIN_OUT   = HERE / "intervention_train_v2.jsonl"
EVAL_OUT    = HERE / "intervention_eval_v2.jsonl"

# Qwen 2.5 uses ChatML; these special tokens must match exactly.
BOS = "<|im_start|>"
EOS = "<|im_end|>"

# ─── Put this constant near the top of format_for_training.py ───────────────

ENRICHED_INSTRUCTION = """\
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


EVAL_FRACTION = 0.10


# ─── Then in format_example(), replace ex["instruction"] with the constant ────
def format_example(ex: dict) -> str:
    inp     = ex["input"]
    out     = ex["output"]
    user_msg = json.dumps(inp, ensure_ascii=False, separators=(",", ":"))
    asst_msg = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{BOS}system\n{ENRICHED_INSTRUCTION}{EOS}\n"
        f"{BOS}user\n{user_msg}{EOS}\n"
        f"{BOS}assistant\n{asst_msg}{EOS}"
    )


def main() -> None:
    # ── Load source ──────────────────────────────────────────────────────────
    records: list[dict] = []
    with SRC.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} examples from {SRC.name}")

    # ── Format ───────────────────────────────────────────────────────────────
    formatted = [{"text": format_example(r)} for r in records]

    # ── Stratified shuffle then split ────────────────────────────────────────
    # Stratify by intervention type so eval covers all 10 types.
    from collections import defaultdict
    by_type: dict[str, list] = defaultdict(list)
    for item, rec in zip(formatted, records):
        out = rec.get("output", {})
        # output may be a dict or a JSON string — handle both
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except json.JSONDecodeError:
                out = {}
        itype = out.get("type", "unknown")
        by_type[itype].append(item)

    # ── Token-count-balanced upsampling ──────────────────────────────────────
    # The causal LM loss is proportional to output token count. Short-output
    # types (chime: ~57 tok, text_reformat: ~56 tok) receive 5x less gradient
    # signal than section_summary (~291 tok) at equal example counts. Upsample
    # the training slice only (eval stays clean at 8 examples per type).
    UPSAMPLE_FACTORS: dict[str, int] = {
        "none":          4,   # 47 tok × 4 ≈ 188  → close to section_summary 291
        "chime":         4,   # 57 tok × 4 ≈ 228
        "text_reformat": 4,   # 56 tok × 4 ≈ 224
        "ambient_sound": 4,   # 56 tok × 4 ≈ 224
        "focus_point":   2,   # 186 tok × 2 ≈ 372
        "gamification":  2,   # 121 tok × 2 ≈ 242
        # break_suggestion, re_engagement, comprehension_check,
        # section_summary: 1× (already long-output; no upsampling needed)
    }

    train_items: list[dict] = []
    eval_items:  list[dict] = []

    for itype, items in by_type.items():
        random.shuffle(items)
        n_eval = max(1, round(len(items) * EVAL_FRACTION))
        eval_items.extend(items[:n_eval])
        factor = UPSAMPLE_FACTORS.get(itype, 1)
        train_items.extend(items[n_eval:] * factor)

    random.shuffle(train_items)
    random.shuffle(eval_items)

    # ── Write ─────────────────────────────────────────────────────────────────
    for path, items in [(TRAIN_OUT, train_items), (EVAL_OUT, eval_items)]:
        with path.open("w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Written: {path.name}  ({len(items)} examples)")

    # ── Sanity-check first example ────────────────────────────────────────────
    print("\n── First training example preview ───────────────────────────────")
    first = train_items[0]["text"]
    print(first[:600] + "\n  ...[truncated]")

    # ── Token-length estimate ─────────────────────────────────────────────────
    all_lens = [len(item["text"]) // 4 for item in train_items + eval_items]
    print(f"\n── Approx token lengths ─────────────────────────────────────────")
    print(f"  min={min(all_lens)}  avg={sum(all_lens)//len(all_lens)}  max={max(all_lens)}")
    print(f"  All examples fit within 2048 tokens: {all(l <= 2048 for l in all_lens)}")


if __name__ == "__main__":
    main()
