#!/usr/bin/env bash
# convert_to_gguf.sh
# ──────────────────
# Converts the merged HuggingFace model in lockin_merged/ to a GGUF Q4_K_M
# file and registers it with Ollama as "lockin-intervention".
#
# Run AFTER merge_adapter.py completes:
#   bash convert_to_gguf.sh
#
# Requires: brew install llama.cpp  (installs llama-cli, llama-quantize, convert scripts)
# Disk needed: ~10 GB for intermediate f16 GGUF + ~4.5 GB for final Q4_K_M

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
MERGED="$HERE/lockin_merged"
GGUF_DIR="$HERE/lockin_gguf"
GGUF_F16="$GGUF_DIR/lockin_f16.gguf"
GGUF_Q4="$GGUF_DIR/lockin_intervention.gguf"
MODELFILE="$HERE/Modelfile"

echo "──────────────────────────────────────────────"
echo " Lock-in Intervention LLM — GGUF Conversion"
echo "──────────────────────────────────────────────"

# ── 0. Verify merged model exists ─────────────────────────────────────────────
if [ ! -d "$MERGED" ]; then
    echo "ERROR: $MERGED not found. Run merge_adapter.py first."
    exit 1
fi

# ── 1. Install llama.cpp via Homebrew (if not already installed) ──────────────
if ! command -v llama-quantize &>/dev/null; then
    echo "Step 1/4 — Installing llama.cpp via Homebrew…"
    brew install llama.cpp
else
    echo "Step 1/4 — llama.cpp already installed ($(llama-quantize --version 2>&1 | head -1))"
fi
echo ""

mkdir -p "$GGUF_DIR"

# ── 2. Convert merged HF model → GGUF f16 ─────────────────────────────────────
# Locate llama.cpp's convert script — brew installs it alongside the binaries
CONVERT_SCRIPT="$(brew --prefix llama.cpp)/bin/convert_hf_to_gguf.py"
if [ ! -f "$CONVERT_SCRIPT" ]; then
    # Fallback: some versions place it in a different path
    CONVERT_SCRIPT="$(brew --prefix llama.cpp)/libexec/convert_hf_to_gguf.py"
fi
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "ERROR: convert_hf_to_gguf.py not found in llama.cpp install."
    echo "       Try: brew reinstall llama.cpp"
    exit 1
fi

echo "Step 2/4 — Converting merged model to GGUF f16…"
# Use our venv's Python so transformers/sentencepiece are available
VENV_PYTHON="$HERE/.venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi
"$VENV_PYTHON" "$CONVERT_SCRIPT" \
    "$MERGED" \
    --outtype f16 \
    --outfile "$GGUF_F16"
echo "f16 GGUF written to: $GGUF_F16"
echo ""

# ── 3. Quantise f16 → Q4_K_M ──────────────────────────────────────────────────
# Q4_K_M is the recommended quantisation for Ollama: good balance of quality
# and speed, ~4.5 GB file size.
echo "Step 3/4 — Quantising to Q4_K_M…"
llama-quantize "$GGUF_F16" "$GGUF_Q4" Q4_K_M
echo "Q4_K_M GGUF written to: $GGUF_Q4"
echo ""

# Remove the intermediate f16 to reclaim ~10 GB
echo "Removing intermediate f16 GGUF to free disk space…"
rm -f "$GGUF_F16"

# ── 4. Write Ollama Modelfile ──────────────────────────────────────────────────
echo "Step 4/4 — Writing Ollama Modelfile…"
cat > "$MODELFILE" <<'MODELFILE_CONTENT'
FROM ./lockin_gguf/lockin_intervention.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 2048

SYSTEM """You are an adaptive reading assistant engine embedded in a digital reading tool called Lock-in. Every 10 seconds you receive a 30-second window of signals about a student's attentional state, drift trajectory, and the text they are currently reading. Your task is to: (1) identify the most appropriate intervention type and tier (subtle | moderate | strong | special) based on the signals, and always generate its content; (2) set 'intervene' to true only when cooldown_status is 'clear' — if cooldown_status is 'cooling', set 'intervene' to false but still output the full content of what you would have fired, so the system can schedule it for the next clear window; (3) if no intervention is warranted at all (tier='none'), set 'intervene' to false and 'content' to null. Respond with a single valid JSON object only — no prose, no markdown fences."""
MODELFILE_CONTENT

echo "Modelfile written to: $MODELFILE"
echo ""

# ── 5. Register with Ollama ────────────────────────────────────────────────────
echo "Registering model with Ollama as 'lockin-intervention'…"
cd "$HERE" && ollama create lockin-intervention -f "$MODELFILE"

echo ""
echo "══════════════════════════════════════════════"
echo " ✓ lockin-intervention registered with Ollama"
echo "══════════════════════════════════════════════"
echo ""
echo "Test it with:"
echo "  ollama run lockin-intervention"
echo ""
echo "Or via the API:"
echo '  curl http://localhost:11434/api/generate -d '"'"'{"model":"lockin-intervention","prompt":"{}","stream":false}'"'"''
