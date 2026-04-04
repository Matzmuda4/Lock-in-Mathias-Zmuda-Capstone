#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "=== Step 1: Merge adapter into base model ==="
.venv/bin/python3 merge_adapter.py

echo ""
echo "=== Step 2: Install llama.cpp if needed ==="
if ! command -v llama-quantize &>/dev/null; then
    brew install llama.cpp
fi

echo ""
echo "=== Step 3: Convert to GGUF ==="
bash convert_to_gguf.sh

echo ""
echo "✓ All done — lockin-intervention is ready in Ollama"
