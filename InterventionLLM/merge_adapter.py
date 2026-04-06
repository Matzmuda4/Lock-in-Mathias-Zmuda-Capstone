"""
merge_adapter.py
────────────────
Merges the QLoRA LoRA adapter into the Qwen 2.5 7B Instruct base model and
saves the merged full-precision model to InterventionLLM/lockin_merged/.

Run from the InterventionLLM directory:
  python merge_adapter.py

Requirements (installed automatically if missing):
  transformers peft accelerate safetensors sentencepiece

Estimated time  : 10-20 min (includes ~14 GB model download on first run)
Peak RAM needed : ~14 GB (bfloat16 weights, Apple Silicon unified memory)
Disk needed     : ~15 GB for merged model output
"""

from __future__ import annotations
import subprocess
import sys
import os

print("Dependencies expected in venv — skipping pip install.")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE         = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(HERE, "Lockin_adapter_final")
MERGED_PATH  = os.path.join(HERE, "lockin_merged")
# Use the locally-downloaded model to avoid re-downloading from HuggingFace
_local_base  = os.path.join(HERE, "Qwen2.5-7B-Instruct")
BASE_MODEL   = _local_base if os.path.isdir(_local_base) else "Qwen/Qwen2.5-7B-Instruct"

print(f"\n{'─'*60}")
print(f"Base model   : {BASE_MODEL}")
print(f"Adapter path : {ADAPTER_PATH}")
print(f"Output path  : {MERGED_PATH}")
print(f"{'─'*60}\n")

# ── 1. Load base model ────────────────────────────────────────────────────────
# low_cpu_mem_usage=True uses memory-mapped tensors — they are paged in on
# demand rather than all at once.  Critical for 16 GB unified-memory Macs.
print("Step 1/4 — Loading base model (bfloat16, memory-mapped)…")
print(f"          Loading from: {BASE_MODEL}\n")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",
    trust_remote_code=True,
)
print("Base model loaded.\n")

# ── 2. Apply LoRA adapter ─────────────────────────────────────────────────────
print("Step 2/4 — Applying LoRA adapter…")
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    torch_dtype=torch.bfloat16,
)
print("Adapter applied.\n")

# ── 3. Merge adapter weights into base model ──────────────────────────────────
# merge_and_unload() adds scaled LoRA deltas directly into the base weights and
# removes the adapter structure, yielding a standard HuggingFace model.
print("Step 3/4 — Merging adapter into base weights…")
model = model.merge_and_unload()
print("Merge complete.\n")

# ── 4. Save merged model ──────────────────────────────────────────────────────
print(f"Step 4/4 — Saving merged model to {MERGED_PATH} …")
os.makedirs(MERGED_PATH, exist_ok=True)
model.save_pretrained(MERGED_PATH, safe_serialization=True, max_shard_size="5GB")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_PATH)

print(f"\n✓ Merged model saved to: {MERGED_PATH}")
print("  Next step: run convert_to_gguf.sh to produce the Ollama-ready GGUF file.")
