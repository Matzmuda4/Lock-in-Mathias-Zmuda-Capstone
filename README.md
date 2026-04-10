# Lock-In

A local-first desktop reading assistant for university students with ADHD. Lock-In monitors implicit attentional signals during sustained reading and delivers AI-generated micro-interventions to restore focus — entirely on-device, with no cloud dependency.

---

## Architecture

```
lock-in/
├── apps/desktop/          # React 18 + Tauri v2 desktop frontend
├── services/api/          # Python FastAPI backend (port 8000)
│   └── app/
│       ├── routers/       # auth, sessions, documents, classification,
│       │                  # drift, interventions, calibration, study_setup
│       └── services/
│           ├── classifier/    # Random Forest attentional-state classifier
│           └── intervention/  # LLM engine, prompt builder, rule fallbacks
├── InterventionLLM/       # Qwen 2.5-7B adapter + GGUF export artefacts
├── TrainingData/          # Classifier & LLM training datasets + scripts
├── UserStudy/             # Condition texts (Descartes / Russell) for study
├── rf_classifier_v2.pkl   # Trained RF model (committed, ~2 MB)
├── docker-compose.yml     # TimescaleDB on port 5433
└── start.sh               # One-command dev launcher
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Node.js | 20+ |
| pnpm | 9+ |
| Python | 3.11 |
| Docker Desktop | latest |
| Rust + Tauri CLI | stable |
| Ollama | latest |

```bash
# Node / pnpm
nvm install 20 && nvm use 20
npm install -g pnpm

# Rust + Tauri
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli

# Ollama (runs the intervention LLM locally)
brew install ollama          # macOS
# or: https://ollama.com/download
```

---

## Quick Start

### 1. Database

```bash
docker compose up -d
# TimescaleDB available at localhost:5433
```

### 2. Python backend

```bash
cd services/api
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # defaults work out of the box
```

### 3. Intervention LLM (Ollama)

The fine-tuned Qwen 2.5-7B model must be running locally before the adaptive reader will fire interventions.

```bash
ollama serve                # start Ollama daemon (if not already running)
# then, from InterventionLLM/:
ollama create lockin -f Modelfile
ollama run lockin           # verify it responds
```

> The GGUF model weights (`lockin_gguf/`) are not committed — pull them separately or run `convert_to_gguf.sh` after merging the adapter.

### 4. Launch the app

```bash
# From repo root — choose one:
./start.sh           # Vite browser dev mode  (http://localhost:5173)
./start.sh --tauri   # Full native Tauri window (first run compiles Rust)
```

`start.sh` automatically checks the venv, starts Docker if needed, and launches both the API and the frontend.

---

## Environment Variables (`services/api/.env`)

| Key | Default | Description |
|-----|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://lockin:lockin@localhost:5433/lockin` | TimescaleDB connection |
| `CLASSIFY_ENABLED` | `true` | Enable RF classifier |
| `CLASSIFY_USE_RF` | `true` | Use trained model (false = mock) |
| `RF_MODEL_PATH` | *(repo root)* | Override path to `rf_classifier_v2.pkl` |
| `DEBUG` | `false` | FastAPI debug mode |

---

## Key Features

- **Attentional state classification** — 19-feature Random Forest (macro F1 0.903, ROC-AUC 0.971) classifies Focused / Hyperfocused / Drifting / Cognitive Overload in real time from reading telemetry.
- **LLM intervention engine** — Fine-tuned Qwen 2.5-7B-Instruct (QLoRA, GGUF via Ollama) generates context-aware interventions (10 types, 5 operational categories) with signal-based rule fallbacks.
- **Calibration** — Personalized baseline normalisation per user before adaptive sessions begin.
- **User study pipeline** — 16-step guided flow with NASA-TLX, SUS, demographics, and automatic server-side data export to `experimentresults/`.
- **Privacy-first** — All inference runs locally; no data leaves the machine.

---

## Archiving / Submitting

The `InterventionLLM/` model weights (~34 GB) and `node_modules/` are git-ignored. To create a clean source archive:

```bash
git archive --format=zip HEAD -o lockin_source.zip
```
