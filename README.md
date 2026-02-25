# Lock-In

A local-first desktop application for university students with ADHD. Monitors implicit attention signals during reading and delivers AI-generated micro-interventions to restore focus.

## Monorepo Layout

```
lock-in/
├── apps/desktop/        # React 18 + Tauri v2 frontend
├── packages/shared/     # Shared TypeScript types
├── services/api/        # Python FastAPI backend
├── training/            # LLM fine-tuning (later phase)
├── docker-compose.yml   # TimescaleDB (port 5433)
└── pnpm-workspace.yaml
```

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Node.js | 20 | `nvm install 20 && nvm use 20` |
| pnpm | 9+ | `npm install -g pnpm` |
| Python | 3.11 | `pyenv install 3.11` or direct |
| Docker Desktop | latest | https://docker.com |
| Rust + Tauri CLI | stable | see below |

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli
```

## Quick Start

```bash
# 1. Start the database
docker compose up -d

# 2. Set up Python env
cd services/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Copy env
cp .env.example .env   # edit as needed

# 4. Run the API
uvicorn app.main:app --reload --port 8000
```

## Development Phases

- [x] Phase 1 — Environment & Infrastructure
- [ ] Phase 2 — Database schema + Alembic migrations
- [ ] Phase 3 — Auth (register / login / JWT)
- [ ] Phase 4 — Document upload
- [ ] Phase 5 — Session management
- [ ] Phase 6 — Telemetry ingestion
- [ ] Phase 7 — Tauri desktop shell
- [ ] Phase 8 — PDF reader
- [ ] Phase 9 — Attention model + interventions
- [ ] Phase 10 — LLM fine-tuning pipeline
