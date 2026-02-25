#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — one-command dev launcher for Lock-In
#
# Usage:
#   ./start.sh           # API + Vite browser (fast dev / UI iteration)
#   ./start.sh --tauri   # API + Tauri desktop window (full native app)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ──────────────────────────────────────────────────────────────────
BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m';  NC='\033[0m'

log()  { echo -e "${BLUE}[lock-in]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC}     $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}   $*"; }
err()  { echo -e "${RED}[error]${NC}  $*"; }

TAURI_MODE=false
[[ "${1:-}" == "--tauri" ]] && TAURI_MODE=true

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ ! -f "$ROOT/services/api/.venv/bin/activate" ]]; then
  err "Python venv not found at services/api/.venv"
  err "Run:  cd services/api && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "$ROOT/apps/desktop/node_modules" ]]; then
  warn "node_modules missing — running pnpm install..."
  (cd "$ROOT/apps/desktop" && pnpm install)
fi

# ── Check Docker / DB ─────────────────────────────────────────────────────────
if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "lockin-db"; then
  warn "Database container not running — starting it..."
  (cd "$ROOT" && docker compose up -d)
  log "Waiting for DB to be healthy..."
  sleep 3
fi

# ── Start FastAPI backend ─────────────────────────────────────────────────────
log "Starting FastAPI API on http://localhost:8000 ..."
(
  cd "$ROOT/services/api"
  source .venv/bin/activate
  uvicorn app.main:app --reload --port 8000
) &
API_PID=$!

# Give the API a moment to initialise tables
sleep 2
ok "API running (pid $API_PID)"

# ── Cleanup trap ──────────────────────────────────────────────────────────────
cleanup() {
  echo ""
  log "Shutting down..."
  kill "$API_PID" 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

# ── Start frontend ────────────────────────────────────────────────────────────
if $TAURI_MODE; then
  log "Starting Tauri desktop app (first run compiles Rust — may take a few minutes)..."
  (cd "$ROOT/apps/desktop" && pnpm tauri dev)
else
  log "Starting Vite dev server on http://localhost:5173 ..."
  (cd "$ROOT/apps/desktop" && pnpm dev)
fi

cleanup
