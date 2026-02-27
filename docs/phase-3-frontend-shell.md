# Phase 3 — Frontend Shell: Auth + Dashboard (No PDF yet)

**Branch:** `base_frontend`
**Completed:** 2026-02-25
**Commit:** `feat(frontend): Phase 3 — auth + dashboard shell (no PDF yet)`

---

## Overview

Phase 3 builds the complete React + TypeScript frontend in `apps/desktop/`, wiring it
directly to the FastAPI backend from Phase 2. The goals for this phase are:

1. A working auth flow (register, login, JWT persistence on refresh, protected routes, logout).
2. A functional dashboard: upload PDFs, view your document library, start/view reading sessions.
3. One command (`./start.sh`) that starts the entire stack — database, API, and frontend —
   with zero other steps.
4. A Tauri v2 scaffold so the app can be wrapped as a native `.dmg` desktop app whenever needed.

The design is intentionally functional-first (dark theme, clean layout, no heavy UI framework)
so that visual redesign decisions can be made once the app is running and visible.

---

## Architecture Decisions

### No UI framework (CSS custom properties instead)

No component library (MUI, shadcn, etc.) was introduced. All styling is done with CSS
custom properties defined in `index.css`. Reasons:

- Keeps the bundle small and fast to iterate on.
- Makes a future visual redesign trivial — changing `--accent` or `--bg-base` affects the
  entire app instantly.
- Avoids fighting a framework's opinion about layout for a highly custom reading environment.

### Inline `<style>` blocks in page components

Page-specific CSS is colocated with each component as a `<style>` JSX block rather than in
separate `.module.css` files. This keeps all logic and appearance for a given screen in one
file, making it easy to see what a page does and looks like at a glance.

### `localStorage` for JWT persistence

The JWT (`lockin_token`) is stored in `localStorage` rather than a cookie or in-memory state.
Reasons for this choice at the thesis prototype stage:

- Survives page refreshes without any server session infrastructure.
- Simple to implement and audit.
- On app startup `AuthContext` validates the stored token by calling `GET /auth/me`; if the
  token is expired or invalid the key is cleared automatically.

### Flat service layer (no React Query / SWR)

API calls live in four plain async service modules (`authService`, `documentService`,
`sessionService`, `apiClient`). No caching library was added at this stage because:

- The app has a single user and a small data set.
- Keeping state local to each page component makes the data flow obvious and debuggable.
- A caching layer can be added transparently later without changing any component logic.

### Vite-first, Tauri-ready

`pnpm dev` starts the Vite dev server in a browser tab — fast hot-reload, instant feedback.
`pnpm tauri dev` wraps the same Vite server inside a native Tauri window. Both modes use
identical source code; the difference is only in how the webview is hosted.

---

## Directory Structure Created

```
apps/desktop/
├── index.html                        ← Vite entry point with CSP header
├── package.json                      ← @lock-in/desktop workspace package
├── tsconfig.json                     ← TypeScript config (ES2020, strict)
├── tsconfig.node.json                ← Vite config type-checking
├── vite.config.ts                    ← Vite + Vitest config (port 5173, strict)
│
├── src/
│   ├── main.tsx                      ← ReactDOM.createRoot entry point
│   ├── App.tsx                       ← BrowserRouter + route definitions
│   ├── index.css                     ← Global CSS design tokens + utilities
│   │
│   ├── contexts/
│   │   └── AuthContext.tsx           ← JWT state, token storage, session restore
│   │
│   ├── components/
│   │   └── ProtectedRoute.tsx        ← Redirects to /login if unauthenticated
│   │
│   ├── services/
│   │   ├── apiClient.ts              ← Central fetch wrapper (JWT injection, ApiError)
│   │   ├── authService.ts            ← login, register, getMe
│   │   ├── documentService.ts        ← list, upload, remove
│   │   └── sessionService.ts         ← list, start, end, complete
│   │
│   ├── pages/
│   │   ├── LoginPage.tsx             ← Sign-in / create-account tab switcher
│   │   └── HomePage.tsx              ← Dashboard: stats, documents, sessions
│   │
│   └── test/
│       ├── setup.ts                  ← Vitest setup (imports jest-dom matchers)
│       ├── apiClient.test.ts         ← 6 unit tests for the fetch wrapper
│       └── AuthContext.test.tsx      ← 7 integration tests for auth lifecycle
│
└── src-tauri/                        ← Tauri v2 Rust backend (thin shell only)
    ├── Cargo.toml
    ├── build.rs
    ├── tauri.conf.json
    ├── capabilities/
    │   └── default.json              ← core:default permissions
    └── src/
        ├── main.rs                   ← Entry point: calls app_lib::run()
        └── lib.rs                    ← tauri::Builder::default().run()
```

---

## Files Created / Modified

| File | Change | Purpose |
|---|---|---|
| `apps/desktop/package.json` | NEW | `@lock-in/desktop` workspace; deps: React 18, react-router-dom; dev: Vite, Vitest, Testing Library, Tauri CLI |
| `apps/desktop/tsconfig.json` | NEW | Strict TypeScript for browser (ES2020, `bundler` module resolution) |
| `apps/desktop/tsconfig.node.json` | NEW | TypeScript config for `vite.config.ts` |
| `apps/desktop/vite.config.ts` | NEW | Vite + `@vitejs/plugin-react`; Vitest `jsdom` environment; port 5173 |
| `apps/desktop/index.html` | NEW | Vite entry HTML with Content-Security-Policy allowing `localhost:8000` |
| `apps/desktop/src/main.tsx` | NEW | React 18 `createRoot` bootstrap |
| `apps/desktop/src/App.tsx` | NEW | `BrowserRouter` + three routes: `/login`, `/` (protected), `*` (catch-all) |
| `apps/desktop/src/index.css` | NEW | CSS design tokens (`--bg-base`, `--accent`, `--border`, etc.), resets, shared utilities (buttons, badges, cards, spinner, modal overlay) |
| `apps/desktop/src/contexts/AuthContext.tsx` | NEW | Stores JWT in `localStorage`; restores session on mount via `GET /auth/me`; exposes `login`, `register`, `logout` |
| `apps/desktop/src/components/ProtectedRoute.tsx` | NEW | Shows spinner while loading; redirects to `/login` if no token |
| `apps/desktop/src/services/apiClient.ts` | NEW | `apiRequest<T>` wrapper — injects `Authorization: Bearer`, throws `ApiError(status, message)`, handles 204 |
| `apps/desktop/src/services/authService.ts` | NEW | `login` (form-encoded for OAuth2), `register` (JSON), `getMe` |
| `apps/desktop/src/services/documentService.ts` | NEW | `list`, `upload` (FormData multipart), `remove` |
| `apps/desktop/src/services/sessionService.ts` | NEW | `list`, `start`, `end`, `complete` |
| `apps/desktop/src/pages/LoginPage.tsx` | NEW | Tab switcher (Sign in / Create account), inline error banner, spinner while submitting |
| `apps/desktop/src/pages/HomePage.tsx` | NEW | Stats row (4 cards), document list + upload modal, session grid + start-session modal |
| `apps/desktop/src-tauri/Cargo.toml` | NEW | Tauri v2 `[lib]` crate with `staticlib + cdylib + rlib` targets |
| `apps/desktop/src-tauri/build.rs` | NEW | `tauri_build::build()` |
| `apps/desktop/src-tauri/tauri.conf.json` | NEW | Product name, identifier, `devUrl` (5173), `frontendDist` (`../dist`), window sizing |
| `apps/desktop/src-tauri/capabilities/default.json` | NEW | `core:default` permission set |
| `apps/desktop/src-tauri/src/main.rs` | NEW | `app_lib::run()` entry point, `windows_subsystem = "windows"` guard |
| `apps/desktop/src-tauri/src/lib.rs` | NEW | `tauri::Builder::default().run()` |
| `apps/desktop/src/test/setup.ts` | NEW | Imports `@testing-library/jest-dom` matchers |
| `apps/desktop/src/test/apiClient.test.ts` | NEW | 6 unit tests for `apiRequest` |
| `apps/desktop/src/test/AuthContext.test.tsx` | NEW | 7 integration tests for `AuthContext` |
| `start.sh` | NEW | One-command dev launcher (API + Vite or Tauri) |
| `package.json` (root) | UPDATED | Added `dev`, `dev:tauri`, `dev:ui`, `test:frontend`, `test:backend` scripts |
| `pnpm-lock.yaml` | NEW | Lockfile generated by `pnpm install` |
| `apps/desktop/.gitkeep` | DELETED | Placeholder no longer needed |

---

## CSS Design System (`index.css`)

All colours and spacing values are CSS custom properties on `:root` so a visual redesign
requires only changing token values, not hunting through component files.

| Token | Value | Usage |
|---|---|---|
| `--bg-base` | `#0d0d0d` | Page background |
| `--bg-surface` | `#161616` | Cards, panels, header |
| `--bg-elevated` | `#1f1f1f` | Inputs, menus |
| `--bg-hover` | `#272727` | Row hover state |
| `--border` | `#2a2a2a` | Dividers, input borders |
| `--border-focus` | `#6366f1` | Focused input ring |
| `--accent` | `#6366f1` | Primary buttons, brand dot |
| `--accent-dim` | `#4f52c4` | Button hover |
| `--accent-glow` | `rgba(99,102,241,0.15)` | Focus glow, active mode bg |
| `--text` | `#e5e5e5` | Body text |
| `--text-muted` | `#888` | Labels, metadata |
| `--text-faint` | `#555` | Placeholder, separator dots |
| `--status-active` | `#22c55e` | Active session badge |
| `--status-paused` | `#f59e0b` | Paused session badge |
| `--status-ended` | `#6b7280` | Ended session badge |
| `--status-completed` | `#6366f1` | Completed session badge |
| `--error` | `#ef4444` | Error banners, danger buttons |

Shared utility classes built on top of these tokens: `.btn`, `.btn--primary`,
`.btn--ghost`, `.btn--danger`, `.btn--sm`, `.btn--full`, `.badge`, `.badge--{status}`,
`.card`, `.form-group`, `.modal-overlay`, `.modal`, `.splash`, `.spinner`,
`.error-banner`.

---

## Routing (`App.tsx`)

| Path | Component | Auth required |
|---|---|---|
| `/login` | `LoginPage` | No |
| `/` | `HomePage` (wrapped in `ProtectedRoute`) | Yes — redirects to `/login` |
| `*` | Redirects to `/` | — |

`ProtectedRoute` checks two states:
1. `isLoading = true` → renders a full-screen spinner (token is being validated against the API).
2. `token = null` → `<Navigate to="/login" replace />`.

---

## Service Layer

### `apiClient.ts` — `apiRequest<T>(path, options)`

```typescript
// Injects Bearer token and parses JSON.
// Throws ApiError(status, message) on non-2xx.
// Returns undefined for 204 No Content.
await apiRequest<User>('/auth/me', { token: 'jwt...' });
```

`ApiError` carries the HTTP `status` code so callers can branch on 401 vs 403 vs 404.

### `authService.ts`

| Function | HTTP | Notes |
|---|---|---|
| `login(username, password)` | `POST /auth/login` | **Form-encoded** (`application/x-www-form-urlencoded`) — required by FastAPI's `OAuth2PasswordRequestForm` |
| `register(username, email, password)` | `POST /auth/register` | JSON body |
| `getMe(token)` | `GET /auth/me` | Used on mount to validate stored token |

### `documentService.ts`

| Function | HTTP | Notes |
|---|---|---|
| `list(token)` | `GET /documents` | Returns `{ documents, total }` |
| `upload(token, title, file)` | `POST /documents/upload` | `FormData` — `title` (text) + `file` (PDF blob) |
| `remove(token, docId)` | `DELETE /documents/{id}` | 204 on success |

### `sessionService.ts`

| Function | HTTP | Notes |
|---|---|---|
| `list(token)` | `GET /sessions` | Returns `{ sessions, total }` |
| `start(token, documentId, name, mode)` | `POST /sessions/start` | `mode` is `'baseline'` or `'adaptive'` |
| `end(token, sessionId)` | `POST /sessions/{id}/end` | Sets status → `ended` |
| `complete(token, sessionId)` | `POST /sessions/{id}/complete` | Sets status → `completed` (thesis completion-rate tracking) |

---

## Auth Context (`AuthContext.tsx`)

```
mount
  └─ read localStorage('lockin_token')
       ├─ no token  → isLoading=false, user=null
       └─ token found → GET /auth/me
            ├─ success → user=..., isLoading=false   ← session restored
            └─ 401/error → clear localStorage, user=null, isLoading=false
```

`login()` and `register()` both follow the same two-step pattern:
1. Call the API → receive `access_token`.
2. Call `GET /auth/me` with the new token → receive the `User` object.
3. Write both to `localStorage` and React state atomically.

`logout()` clears `localStorage` and resets state to `{ user: null, token: null }`.

---

## Pages

### `LoginPage.tsx`

- Two-tab switcher: **Sign in** / **Create account** (rendered with a CSS grid pill, no JS tab library).
- The Email field is only rendered when the "Create account" tab is active.
- Form validation is delegated to the backend (FastAPI/Pydantic returns descriptive messages).
- Errors from the API are displayed in the `.error-banner` below the form fields.
- Navigates to `/` on success via `react-router-dom`'s `useNavigate`.

### `HomePage.tsx` (Dashboard)

**Stats row** — four metric cards computed from loaded data:

| Card | Source |
|---|---|
| Documents | `documents.length` |
| Sessions | `sessions.length` |
| Completed | sessions where `status === 'completed'` |
| Completion rate | `Math.round((completed / total) * 100)%` |

**Documents section**

- Lists all documents for the logged-in user, each as a `DocumentRow` showing title, filename, size, and upload date.
- **Delete**: calls `documentService.remove()`, removes the row from local state optimistically.
- **Start session**: opens the `StartSessionModal` for that document.
- **Upload PDF** button opens `UploadModal`.

**Upload modal** (`UploadModal`)

- File input (`accept=".pdf"`) + title field (auto-populated from filename).
- Calls `documentService.upload()` on submit; on success prepends the new document to the list.

**Start session modal** (`StartSessionModal`)

- Session name (pre-filled: `"[doc title] — [today's date]"`).
- Mode selector — two pill buttons: **baseline** (log only, no interventions) and **adaptive** (AI intervention loop active). These map directly to the `mode` field in the `sessions` table.
- Calls `sessionService.start()` on submit; on success prepends the new session to the grid.

**Sessions section** — grid of `SessionCard` components, each showing:
- Session name
- Status badge (colour-coded: green=active, amber=paused, grey=ended, indigo=completed)
- Mode (baseline / adaptive), start date, duration (if finished)
- Limited to the 12 most recent sessions on the dashboard.

---

## One-Command Launcher (`start.sh`)

```bash
./start.sh           # API (uvicorn :8000) + Vite browser (localhost:5173)
./start.sh --tauri   # API (uvicorn :8000) + Tauri native desktop window
```

**What the script does, in order:**

1. **Pre-flight checks** — verifies the Python venv exists and `node_modules` is installed
   (runs `pnpm install` automatically if missing).
2. **Database** — checks if the `lockin-db` Docker container is running; starts it via
   `docker compose up -d` if not, then waits 3 seconds for readiness.
3. **FastAPI** — activates `.venv`, starts `uvicorn app.main:app --reload --port 8000` in
   the background, saves its PID.
4. **Frontend** — starts either `pnpm dev` (Vite, opens in browser) or `pnpm tauri dev`
   (Tauri desktop window, first run compiles Rust which takes a few minutes).
5. **Cleanup** — a `trap` on `SIGINT`/`SIGTERM` kills the API process cleanly when you
   press Ctrl+C.

**Root `package.json` shortcuts added:**

| Script | Runs |
|---|---|
| `pnpm dev` | `./start.sh` — API + Vite browser |
| `pnpm dev:tauri` | `./start.sh --tauri` — API + Tauri desktop |
| `pnpm dev:ui` | `pnpm --filter @lock-in/desktop dev` — Vite browser only |
| `pnpm test:frontend` | `pnpm --filter @lock-in/desktop test` — Vitest suite |
| `pnpm test:backend` | `.venv/bin/python -m pytest` in `services/api/` |

---

## Tauri v2 Scaffold (`src-tauri/`)

The Tauri scaffold is minimal — it is a thin native window that hosts the Vite webview.
No Tauri IPC commands are used yet; all data flows over HTTP to the FastAPI backend.

| File | Content |
|---|---|
| `Cargo.toml` | `tauri = "2"`, `serde`, `serde_json`; lib crate with `staticlib + cdylib + rlib` |
| `build.rs` | `tauri_build::build()` |
| `tauri.conf.json` | `devUrl: http://localhost:5173`, `frontendDist: ../dist`, 1280×820 window, `minWidth: 900` |
| `capabilities/default.json` | `core:default` permission set (minimal) |
| `src/main.rs` | `app_lib::run()` — `#[cfg_attr(not(debug_assertions), windows_subsystem = "windows")]` |
| `src/lib.rs` | `tauri::Builder::default().run()` |

To open the native Tauri window:
```bash
cd apps/desktop && pnpm tauri dev
# or from the repo root:
./start.sh --tauri
```

The first run compiles all Rust crates (~3–6 minutes on Apple Silicon). Subsequent runs
are instant thanks to Cargo's incremental compilation cache.

---

## Test Suite

Run from `apps/desktop/`:

```bash
pnpm test
# or from the repo root:
pnpm test:frontend
```

### `apiClient.test.ts` — 6 tests

| Test | Assertion |
|---|---|
| Sends GET by default and returns parsed JSON | Correct URL called, body returned |
| Injects Authorization header when token is provided | `Bearer <token>` in headers |
| Does NOT inject Authorization when token is null | No `Authorization` key in headers |
| Serialises body as JSON and sets Content-Type | `application/json`, stringified body |
| Throws `ApiError` with status code on non-2xx | `{ name: 'ApiError', status: 404, message: 'Not found' }` |
| Returns undefined for 204 No Content | Return value is `undefined` |

### `AuthContext.test.tsx` — 7 tests

| Group | Test | Assertion |
|---|---|---|
| Initial load | `isLoading` resolves to `false` with no token | User = null, token = null |
| Initial load | Restores session when valid token is in `localStorage` | `getMe` called, user populated |
| Initial load | Clears storage when stored token is invalid | `localStorage` cleared, user = null |
| Login | Stores token + populates user on success | `localStorage` updated, username shown |
| Login | Propagates errors from `authService.login` | Error message thrown, `localStorage` unchanged |
| Register | Stores token + populates user on success | Same as login |
| Logout | Clears token from state and `localStorage` | Both cleared |

**Mock strategy:** `authService` is fully mocked via `vi.mock(...)`. Tests never make real
HTTP requests. `localStorage` is cleared before and after each test in `beforeEach`/`afterEach`.

### Results

```
 RUN  v2.1.9

 ✓ src/test/apiClient.test.ts     (6 tests)
 ✓ src/test/AuthContext.test.tsx  (7 tests)

 Test Files  2 passed (2)
      Tests  13 passed (13)
   Duration  ~1.0s
```

**All 13 tests passed.**

---

## How to Run the App

### Full stack (recommended)

```bash
# Make sure Docker Desktop is running
./start.sh

# Then open:  http://localhost:5173
```

### Individual services

```bash
# API only
pnpm dev:api

# Frontend (Vite browser) only — requires API already running
pnpm dev:ui

# Frontend tests only
pnpm test:frontend

# Backend tests only
pnpm test:backend
```

### First-time setup (if node_modules not yet installed)

```bash
cd apps/desktop
pnpm install
```

---

## Known Limitations / Deferred to Next Phase

| Item | Status |
|---|---|
| PDF reader | Deliberately excluded — Phase 4 |
| Active session timer UI (pause/resume controls) | Deferred — sessions can be started but only via the modal; in-session controls come with the reader |
| Telemetry hooks (`POST /activity`) | Not wired yet — will fire on scroll/focus/blur events inside the PDF reader |
| Tauri native file dialog for PDF upload | Using browser `<input type="file">` for now; will switch to `@tauri-apps/api` dialog in Phase 4 |
| `act(...)` warnings in AuthContext tests | Cosmetic only — appear because `userEvent` triggers async state updates inside `act()`; all assertions pass correctly |
| Error boundary | No React error boundary yet — add before thesis experiments |

---

## What Is Next

- **Phase 4 — PDF Reader**: Integrate `react-pdf` (PDF.js) into the dashboard. Clicking a
  document opens a full-screen canvas reader. The active session state is managed in React,
  and `POST /activity` events fire on scroll, focus, blur, and heartbeat.
- Wire pause / resume / complete controls into the reader overlay.
- Add the Tauri native file dialog for uploads.
