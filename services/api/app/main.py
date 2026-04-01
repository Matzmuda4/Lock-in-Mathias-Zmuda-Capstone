import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import activity, auth, calibration, classification, documents, drift, exports, parsing, sessions, training_export

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from app.db.engine import init_db

    await init_db()

    # ── Phase 9: Attentional-state classifier ─────────────────────────────────
    # Loads the configured classifier once at startup and registers it in the
    # module-level registry so store.py and drift.py can access it without DI.
    # Safe to skip: if classify_enabled=False, classification simply never runs.
    if settings.classify_enabled:
        from app.services.classifier.registry import set_classifier

        if settings.classify_use_mock:
            from app.services.classifier.mock import MockClassifier
            set_classifier(MockClassifier())
            log.info("Classifier: MockClassifier active (CLASSIFY_USE_MOCK=true).")
        else:
            from app.services.classifier.ollama import OllamaClassifier
            clf = OllamaClassifier(
                base_url=settings.ollama_url,
                model=settings.ollama_classifier_model,
            )
            ok = await clf.health_check()
            if ok:
                set_classifier(clf)
                log.info(
                    "Classifier: OllamaClassifier ready — model='%s'.",
                    settings.ollama_classifier_model,
                )
            else:
                log.warning(
                    "Classifier: Ollama health check failed for model='%s'. "
                    "Classification disabled until the model is available. "
                    "Check that Ollama is running and the model is loaded.",
                    settings.ollama_classifier_model,
                )
    else:
        log.info(
            "Classifier: disabled (CLASSIFY_ENABLED=false). "
            "Set CLASSIFY_ENABLED=true in .env when the adapter is ready."
        )

    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:1420",  # Tauri dev server
        "tauri://localhost",      # Tauri production webview (macOS/Linux)
        "https://tauri.localhost",# Tauri production webview (Windows)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(sessions.router)
app.include_router(activity.router)
app.include_router(parsing.router)
app.include_router(calibration.router)
app.include_router(drift.router)
app.include_router(exports.router)
app.include_router(training_export.router)
app.include_router(classification.router)


@app.get("/health", tags=["meta"])
async def health_check() -> dict:
    return {"status": "ok", "version": settings.app_version}
