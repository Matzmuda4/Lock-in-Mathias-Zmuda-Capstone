import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import activity, auth, calibration, classification, documents, drift, exports, interventions, parsing, sessions, training_export

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from app.db.engine import init_db

    await init_db()

    # ── Attentional-state classifier (RF production) ──────────────────────────
    # Priority: MockClassifier > RFClassifier > disabled.
    # Safe to skip: if classify_enabled=False, classification never runs.
    if settings.classify_enabled:
        from app.services.classifier.registry import set_classifier

        if settings.classify_use_mock:
            from app.services.classifier.mock import MockClassifier
            set_classifier(MockClassifier())
            log.info("Classifier: MockClassifier active (CLASSIFY_USE_MOCK=true).")

        elif settings.classify_use_rf:
            from app.services.classifier.rf_classifier import RFClassifier
            clf = RFClassifier(model_path=settings.rf_model_path)
            if clf.is_loaded():
                set_classifier(clf)
                log.info(
                    "Classifier: RFClassifier ready — model='%s'.",
                    settings.rf_model_path.name,
                )
            else:
                log.warning(
                    "Classifier: RF model not found at '%s'. "
                    "Classification disabled. Copy rf_classifier_v2.pkl to the "
                    "repo root or set RF_MODEL_PATH in .env.",
                    settings.rf_model_path,
                )

        else:
            # Legacy Ollama path — retained for future intervention LLM use
            from app.services.classifier.ollama import OllamaClassifier
            ollama_clf = OllamaClassifier(
                base_url=settings.ollama_url,
                model=settings.ollama_classifier_model,
            )
            ok = await ollama_clf.health_check()
            if ok:
                set_classifier(ollama_clf)
                log.info(
                    "Classifier: OllamaClassifier ready — model='%s'.",
                    settings.ollama_classifier_model,
                )
            else:
                log.warning(
                    "Classifier: Ollama health check failed for model='%s'. "
                    "Classification disabled.",
                    settings.ollama_classifier_model,
                )
    else:
        log.info(
            "Classifier: disabled (CLASSIFY_ENABLED=false). "
            "Set CLASSIFY_ENABLED=true and CLASSIFY_USE_RF=true in .env."
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
app.include_router(interventions.router)


@app.get("/health", tags=["meta"])
async def health_check() -> dict:
    return {"status": "ok", "version": settings.app_version}
