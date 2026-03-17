from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved once at import time so it's stable regardless of cwd.
# services/api/app/core/config.py → up 5 levels → repo root
# (core → app → api → services → repo root)
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "Lock-In API"
    app_version: str = "0.1.0"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://lockin:lockin@localhost:5433/lockin"

    # Auth
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # File storage — relative to wherever the API process runs
    upload_dir: Path = Path("uploads")
    # Extracted images and docling artifacts (git-ignored)
    parsed_cache_dir: Path = Path("parsed_cache")
    # Telemetry CSV exports (git-ignored)
    exports_dir: Path = Path("exports")
    # Training data exports — stored at repo root training/exports/ (git-ignored)
    # Override via TRAINING_EXPORTS_DIR env var if needed.
    training_exports_dir: Path = _REPO_ROOT / "training" / "exports"
    # Consolidated training datasets (CSV / JSONL) written by training_export router.
    # Override via TRAINING_DATA_DIR env var if needed.
    training_data_dir: Path = _REPO_ROOT / "training" / "data"


settings = Settings()
