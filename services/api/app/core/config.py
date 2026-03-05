from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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


settings = Settings()
