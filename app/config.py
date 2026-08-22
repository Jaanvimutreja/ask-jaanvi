"""
Application configuration loaded from environment variables.

Settings are managed via pydantic-settings, which reads from .env files
and environment variables. All secrets (e.g., GROQ_API_KEY) are loaded
from the environment and never hardcoded.

Usage:
    from app.config import get_settings
    settings = get_settings()
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "Ask Jaanvi"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Server ───────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000

    # ── CORS ─────────────────────────────────────────────────────
    # TODO(production): Replace with actual production origins.
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins. Restrict to production domains before deployment.",
    )

    # ── Groq LLM (Phase 4) ──────────────────────────────────────
    # The API key is required for LLM calls but not for startup.
    # The model is intentionally configurable — do not hardcode a default
    # that assumes availability. Verify against Groq's current model list.
    groq_api_key: str = ""
    groq_model: str = Field(
        default="",
        description="Groq model ID. Left empty until verified in Phase 4.",
    )
    groq_max_tokens: int = 1024
    groq_timeout: float = 30.0

    @property
    def groq_configured(self) -> bool:
        """Check whether the Groq API key has been provided."""
        return bool(self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Uses lru_cache so the .env file is read once per process.
    In tests, override this dependency via FastAPI's dependency_overrides.
    """
    return Settings()
