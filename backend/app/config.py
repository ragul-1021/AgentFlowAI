"""Typed application settings, loaded from environment variables / .env.

Secrets are marked ``repr=False`` and never logged. The database URL must use
SQLAlchemy's asyncpg dialect and TLS for non-local hosts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ALLOWED_ENVS = {"development", "test", "production"}
_ALLOWED_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
_TLS_VALUES = {"require", "true", "verify-ca", "verify-full"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class Settings(BaseSettings):
    """Application configuration.

    Values come from environment variables first, then ``backend/.env`` when
    the process runs from the backend directory.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", _BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "agentflow"
    app_env: str = "development"
    log_level: str = "INFO"
    ai_mock_mode: bool = False

    # Database — required. Kept out of repr/logs.
    database_url: str = Field(min_length=1, repr=False)
    migration_database_url: str | None = Field(default=None, repr=False)

    # AI providers — optional at startup; health reports whether each is set.
    # Runs will require the relevant keys once provider integrations land.
    gemini_api_key: str | None = Field(default=None, repr=False)
    groq_api_key: str | None = Field(default=None, repr=False)
    tavily_api_key: str | None = Field(default=None, repr=False)
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = ""
    gemini_timeout: float = Field(default=30.0, gt=0, le=300)
    groq_timeout: float = Field(default=30.0, gt=0, le=300)
    tavily_timeout: float = Field(default=20.0, gt=0, le=300)
    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_base_delay: float = Field(default=1.0, ge=0.01, le=10.0)
    gemini_max_concurrency: int = Field(default=5, ge=1, le=50)
    groq_max_concurrency: int = Field(default=10, ge=1, le=100)
    tavily_max_concurrency: int = Field(default=5, ge=1, le=50)

    # Free-only cost protection
    free_only: bool = True
    allow_paid_models: bool = False
    free_model_allowlist: str = ""
    max_request_tokens: int = Field(default=8192, ge=1)
    tavily_max_monthly_queries: int = Field(default=1000, ge=1)
    tavily_quota_file_path: str = ".tavily_quota.json"

    # Provider rate and daily limits (Free Tier)
    gemini_daily_request_limit: int = Field(default=1500, ge=1)
    gemini_rpm_limit: int = Field(default=15, ge=1)
    groq_daily_request_limit: int = Field(default=14400, ge=1)
    groq_rpm_limit: int = Field(default=30, ge=1)
    cohere_monthly_request_limit: int = Field(default=1000, ge=1)
    mistral_rpm_limit: int = Field(default=60, ge=1)

    # Agent free-tier model assignments (all default to verified free-tier models)
    planner_model: str = "gemini-2.5-flash"
    researcher_model: str = "gemini-2.5-flash-lite"
    analyst_model: str = "gemini-2.5-flash"
    writer_model: str = "gemini-2.5-flash"
    reviewer_model: str = "gemini-2.5-flash"

    # Free-tier fallbacks (strictly verified free models)
    gemini_fallback_model: str = "gemini-2.0-flash"
    groq_fallback_model: str = ""

    # HTTP
    cors_origins: str = "http://localhost:5173"

    # Database pool / timeouts
    db_pool_size: int = Field(default=5, ge=1, le=20)
    db_max_overflow: int = Field(default=5, ge=0, le=20)
    db_pool_timeout: float = Field(default=30.0, gt=0, le=300)
    db_pool_recycle: int = Field(default=1800, ge=60)
    db_connect_timeout: float = Field(default=10.0, gt=0, le=120)
    db_command_timeout: float = Field(default=60.0, gt=0, le=600)
    db_statement_timeout_ms: int = Field(default=60_000, ge=1_000, le=600_000)
    db_idle_in_tx_timeout_ms: int = Field(default=30_000, ge=1_000, le=600_000)

    @field_validator("app_env")
    @classmethod
    def _validate_app_env(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_ENVS:
            allowed = ", ".join(sorted(_ALLOWED_ENVS))
            raise ValueError(f"APP_ENV must be one of: {allowed}")
        return normalized

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in _ALLOWED_LOG_LEVELS:
            allowed = ", ".join(sorted(_ALLOWED_LOG_LEVELS))
            raise ValueError(f"LOG_LEVEL must be one of: {allowed}")
        return normalized

    @field_validator(
        "migration_database_url",
        "gemini_api_key",
        "groq_api_key",
        "tavily_api_key",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("database_url", "migration_database_url")
    @classmethod
    def _validate_database_url(cls, value: str | None) -> str | None:
        if value is None:
            return None

        parsed = urlparse(value)
        if parsed.scheme != "postgresql+asyncpg":
            raise ValueError(
                "database URL must start with postgresql+asyncpg:// "
                "(SQLAlchemy async driver)"
            )
        if not parsed.hostname:
            raise ValueError("database URL must include a hostname")

        query = parse_qs(parsed.query)
        tls_values = query.get("ssl", []) + query.get("sslmode", [])
        tls = (tls_values[0] if tls_values else "").strip().lower()
        host = parsed.hostname.lower()

        if host not in _LOCAL_HOSTS and tls not in _TLS_VALUES:
            raise ValueError(
                "TLS is required for non-local database hosts: add "
                "?ssl=require (or sslmode=require) to DATABASE_URL"
            )
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def providers_configured(self) -> dict[str, bool]:
        return {
            "gemini": bool(self.gemini_api_key),
            "groq": bool(self.groq_api_key),
            "tavily": bool(self.tavily_api_key),
        }

    def redaction_secrets(self) -> tuple[str, ...]:
        """Values that must never appear in logs."""
        secrets: list[str] = []
        for value in (self.gemini_api_key, self.groq_api_key, self.tavily_api_key):
            if value:
                secrets.append(value)
        password = urlparse(self.database_url).password
        if password:
            secrets.append(password)
            decoded = unquote(password)
            if decoded != password:
                secrets.append(decoded)
        # Deduplicate while preserving order.
        return tuple(dict.fromkeys(secrets))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
