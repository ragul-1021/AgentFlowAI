from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

LOCAL_URL = "postgresql+asyncpg://user:pass@localhost:5432/agentflow?ssl=require"
REMOTE_URL = "postgresql+asyncpg://user:pass@db.example.com:5432/agentflow?ssl=require"


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"database_url": LOCAL_URL, **overrides}
    return Settings(_env_file=None, **values)


def test_accepts_asyncpg_database_url() -> None:
    settings = make_settings()
    assert settings.database_url == LOCAL_URL


def test_rejects_missing_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_sqlite_url() -> None:
    with pytest.raises(ValidationError):
        make_settings(database_url="sqlite+aiosqlite:///./agentflow.db")


def test_rejects_sync_postgres_url() -> None:
    with pytest.raises(ValidationError):
        make_settings(database_url="postgresql://user:pass@db.example.com:5432/postgres")


def test_requires_tls_for_remote_hosts() -> None:
    with pytest.raises(ValidationError):
        make_settings(
            database_url="postgresql+asyncpg://user:pass@db.example.com:5432/postgres"
        )


def test_accepts_sslmode_parameter() -> None:
    settings = make_settings(
        database_url="postgresql+asyncpg://user:pass@db.example.com:5432/postgres?sslmode=require"
    )
    assert "sslmode=require" in settings.database_url


def test_allows_localhost_without_tls() -> None:
    settings = make_settings(
        database_url="postgresql+asyncpg://user:pass@localhost:5432/agentflow"
    )
    assert settings.database_url.endswith("/agentflow")


def test_blank_optional_values_become_none() -> None:
    settings = make_settings(
        migration_database_url="  ",
        gemini_api_key="  ",
        groq_api_key="",
    )
    assert settings.migration_database_url is None
    assert settings.gemini_api_key is None
    assert settings.groq_api_key is None


def test_provider_flags() -> None:
    settings = make_settings(
        gemini_api_key="gemini-key",
        groq_api_key="",
        tavily_api_key="tavily-key",
    )
    assert settings.providers_configured == {
        "gemini": True,
        "groq": False,
        "tavily": True,
    }


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = make_settings(
        cors_origins="http://localhost:5173, https://app.example.com ,"
    )
    assert settings.cors_origin_list == [
        "http://localhost:5173",
        "https://app.example.com",
    ]


def test_pool_bounds_are_validated() -> None:
    with pytest.raises(ValidationError):
        make_settings(db_pool_size=0)
    with pytest.raises(ValidationError):
        make_settings(db_max_overflow=-1)
    with pytest.raises(ValidationError):
        make_settings(db_pool_timeout=0)


def test_log_level_is_normalized_and_validated() -> None:
    assert make_settings(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        make_settings(log_level="verbose")


def test_remote_database_url_is_valid() -> None:
    settings = make_settings(database_url=REMOTE_URL)
    assert settings.database_url == REMOTE_URL


def test_redaction_secrets_include_database_password() -> None:
    settings = make_settings(
        database_url="postgresql+asyncpg://user:p%40ss@localhost:5432/agentflow",
        gemini_api_key="gemini-key",
    )
    secrets = settings.redaction_secrets()
    assert "gemini-key" in secrets
    assert "p%40ss" in secrets
    assert "p@ss" in secrets
