from __future__ import annotations

import logging

from app.core.logging import RedactionFilter
from app.core.redaction import redact, redact_url

DATABASE_URL = (
    "postgresql+asyncpg://postgres.ref:p%40ss@aws-0-eu.pooler.supabase.com:5432"
    "/postgres?ssl=require"
)


def test_redact_url_removes_credentials() -> None:
    result = redact_url(DATABASE_URL)
    assert "p%40ss" not in result
    assert "postgres.ref" not in result
    assert "aws-0-eu.pooler.supabase.com:5432" in result
    assert result.endswith("?ssl=require")


def test_redact_url_without_credentials_is_unchanged() -> None:
    url = "postgresql+asyncpg://localhost:5432/agentflow"
    assert redact_url(url) == url


def test_redact_literal_secret() -> None:
    result = redact("provider key is supersecret-value", ["supersecret-value"])
    assert "supersecret-value" not in result
    assert "[redacted]" in result


def test_redact_key_value_pairs() -> None:
    result = redact("api_key=abc123 password=hunter2 token: xyz789")
    assert "abc123" not in result
    assert "hunter2" not in result
    assert "xyz789" not in result
    assert "api_key=[redacted]" in result
    assert "password=[redacted]" in result


def test_redact_bearer_tokens() -> None:
    result = redact("Authorization: Bearer eyJhbGciOi.payload.signature")
    assert "eyJhbGciOi" not in result
    assert "Bearer [redacted]" in result


def test_redact_provider_key_formats() -> None:
    google = "AIza" + "A" * 35
    groq = "gsk_" + "b" * 40
    tavily = "tvly-" + "c" * 20
    result = redact(f"{google} {groq} {tavily}")
    assert google not in result
    assert groq not in result
    assert tavily not in result


def test_redact_database_url_embedded_in_text() -> None:
    result = redact(f"connecting with {DATABASE_URL}")
    assert "p%40ss" not in result
    assert "postgres.ref" not in result


def test_logging_filter_redacts_message() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="password=hunter2 api_key=abc123",
        args=(),
        exc_info=None,
    )
    RedactionFilter().filter(record)
    assert "hunter2" not in record.getMessage()
    assert "abc123" not in record.getMessage()
    assert "password=[redacted]" in record.getMessage()
