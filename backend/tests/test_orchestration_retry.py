"""Tests for pipeline retry policies, transient errors, and immediate failures."""

from __future__ import annotations

import pytest

from app.agents.exceptions import AgentProviderError, AgentValidationError
from app.orchestration import (
    PipelineCancelledError,
    execute_with_pipeline_retry,
    is_retryable_pipeline_error,
)
from app.providers import (
    PaidModelBlockedError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from pydantic import ValidationError


def test_fatal_errors_are_never_retryable():
    fatal_errors = [
        PaidModelBlockedError("Paid model blocked", provider="gemini", model="gpt-4o"),
        QuotaExceededError("Monthly quota exceeded", provider="tavily"),
        AgentValidationError("Malformed plan output"),
        PipelineCancelledError("Task was cancelled"),
    ]
    for err in fatal_errors:
        assert is_retryable_pipeline_error(err) is False


def test_transient_provider_errors_are_retryable():
    rate_limit = ProviderRateLimitError("429 Too Many Requests", provider="gemini", retry_after=5.0)
    assert is_retryable_pipeline_error(rate_limit) is True

    unavailable = ProviderUnavailableError("503 Service Unavailable", provider="gemini")
    assert is_retryable_pipeline_error(unavailable) is True

    wrapped_rate_limit = AgentProviderError("Rate limit", status_code=429)
    assert is_retryable_pipeline_error(wrapped_rate_limit) is True

    wrapped_fatal = AgentProviderError("Auth failure", status_code=401)
    assert is_retryable_pipeline_error(wrapped_fatal) is False


@pytest.mark.asyncio
async def test_execute_with_retry_succeeds_after_transient_failures():
    calls = 0

    async def transient_op():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ProviderRateLimitError("Rate limit hit", provider="gemini", retry_after=0.01)
        return "success"

    result = await execute_with_pipeline_retry(
        transient_op,
        stage_name="test_stage",
        max_attempts=3,
        base_delay=0.01,
    )
    assert result == "success"
    assert calls == 2


@pytest.mark.asyncio
async def test_execute_with_retry_fails_immediately_on_fatal_error():
    calls = 0

    async def fatal_op():
        nonlocal calls
        calls += 1
        raise PaidModelBlockedError("Paid model blocked", provider="gemini", model="gpt-4o")

    with pytest.raises(PaidModelBlockedError):
        await execute_with_pipeline_retry(
            fatal_op,
            stage_name="test_stage",
            max_attempts=3,
            base_delay=0.01,
        )

    # Must NOT retry
    assert calls == 1
