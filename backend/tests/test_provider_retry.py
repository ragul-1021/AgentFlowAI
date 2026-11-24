"""Unit tests for provider retry, jitter, rate-limit, and timeout handling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import httpx

from app.providers import (
    ChatMessage,
    ConcurrencyLimiter,
    GeminiProvider,
    LLMRequest,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StructuredOutputError,
    execute_with_retry,
    is_retryable_provider_error,
)
from app.providers.retry import calculate_delay


def test_is_retryable_provider_error() -> None:
    assert is_retryable_provider_error(ProviderRateLimitError("429", provider="test"))
    assert is_retryable_provider_error(ProviderUnavailableError("503", provider="test"))
    assert is_retryable_provider_error(ProviderTimeoutError("timeout", provider="test"))
    assert is_retryable_provider_error(asyncio.TimeoutError())

    assert not is_retryable_provider_error(ProviderAuthError("401", provider="test"))
    assert not is_retryable_provider_error(ProviderBadRequestError("400", provider="test"))
    assert not is_retryable_provider_error(StructuredOutputError("bad json", provider="test"))
    assert not is_retryable_provider_error(ValueError("logic error"))


def test_calculate_delay_respects_retry_after() -> None:
    delay = calculate_delay(attempt=1, base_delay=1.0, retry_after=5.5)
    assert delay == 5.5

    # Capped at max_delay
    delay_capped = calculate_delay(attempt=1, base_delay=1.0, retry_after=120.0, max_delay=60.0)
    assert delay_capped == 60.0


def test_calculate_delay_exponential_with_jitter() -> None:
    for attempt in range(1, 4):
        delay = calculate_delay(attempt=attempt, base_delay=1.0)
        assert 0.1 <= delay <= (1.0 * (2 ** (attempt - 1)))


@pytest.mark.asyncio
async def test_retry_on_429_rate_limit_succeeds() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderRateLimitError("Rate limit exceeded", provider="test", retry_after=0.01)
        return "success"

    result = await execute_with_retry(flaky, provider="test", max_attempts=3, base_delay=0.01)
    assert result == "success"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_respects_retry_after_header() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0.05"},
                json={"error": {"message": "Rate limit exceeded"}},
            )
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider("fake-key", client=client, base_delay=0.01)

    res = await provider.generate_text(LLMRequest(messages=[ChatMessage(role="user", content="hi")]))
    assert res.content == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_on_5xx_unavailable_succeeds() -> None:
    attempts = 0

    async def server_error() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ProviderUnavailableError("503 Service Unavailable", provider="test")
        return "recovered"

    result = await execute_with_retry(server_error, provider="test", max_attempts=3, base_delay=0.01)
    assert result == "recovered"
    assert attempts == 2


@pytest.mark.asyncio
async def test_timeout_retried_and_succeeds() -> None:
    attempts = 0

    async def timing_out() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProviderTimeoutError("Read timeout", provider="test")
        return "finished"

    result = await execute_with_retry(timing_out, provider="test", max_attempts=3, base_delay=0.01)
    assert result == "finished"
    assert attempts == 2


@pytest.mark.asyncio
async def test_auth_failure_not_retried() -> None:
    attempts = 0

    async def auth_fail() -> None:
        nonlocal attempts
        attempts += 1
        raise ProviderAuthError("Invalid credentials", provider="test")

    with pytest.raises(ProviderAuthError):
        await execute_with_retry(auth_fail, provider="test", max_attempts=3, base_delay=0.01)
    assert attempts == 1


@pytest.mark.asyncio
async def test_max_retries_exhaustion_raises() -> None:
    attempts = 0

    async def always_fails() -> None:
        nonlocal attempts
        attempts += 1
        raise ProviderUnavailableError("Down", provider="test")

    with pytest.raises(ProviderUnavailableError):
        await execute_with_retry(always_fails, provider="test", max_attempts=3, base_delay=0.01)
    assert attempts == 3


@pytest.mark.asyncio
async def test_concurrency_limiter_bounds() -> None:
    limiter = ConcurrencyLimiter({"groq": 2})
    active = 0
    max_active = 0

    async def worker() -> None:
        nonlocal active, max_active
        async with limiter.acquire("groq"):
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(*(worker() for _ in range(6)))
    assert max_active <= 2
