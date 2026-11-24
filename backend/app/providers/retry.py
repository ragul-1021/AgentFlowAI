"""Reliable retry wrapper with exponential backoff, jitter, and Retry-After support."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from app.providers.exceptions import (
    AllFreeTiersExhaustedError,
    PaidModelBlockedError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    StructuredOutputError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Return True only for transient errors that are safe to retry."""
    if isinstance(
        exc,
        (
            ProviderRateLimitError,
            ProviderTimeoutError,
            ProviderUnavailableError,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return True

    # Never retry auth, bad request, schema errors, or cost protection blocks
    if isinstance(
        exc,
        (
            ProviderAuthError,
            ProviderBadRequestError,
            StructuredOutputError,
            PaidModelBlockedError,
            QuotaExceededError,
            AllFreeTiersExhaustedError,
        ),
    ):
        return False

    return False


def calculate_delay(
    attempt: int,
    base_delay: float = 1.0,
    retry_after: float | None = None,
    max_delay: float = 60.0,
) -> float:
    """Calculate sleep time in seconds using exponential backoff with full jitter."""
    if retry_after is not None and retry_after > 0:
        return min(float(retry_after), max_delay)

    # Exponential backoff: base * 2^(attempt - 1)
    exponential = base_delay * (2 ** (attempt - 1))
    # Full jitter: random between 0 and exponential, plus small base
    jittered = random.uniform(0.1, exponential)
    return min(jittered, max_delay)


async def execute_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    provider: str,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Run an async provider operation, retrying transient errors with backoff."""
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except BaseException as exc:
            last_exc = exc
            if not is_retryable_provider_error(exc) or attempt >= max_attempts:
                raise

            retry_after = getattr(exc, "retry_after", None)
            delay = calculate_delay(attempt, base_delay=base_delay, retry_after=retry_after)

            logger.warning(
                "[%s] Transient error on attempt %d/%d (%s). Retrying in %.2fs...",
                provider,
                attempt,
                max_attempts,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise ProviderError(f"Operation failed after {max_attempts} attempts", provider=provider)
