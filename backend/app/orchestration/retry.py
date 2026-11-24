"""Retry handling for transient pipeline, provider, and database failures."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging
import random
import time
from typing import TypeVar

from app.agents.exceptions import AgentProviderError, AgentValidationError
from app.core.redaction import redact
from app.db import is_transient_db_error
from app.orchestration.exceptions import PipelineCancelledError
from app.providers.exceptions import (
    AllFreeTiersExhaustedError,
    PaidModelBlockedError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Non-retryable error types that must fail immediately
FATAL_ERROR_TYPES = (
    PaidModelBlockedError,
    QuotaExceededError,
    AllFreeTiersExhaustedError,
    AgentValidationError,
    ValidationError,
    PipelineCancelledError,
)


def is_retryable_pipeline_error(exc: BaseException) -> bool:
    """Return True only if the failure is transient and safe to retry."""
    if isinstance(exc, FATAL_ERROR_TYPES):
        return False

    if isinstance(exc, (ProviderRateLimitError, ProviderUnavailableError)):
        return True

    if isinstance(exc, AgentProviderError):
        # Retry only if wrapped error was a rate limit or 5xx status code
        if exc.status_code and exc.status_code in (429, 502, 503, 504):
            return True
        if exc.retry_after is not None:
            return True
        return False

    if isinstance(exc, SQLAlchemyError) and is_transient_db_error(exc):
        return True

    return False


async def execute_with_pipeline_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    stage_name: str,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    secrets: tuple[str, ...] = (),
) -> T:
    """Execute an asynchronous operation with jittered backoff on transient errors."""
    last_exception: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except BaseException as exc:
            last_exception = exc
            if not is_retryable_pipeline_error(exc) or attempt >= max_attempts:
                safe_msg = redact(str(exc), secrets)
                logger.error(
                    "[%s] Non-retryable or exhausted failure on attempt %d/%d: %s",
                    stage_name,
                    attempt,
                    max_attempts,
                    safe_msg,
                )
                raise

            # Calculate delay, respecting Retry-After header if provided
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None and retry_after > 0:
                delay = min(float(retry_after), max_delay)
            else:
                backoff = base_delay * (2 ** (attempt - 1))
                delay = min(backoff + random.uniform(0.1, 0.5), max_delay)

            safe_msg = redact(str(exc), secrets)
            logger.warning(
                "[%s] Transient failure on attempt %d/%d: %s. Retrying in %.2fs...",
                stage_name,
                attempt,
                max_attempts,
                safe_msg,
                delay,
            )
            await asyncio.sleep(delay)

    raise last_exception or RuntimeError(f"Retry loop exhausted for {stage_name}")
