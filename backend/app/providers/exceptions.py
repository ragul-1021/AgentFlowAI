"""Provider error hierarchy.

Ensures sensitive values like API keys or authorization headers are never
leaked into logs, traces, or client-facing exceptions.
"""

from __future__ import annotations

from typing import Any


class ProviderError(Exception):
    """Base exception for all AI provider integrations."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        status_code: int | None = None,
        raw_error: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.raw_error = raw_error

    def __str__(self) -> str:
        parts = [f"[{self.provider}] {self.message}"]
        if self.status_code:
            parts.append(f"(status: {self.status_code})")
        if self.model:
            parts.append(f"(model: {self.model})")
        return " ".join(parts)


class ProviderAuthError(ProviderError):
    """Raised on HTTP 401 or 403 authentication / permission failure. Not retryable."""


class ProviderRateLimitError(ProviderError):
    """Raised on HTTP 429 rate limit or quota exhaustion. Retryable."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        status_code: int = 429,
        retry_after: float | None = None,
        raw_error: Any = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
            raw_error=raw_error,
        )
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Raised when an outbound request to a provider times out. Retryable."""


class ProviderUnavailableError(ProviderError):
    """Raised on HTTP 500, 502, 503, 504 temporary server errors. Retryable."""


class ProviderBadRequestError(ProviderError):
    """Raised on HTTP 400 invalid request parameter. Not retryable."""


class StructuredOutputError(ProviderError):
    """Raised when structured generation fails to parse or validate schema. Not retryable."""


class PaidModelBlockedError(ProviderError):
    """Raised when a paid model is requested while free_only is enabled. Never retryable."""


class QuotaExceededError(ProviderError):
    """Raised when daily or monthly query/token budget is exceeded. Never retryable."""


class AllFreeTiersExhaustedError(ProviderError):
    """Raised when all free-tier providers/fallbacks are rate-limited. Never upgrade to paid."""

    def __init__(
        self,
        message: str = "All free-tier providers have reached their rate limits. Please try again later.",
        *,
        provider: str = "all",
        model: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model, status_code=429)
        self.retry_after = retry_after
