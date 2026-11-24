"""Provider factory for constructing live or mock AI providers based on configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import httpx

from app.config import Settings
from app.providers.base import BaseLLMProvider, BaseSearchProvider
from app.providers.fallback import FreeTierFallbackRouter
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.mock import MockGeminiProvider, MockGroqProvider, MockTavilyProvider
from app.providers.ratelimit import ConcurrencyLimiter
from app.providers.tavily import TavilyProvider


@lru_cache(maxsize=1)
def get_concurrency_limiter(
    gemini_max: int = 5, groq_max: int = 10, tavily_max: int = 5
) -> ConcurrencyLimiter:
    """Singleton concurrency limiter enforcing per-provider rate bounds."""
    return ConcurrencyLimiter(
        {
            "gemini": gemini_max,
            "groq": groq_max,
            "tavily": tavily_max,
        }
    )


def create_gemini_provider(
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
    limiter: ConcurrencyLimiter | None = None,
) -> BaseLLMProvider:
    """Build a Gemini provider or return a mock instance if mock mode is active."""
    if settings.ai_mock_mode or not settings.gemini_api_key:
        return MockGeminiProvider(
            default_model=settings.gemini_model or "gemini-2.5-flash",
            free_only=settings.free_only,
            allow_paid_models=settings.allow_paid_models,
            custom_allowlist=settings.free_model_allowlist,
            max_request_tokens=settings.max_request_tokens,
        )

    active_limiter = limiter or get_concurrency_limiter(
        settings.gemini_max_concurrency,
        settings.groq_max_concurrency,
        settings.tavily_max_concurrency,
    )
    return GeminiProvider(
        api_key=settings.gemini_api_key,
        default_model=settings.gemini_model or "gemini-2.5-flash",
        client=client,
        timeout=settings.gemini_timeout,
        max_attempts=settings.retry_max_attempts,
        base_delay=settings.retry_base_delay,
        limiter=active_limiter,
        free_only=settings.free_only,
        allow_paid_models=settings.allow_paid_models,
        custom_allowlist=settings.free_model_allowlist,
        max_request_tokens=settings.max_request_tokens,
    )


def create_groq_provider(
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
    limiter: ConcurrencyLimiter | None = None,
) -> BaseLLMProvider:
    """Build a Groq provider or return a mock instance if mock mode is active."""
    if settings.ai_mock_mode or not settings.groq_api_key:
        return MockGroqProvider(
            default_model=settings.groq_model or "mock-groq",
            free_only=settings.free_only,
            allow_paid_models=settings.allow_paid_models,
            custom_allowlist=settings.free_model_allowlist or "mock-groq",
            max_request_tokens=settings.max_request_tokens,
        )

    active_limiter = limiter or get_concurrency_limiter(
        settings.gemini_max_concurrency,
        settings.groq_max_concurrency,
        settings.tavily_max_concurrency,
    )
    return GroqProvider(
        api_key=settings.groq_api_key,
        default_model=settings.groq_model or "",
        client=client,
        timeout=settings.groq_timeout,
        max_attempts=settings.retry_max_attempts,
        base_delay=settings.retry_base_delay,
        limiter=active_limiter,
        free_only=settings.free_only,
        allow_paid_models=settings.allow_paid_models,
        custom_allowlist=settings.free_model_allowlist,
        max_request_tokens=settings.max_request_tokens,
    )


def create_tavily_provider(
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
    limiter: ConcurrencyLimiter | None = None,
) -> BaseSearchProvider:
    """Build a Tavily provider or return a mock instance if mock mode is active."""
    if settings.ai_mock_mode or not settings.tavily_api_key:
        return MockTavilyProvider(monthly_budget=settings.tavily_max_monthly_queries)

    active_limiter = limiter or get_concurrency_limiter(
        settings.gemini_max_concurrency,
        settings.groq_max_concurrency,
        settings.tavily_max_concurrency,
    )
    return TavilyProvider(
        api_key=settings.tavily_api_key,
        client=client,
        timeout=settings.tavily_timeout,
        max_attempts=settings.retry_max_attempts,
        base_delay=settings.retry_base_delay,
        limiter=active_limiter,
        monthly_budget=settings.tavily_max_monthly_queries,
        quota_storage_path=settings.tavily_quota_file_path,
    )


def create_gemini_fallback_router(
    settings: Settings,
) -> FreeTierFallbackRouter:
    """Build a fallback router for Gemini free-tier models."""
    return FreeTierFallbackRouter(
        primary_model=settings.gemini_model or "gemini-2.5-flash",
        fallback_models=[settings.gemini_fallback_model or "gemini-2.0-flash"],
        provider_name="gemini",
        free_only=settings.free_only,
        allow_paid_models=settings.allow_paid_models,
        custom_allowlist=settings.free_model_allowlist,
    )


def create_groq_fallback_router(
    settings: Settings,
) -> FreeTierFallbackRouter:
    """Build a fallback router for Groq free-tier models."""
    primary = settings.groq_model or ""
    fallbacks = [settings.groq_fallback_model] if settings.groq_fallback_model else []
    return FreeTierFallbackRouter(
        primary_model=primary,
        fallback_models=fallbacks,
        provider_name="groq",
        free_only=settings.free_only,
        allow_paid_models=settings.allow_paid_models,
        custom_allowlist=settings.free_model_allowlist,
    )
