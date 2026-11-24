"""Automated tests for free-only API enforcement, access tiers, and cost protection guardrails."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import httpx

from app.providers import (
    AccessTier,
    AllFreeTiersExhaustedError,
    ChatMessage,
    DailyRequestQuotaTracker,
    ExtractRequest,
    FreeTierFallbackRouter,
    GeminiProvider,
    GroqProvider,
    LLMRequest,
    MockGeminiProvider,
    MockGroqProvider,
    MockTavilyProvider,
    PaidModelBlockedError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    QuotaExceededError,
    SearchQuotaTracker,
    SearchRequest,
    TavilyProvider,
    get_model_access_tier,
    get_model_metadata,
    is_free_tier_model,
    validate_model_allowed,
    validate_token_limit,
)


def test_paid_models_blocked_by_default() -> None:
    paid_candidates = [
        "gpt-4o",
        "gpt-4-turbo",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus",
        "meta-llama/llama-3.1-405b",
        "gemini-1.5-pro-paid",
        "random-unregistered-model",
    ]
    for model in paid_candidates:
        with pytest.raises(PaidModelBlockedError):
            validate_model_allowed(model, provider="test", free_only=True, allow_paid_models=False)


def test_priced_and_deprecated_groq_models_blocked() -> None:
    """Verify that priced Groq models (openai/gpt-oss-120b) and deprecated models are blocked by default."""
    blocked_candidates = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "llama-3.3-70b-versatile",
        "llama-3.3-70b-specdec",
        "mixtral-8x7b-32768",
    ]
    for model in blocked_candidates:
        assert not is_free_tier_model(model)
        with pytest.raises(PaidModelBlockedError):
            validate_model_allowed(model, provider="groq", free_only=True, allow_paid_models=False)


def test_missing_or_empty_model_fails_safely() -> None:
    """Requirement: Make missing or unverified model configuration fail safely with a clear error."""
    for empty_val in ["", "   ", None]:
        with pytest.raises(PaidModelBlockedError, match="No model configured"):
            validate_model_allowed(empty_val, provider="groq", free_only=True)
        with pytest.raises(PaidModelBlockedError, match="No model configured"):
            validate_model_allowed(empty_val, provider="gemini", free_only=True)


def test_verified_free_models_allowed() -> None:
    free_candidates = [
        # Gemini verified free tier
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite",
        # Cloudflare verified free allowance models
        "@cf/meta/llama-3.2-3b-instruct",
        "@cf/meta/llama-3.2-1b-instruct",
        "@cf/mistral/mistral-7b-instruct-v0.1",
        # Mistral free experimentation
        "mistral-small-latest",
        "open-mistral-7b",
        "open-mistral-nemo",
        # Cohere trial
        "command-r",
        "command-r-plus",
        "command-light",
    ]
    for model in free_candidates:
        validate_model_allowed(model, provider="test", free_only=True, allow_paid_models=False)
        assert is_free_tier_model(model)


def test_cloudflare_namespace_not_blanket_free() -> None:
    """Requirement: Do not classify an entire provider namespace as free.

    Only explicitly verified Cloudflare models are free; arbitrary @cf/ models must be rejected.
    """
    # Explicitly verified models pass
    assert is_free_tier_model("@cf/meta/llama-3.2-3b-instruct")
    assert is_free_tier_model("@cf/meta/llama-3.2-1b-instruct")

    # Unverified models in @cf/ namespace MUST fail
    unverified_cf = [
        "@cf/meta/llama-3.1-405b",
        "@cf/meta/llama-3.1-70b-instruct",
        "@cf/anthropic/claude-something",
        "@cf/arbitrary/paid-model",
    ]
    for model in unverified_cf:
        assert not is_free_tier_model(model)
        with pytest.raises(PaidModelBlockedError):
            validate_model_allowed(model, provider="cloudflare", free_only=True)


def test_model_access_tiers_distinguished() -> None:
    """Requirement: Distinguish permanent free, rate-limited free-tier, and temporary trial."""
    # Permanent Free (OpenRouter :free endpoints)
    assert get_model_access_tier("meta-llama/llama-3.3-70b-instruct:free") == AccessTier.PERMANENT_FREE
    assert get_model_access_tier("google/gemini-2.0-flash-exp:free") == AccessTier.PERMANENT_FREE

    # Free Tier Rate Limited (Gemini, CF free allowance)
    assert get_model_access_tier("gemini-2.5-flash") == AccessTier.FREE_TIER_RATE_LIMITED
    assert get_model_access_tier("gemini-2.0-flash") == AccessTier.FREE_TIER_RATE_LIMITED
    assert get_model_access_tier("@cf/meta/llama-3.2-3b-instruct") == AccessTier.FREE_TIER_RATE_LIMITED

    # Temporary Trial (Cohere trial, Mistral experimentation)
    assert get_model_access_tier("command-r-plus") == AccessTier.TEMPORARY_TRIAL
    assert get_model_access_tier("command-r") == AccessTier.TEMPORARY_TRIAL
    assert get_model_access_tier("mistral-small-latest") == AccessTier.TEMPORARY_TRIAL
    assert get_model_access_tier("open-mistral-7b") == AccessTier.TEMPORARY_TRIAL

    # Unknown / Priced / Unverified returns None
    assert get_model_access_tier("gpt-4o") is None
    assert get_model_access_tier("openai/gpt-oss-120b") is None
    assert get_model_access_tier("llama-3.1-8b-instant") is None


def test_model_metadata_inspection() -> None:
    meta = get_model_metadata("gemini-2.5-flash")
    assert meta is not None
    assert meta.provider == "gemini"
    assert meta.access_tier == AccessTier.FREE_TIER_RATE_LIMITED
    assert meta.rpm_limit == 15
    assert meta.rpd_limit == 1500

    lite_meta = get_model_metadata("gemini-2.5-flash-lite")
    assert lite_meta is not None
    assert lite_meta.provider == "gemini"
    assert lite_meta.rpm_limit == 15


def test_openrouter_free_suffix_allowed() -> None:
    assert is_free_tier_model("meta-llama/llama-3.3-70b-instruct:free")
    validate_model_allowed(
        "meta-llama/llama-3.3-70b-instruct:free",
        provider="openrouter",
        free_only=True,
    )

    # Models without :free suffix are blocked
    assert not is_free_tier_model("meta-llama/llama-3.3-70b-instruct")
    with pytest.raises(PaidModelBlockedError):
        validate_model_allowed(
            "meta-llama/llama-3.3-70b-instruct",
            provider="openrouter",
            free_only=True,
        )


def test_custom_allowlist_expansion() -> None:
    custom_model = "my-custom-internal-free-model"
    with pytest.raises(PaidModelBlockedError):
        validate_model_allowed(custom_model, provider="custom", free_only=True)

    # Allowed when added to custom_allowlist
    validate_model_allowed(
        custom_model,
        provider="custom",
        free_only=True,
        custom_allowlist=f"other-model, {custom_model}",
    )


@pytest.mark.asyncio
async def test_gemini_provider_rejects_paid_or_unknown_model() -> None:
    provider = GeminiProvider("fake-key", default_model="gemini-2.5-flash", free_only=True)
    with pytest.raises(PaidModelBlockedError, match="Cost protection is active"):
        await provider.generate_text(
            LLMRequest(messages=[ChatMessage(role="user", content="hi")]),
            model="claude-3-opus",
        )


@pytest.mark.asyncio
async def test_groq_provider_rejects_deprecated_or_paid_model() -> None:
    provider = GroqProvider("fake-key", default_model="openai/gpt-oss-120b", free_only=True)
    with pytest.raises(PaidModelBlockedError, match="Cost protection is active"):
        await provider.generate_text(
            LLMRequest(messages=[ChatMessage(role="user", content="hi")]),
            model="llama-3.1-8b-instant",  # Deprecated!
        )


@pytest.mark.asyncio
async def test_mock_provider_rejects_paid_model() -> None:
    mock_gemini = MockGeminiProvider(free_only=True)
    with pytest.raises(PaidModelBlockedError):
        await mock_gemini.generate_text(
            LLMRequest(messages=[ChatMessage(role="user", content="hi")]),
            model="gpt-4-turbo",
        )

    mock_groq = MockGroqProvider(free_only=True)
    with pytest.raises(PaidModelBlockedError):
        await mock_groq.generate_text(
            LLMRequest(messages=[ChatMessage(role="user", content="hi")]),
            model="claude-3-5-sonnet",
        )


def test_token_limit_enforcement() -> None:
    validate_token_limit(max_tokens=4096, max_request_tokens=8192)

    with pytest.raises(QuotaExceededError, match="exceeds the allowed request limit"):
        validate_token_limit(max_tokens=16384, max_request_tokens=8192)


def test_tavily_search_quota_persistence_and_restart(tmp_path: Path) -> None:
    """Requirement: Ensure Tavily quota tracking cannot reset unexpectedly when backend restarts."""
    quota_file = tmp_path / ".tavily_quota.json"

    # 1. First process run: consume 3 queries
    tracker1 = SearchQuotaTracker(monthly_budget=5, persistence_path=quota_file)
    tracker1.reset()
    tracker1.check_and_increment(3)
    assert tracker1.consumed == 3
    assert tracker1.remaining == 2
    assert quota_file.exists()

    # 2. Simulate backend restart: create a new tracker instance pointing to same file
    tracker2 = SearchQuotaTracker(monthly_budget=5, persistence_path=quota_file)
    assert tracker2.consumed == 3, "Consumed count must persist across backend restarts!"
    assert tracker2.remaining == 2

    # 3. Consume 2 more queries (reaching limit of 5)
    tracker2.check_and_increment(2)
    assert tracker2.consumed == 5

    # 4. 6th query must raise QuotaExceededError
    with pytest.raises(QuotaExceededError, match="monthly query budget"):
        tracker2.check_and_increment(1)

    # 5. Simulate monthly reset
    tracker2.reset()
    assert tracker2.consumed == 0

    # 6. Verify reset was persisted to disk for next restart
    tracker3 = SearchQuotaTracker(monthly_budget=5, persistence_path=quota_file)
    assert tracker3.consumed == 0


def test_daily_request_quota_tracker() -> None:
    """Requirement: Enforce daily limits where providers enforce them."""
    tracker = DailyRequestQuotaTracker({"gemini": 3, "groq": 5})

    tracker.check_and_increment("gemini", 2)
    assert tracker.get_count("gemini") == 2

    tracker.check_and_increment("gemini", 1)
    assert tracker.get_count("gemini") == 3

    # Exceeding gemini daily limit of 3 raises QuotaExceededError
    with pytest.raises(QuotaExceededError, match="Daily request limit"):
        tracker.check_and_increment("gemini", 1)

    # Groq quota is independent
    tracker.check_and_increment("groq", 4)
    assert tracker.get_count("groq") == 4


@pytest.mark.asyncio
async def test_tavily_provider_enforces_quota(tmp_path: Path) -> None:
    quota_file = tmp_path / "tavily_test.json"
    tracker = SearchQuotaTracker(monthly_budget=1, persistence_path=quota_file)
    tracker.reset()

    provider = TavilyProvider("fake-key", quota_tracker=tracker)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    # First query succeeds
    await provider.search(SearchRequest(query="test 1"))
    assert tracker.consumed == 1

    # Second query is blocked before network request
    with pytest.raises(QuotaExceededError, match="monthly query budget"):
        await provider.search(SearchRequest(query="test 2"))


@pytest.mark.asyncio
async def test_free_tier_fallback_router_success_primary() -> None:
    router = FreeTierFallbackRouter(
        primary_model="gemini-2.5-flash",
        fallback_models=["gemini-2.0-flash"],
        provider_name="gemini",
        free_only=True,
    )

    async def operation(model: str) -> str:
        return f"result from {model}"

    result = await router.execute(operation)
    assert result == "result from gemini-2.5-flash"


@pytest.mark.asyncio
async def test_free_tier_fallback_router_routes_to_free_fallback_on_429() -> None:
    router = FreeTierFallbackRouter(
        primary_model="gemini-2.5-flash",
        fallback_models=["gemini-2.0-flash"],
        provider_name="gemini",
        free_only=True,
    )

    calls: list[str] = []

    async def operation(model: str) -> str:
        calls.append(model)
        if model == "gemini-2.5-flash":
            raise ProviderRateLimitError("Rate limit reached for 2.5 flash", provider="gemini")
        return f"success from {model}"

    result = await router.execute(operation)
    assert result == "success from gemini-2.0-flash"
    assert calls == ["gemini-2.5-flash", "gemini-2.0-flash"]


@pytest.mark.asyncio
async def test_free_tier_fallback_router_exhaustion_never_upgrades() -> None:
    router = FreeTierFallbackRouter(
        primary_model="gemini-2.5-flash",
        fallback_models=["gemini-2.0-flash"],
        provider_name="gemini",
        free_only=True,
    )

    async def always_rate_limit(model: str) -> str:
        raise ProviderRateLimitError(f"{model} quota exhausted", provider="gemini", retry_after=30.0)

    with pytest.raises(AllFreeTiersExhaustedError) as exc_info:
        await router.execute(always_rate_limit)

    assert "All free-tier models for [gemini] reached their limits" in str(exc_info.value)
    assert exc_info.value.retry_after == 30.0


def test_fallback_router_blocks_paid_model_in_chain() -> None:
    # Attempting to configure a paid model in the fallback chain fails fast at construction
    with pytest.raises(PaidModelBlockedError):
        FreeTierFallbackRouter(
            primary_model="gemini-2.5-flash",
            fallback_models=["gpt-4o"],  # Paid model!
            provider_name="gemini",
            free_only=True,
        )


def test_fallback_router_blocks_deprecated_or_priced_groq_model() -> None:
    # Attempting to configure priced or unverified models fails fast
    with pytest.raises(PaidModelBlockedError):
        FreeTierFallbackRouter(
            primary_model="gemini-2.5-flash",
            fallback_models=["openai/gpt-oss-120b"],  # Priced model!
            provider_name="gemini",
            free_only=True,
        )
