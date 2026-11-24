"""Unit tests for token usage calculation, pricing catalog, and audit tracking."""

from __future__ import annotations

from decimal import Decimal

from app.core.tokens import (
    DEFAULT_PRICING,
    ModelPricing,
    TokenUsage,
    UsageRecord,
    calculate_cost,
    get_model_pricing,
)


def test_token_usage_addition() -> None:
    usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_estimate=Decimal("0.000100"))
    usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300, cost_estimate=Decimal("0.000250"))

    total = usage1 + usage2
    assert total.prompt_tokens == 300
    assert total.completion_tokens == 150
    assert total.total_tokens == 450
    assert total.cost_estimate == Decimal("0.000350")


def test_pricing_catalog_gemini() -> None:
    pricing = get_model_pricing("gemini-2.5-flash")
    assert pricing.prompt_price_per_m == Decimal("0.15")
    assert pricing.completion_price_per_m == Decimal("0.60")

    # Prefix match
    pricing_prefix = get_model_pricing("models/gemini-1.5-pro")
    assert pricing_prefix.prompt_price_per_m == Decimal("1.25")


def test_pricing_catalog_groq() -> None:
    pricing = get_model_pricing("openai/gpt-oss-120b")
    assert pricing.prompt_price_per_m == Decimal("0.59")
    assert pricing.completion_price_per_m == Decimal("0.79")


def test_calculate_cost_zero_tokens() -> None:
    cost = calculate_cost("gemini-2.5-flash", 0, 0)
    assert cost == Decimal("0.000000")


def test_calculate_cost_exact_values() -> None:
    # 1,000,000 prompt tokens of gemini-2.5-flash ($0.15) + 1,000,000 completion ($0.60) = $0.75
    cost = calculate_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
    assert cost == Decimal("0.750000")

    # 1,000 prompt tokens of gemini-2.5-flash = 0.00015
    cost_small = calculate_cost("gemini-2.5-flash", 1_000, 0)
    assert cost_small == Decimal("0.000150")


def test_calculate_cost_fallback_model() -> None:
    cost = calculate_cost("unknown-experimental-model", 1_000_000, 1_000_000)
    # Fallback is $0.50 + $1.50 = $2.00
    assert cost == Decimal("2.000000")


def test_usage_record_creation() -> None:
    usage = TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60, cost_estimate=Decimal("0.000015"))
    record = UsageRecord.from_usage(
        provider="gemini",
        model="gemini-2.5-flash",
        usage=usage,
        latency_ms=120,
        status="succeeded",
    )

    assert record.provider == "gemini"
    assert record.model == "gemini-2.5-flash"
    assert record.prompt_tokens == 40
    assert record.completion_tokens == 20
    assert record.total_tokens == 60
    assert record.latency_ms == 120
    assert record.status == "succeeded"
    assert record.created_at is not None
