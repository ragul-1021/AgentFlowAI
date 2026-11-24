"""Token usage tracking and cost calculation catalog.

Tracks prompt tokens, completion tokens, total tokens, latency, and estimated cost
based on configurable model pricing tiers with high-precision Decimal arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ModelPricing:
    """Pricing in USD per 1,000,000 tokens."""
    prompt_price_per_m: Decimal
    completion_price_per_m: Decimal


# Pricing catalog: USD per 1M tokens (verified reference pricing)
DEFAULT_PRICING: dict[str, ModelPricing] = {
    # Google Gemini Models
    "gemini-2.5-flash": ModelPricing(
        prompt_price_per_m=Decimal("0.15"),
        completion_price_per_m=Decimal("0.60"),
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        prompt_price_per_m=Decimal("0.075"),
        completion_price_per_m=Decimal("0.30"),
    ),
    "gemini-2.0-flash": ModelPricing(
        prompt_price_per_m=Decimal("0.10"),
        completion_price_per_m=Decimal("0.40"),
    ),
    "gemini-1.5-flash": ModelPricing(
        prompt_price_per_m=Decimal("0.075"),
        completion_price_per_m=Decimal("0.30"),
    ),
    "gemini-1.5-pro": ModelPricing(
        prompt_price_per_m=Decimal("1.25"),
        completion_price_per_m=Decimal("5.00"),
    ),

    # Groq Supported Models
    "openai/gpt-oss-120b": ModelPricing(
        prompt_price_per_m=Decimal("0.59"),
        completion_price_per_m=Decimal("0.79"),
    ),
    "openai/gpt-oss-20b": ModelPricing(
        prompt_price_per_m=Decimal("0.20"),
        completion_price_per_m=Decimal("0.40"),
    ),
    "openai/gpt-oss-safeguard-20b": ModelPricing(
        prompt_price_per_m=Decimal("0.20"),
        completion_price_per_m=Decimal("0.40"),
    ),
}

# Conservative fallback pricing for custom / unlisted models
FALLBACK_PRICING = ModelPricing(
    prompt_price_per_m=Decimal("0.50"),
    completion_price_per_m=Decimal("1.50"),
)


class TokenUsage(BaseModel):
    """Normalized token usage and cost metrics."""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_estimate: Decimal = Field(default=Decimal("0.000000"))

    def __add__(self, other: TokenUsage) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_estimate=self.cost_estimate + other.cost_estimate,
        )


def get_model_pricing(model_name: str) -> ModelPricing:
    """Find pricing for model or matching prefix."""
    normalized = model_name.strip().lower()
    if normalized in DEFAULT_PRICING:
        return DEFAULT_PRICING[normalized]

    # Prefix match (e.g. 'models/gemini-2.5-flash')
    for key, pricing in DEFAULT_PRICING.items():
        if key in normalized or normalized.endswith(key):
            return pricing

    return FALLBACK_PRICING


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_catalog: dict[str, ModelPricing] | None = None,
) -> Decimal:
    """Calculate estimated cost in USD with 6 decimal places."""
    catalog = pricing_catalog or DEFAULT_PRICING
    pricing = get_model_pricing(model) if model not in catalog else catalog[model]

    one_million = Decimal("1000000")
    p_tokens = Decimal(max(0, prompt_tokens))
    c_tokens = Decimal(max(0, completion_tokens))

    prompt_cost = (p_tokens / one_million) * pricing.prompt_price_per_m
    completion_cost = (c_tokens / one_million) * pricing.completion_price_per_m
    total = prompt_cost + completion_cost
    return total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


@dataclass
class UsageRecord:
    """Audit record capturing an individual provider invocation."""
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_estimate: Decimal
    latency_ms: int
    status: str
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_usage(
        cls,
        provider: str,
        model: str,
        usage: TokenUsage,
        latency_ms: int,
        status: str = "succeeded",
        error: str | None = None,
    ) -> UsageRecord:
        return cls(
            provider=provider,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_estimate=usage.cost_estimate,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )
