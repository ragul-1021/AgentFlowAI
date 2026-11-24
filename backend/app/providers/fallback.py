"""Free-only model fallback router.

Coordinates fallback strictly between verified free-tier models when rate limits
occur, preventing any automatic or silent upgrades to paid tiers.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.providers.base import LLMRequest, LLMResponse
from app.providers.exceptions import (
    AllFreeTiersExhaustedError,
    PaidModelBlockedError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from app.providers.guardrails import validate_model_allowed

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FreeTierFallbackRouter:
    """Routes generation requests across an ordered chain of free-tier models."""

    def __init__(
        self,
        primary_model: str,
        fallback_models: list[str],
        provider_name: str,
        *,
        free_only: bool = True,
        allow_paid_models: bool = False,
        custom_allowlist: str = "",
    ) -> None:
        self.provider_name = provider_name
        self.free_only = free_only
        self.allow_paid_models = allow_paid_models
        self.custom_allowlist = custom_allowlist

        # Validate every model in the chain before allowing router construction
        all_models = [primary_model] + fallback_models
        for m in all_models:
            validate_model_allowed(
                m,
                provider_name,
                free_only=free_only,
                allow_paid_models=allow_paid_models,
                custom_allowlist=custom_allowlist,
            )

        self.model_chain = all_models

    async def execute(
        self,
        operation: Callable[[str], Awaitable[T]],
    ) -> T:
        """Attempt operation through the free-tier model chain.

        If all free models are rate-limited or unavailable, raises AllFreeTiersExhaustedError.
        Never upgrades to a paid model.
        """
        last_retry_after: float | None = None
        errors: list[str] = []

        for attempt_idx, model in enumerate(self.model_chain, start=1):
            try:
                # Re-verify model safety before dispatching
                validate_model_allowed(
                    model,
                    self.provider_name,
                    free_only=self.free_only,
                    allow_paid_models=self.allow_paid_models,
                    custom_allowlist=self.custom_allowlist,
                )
                logger.info(
                    "[%s] Executing request on free model '%s' (candidate %d/%d)",
                    self.provider_name,
                    model,
                    attempt_idx,
                    len(self.model_chain),
                )
                return await operation(model)
            except (ProviderRateLimitError, ProviderUnavailableError) as exc:
                errors.append(f"{model}: {exc.message}")
                if getattr(exc, "retry_after", None):
                    last_retry_after = exc.retry_after

                logger.warning(
                    "[%s] Free model '%s' hit limits: %s. Checking free fallbacks...",
                    self.provider_name,
                    model,
                    exc.message,
                )
                continue

        # If we reached here, every free-tier option in the chain failed
        summary = "; ".join(errors)
        raise AllFreeTiersExhaustedError(
            f"All free-tier models for [{self.provider_name}] reached their limits ({summary}). "
            f"Paid models are blocked. Please retry later.",
            provider=self.provider_name,
            retry_after=last_retry_after or 60.0,
        )
