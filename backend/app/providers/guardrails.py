"""Cost protection guardrails, model access tier classification, and quota tracking.

Ensures AgentFlow AI never calls paid models, never auto-upgrades to paid plans,
distinguishes free model access tiers (permanent, rate-limited free-tier, temporary trial),
enforces request token limits, and persistently tracks search and request quotas across restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from app.providers.exceptions import PaidModelBlockedError, QuotaExceededError

logger = logging.getLogger(__name__)

# Backend base directory for default persistence files
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class AccessTier(str, Enum):
    """Classification of free model access mechanisms."""

    PERMANENT_FREE = "permanent_free"
    """Permanently free model endpoints (e.g. OpenRouter :free endpoints, open community routes)."""

    FREE_TIER_RATE_LIMITED = "free_tier_rate_limited"
    """Account free-tier with ongoing recurring daily/minute rate limits (e.g. Gemini Developer API, Groq Free Plan)."""

    TEMPORARY_TRIAL = "temporary_trial"
    """Evaluation, experimentation, or promotional trial access with fixed or time-bound quotas (e.g. Cohere Trial, Mistral Exp)."""


@dataclass(frozen=True)
class FreeModelMetadata:
    """Metadata describing a verified free-tier model."""

    model_id: str
    provider: str
    access_tier: AccessTier
    rpm_limit: int | None = None
    rpd_limit: int | None = None
    tpm_limit: int | None = None
    monthly_limit: int | None = None
    description: str = ""


# Verified free-tier models catalog with official documentation mappings
VERIFIED_FREE_MODELS_REGISTRY: dict[str, FreeModelMetadata] = {
    # -------------------------------------------------------------------------
    # Google Gemini Developer API (Free Tier)
    # Docs: 15 RPM, 1,000,000 TPM, 1,500 RPD on free tier accounts
    # -------------------------------------------------------------------------
    "gemini-2.5-flash": FreeModelMetadata(
        model_id="gemini-2.5-flash",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=15,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 2.5 Flash flagship multimodal reasoning free tier",
    ),
    "models/gemini-2.5-flash": FreeModelMetadata(
        model_id="models/gemini-2.5-flash",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=15,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 2.5 Flash (models/ prefix)",
    ),
    "gemini-2.5-flash-lite": FreeModelMetadata(
        model_id="gemini-2.5-flash-lite",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=15,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 2.5 Flash Lite high-throughput free tier",
    ),
    "models/gemini-2.5-flash-lite": FreeModelMetadata(
        model_id="models/gemini-2.5-flash-lite",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=15,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 2.5 Flash Lite (models/ prefix)",
    ),
    "gemini-2.0-flash": FreeModelMetadata(
        model_id="gemini-2.0-flash",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=15,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 2.0 Flash GA free tier",
    ),
    "models/gemini-2.0-flash": FreeModelMetadata(
        model_id="models/gemini-2.0-flash",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=15,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 2.0 Flash (models/ prefix)",
    ),
    "gemini-1.5-flash": FreeModelMetadata(
        model_id="gemini-1.5-flash",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=15,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 1.5 Flash previous generation free tier",
    ),
    "models/gemini-1.5-flash": FreeModelMetadata(
        model_id="models/gemini-1.5-flash",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=15,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 1.5 Flash (models/ prefix)",
    ),
    "gemini-2.0-flash-exp": FreeModelMetadata(
        model_id="gemini-2.0-flash-exp",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=10,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 2.0 Flash Experimental free tier",
    ),
    "models/gemini-2.0-flash-exp": FreeModelMetadata(
        model_id="models/gemini-2.0-flash-exp",
        provider="gemini",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpm_limit=10,
        rpd_limit=1500,
        tpm_limit=1_000_000,
        description="Google Gemini 2.0 Flash Experimental (models/ prefix)",
    ),

    # -------------------------------------------------------------------------
    # Note on Groq: Priced models (e.g. openai/gpt-oss-120b, llama-3.1-8b-instant)
    # have been removed. If a verified free model becomes available for Groq,
    # configure it via FREE_MODEL_ALLOWLIST or update this registry.
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Cloudflare Workers AI (Explicit Free Allowance Models Only)
    # Docs: 10,000 Neurons per day free allowance across verified endpoints.
    # Entire @cf/ namespace is NOT classified as free.
    # -------------------------------------------------------------------------
    "@cf/meta/llama-3.2-3b-instruct": FreeModelMetadata(
        model_id="@cf/meta/llama-3.2-3b-instruct",
        provider="cloudflare",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpd_limit=1000,
        description="Cloudflare Workers AI Llama 3.2 3B within 10k daily Neurons",
    ),
    "@cf/meta/llama-3.2-1b-instruct": FreeModelMetadata(
        model_id="@cf/meta/llama-3.2-1b-instruct",
        provider="cloudflare",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpd_limit=2000,
        description="Cloudflare Workers AI Llama 3.2 1B within 10k daily Neurons",
    ),
    "@cf/mistral/mistral-7b-instruct-v0.1": FreeModelMetadata(
        model_id="@cf/mistral/mistral-7b-instruct-v0.1",
        provider="cloudflare",
        access_tier=AccessTier.FREE_TIER_RATE_LIMITED,
        rpd_limit=1000,
        description="Cloudflare Workers AI Mistral 7B within 10k daily Neurons",
    ),

    # -------------------------------------------------------------------------
    # Mistral AI Free Experimentation Tier
    # Docs: 1 request/second rate-limited evaluation/experimentation
    # -------------------------------------------------------------------------
    "mistral-small-latest": FreeModelMetadata(
        model_id="mistral-small-latest",
        provider="mistral",
        access_tier=AccessTier.TEMPORARY_TRIAL,
        rpm_limit=60,
        description="Mistral AI free experimentation tier",
    ),
    "open-mistral-7b": FreeModelMetadata(
        model_id="open-mistral-7b",
        provider="mistral",
        access_tier=AccessTier.TEMPORARY_TRIAL,
        rpm_limit=60,
        description="Mistral AI open 7B experimentation tier",
    ),
    "open-mistral-nemo": FreeModelMetadata(
        model_id="open-mistral-nemo",
        provider="mistral",
        access_tier=AccessTier.TEMPORARY_TRIAL,
        rpm_limit=60,
        description="Mistral AI NeMo 12B experimentation tier",
    ),

    # -------------------------------------------------------------------------
    # Cohere Free Trial
    # Docs: 10 calls/minute, up to 1,000 calls/month total trial limit
    # -------------------------------------------------------------------------
    "command-r": FreeModelMetadata(
        model_id="command-r",
        provider="cohere",
        access_tier=AccessTier.TEMPORARY_TRIAL,
        rpm_limit=10,
        monthly_limit=1000,
        description="Cohere trial key (10 RPM, 1,000 monthly calls)",
    ),
    "command-r-plus": FreeModelMetadata(
        model_id="command-r-plus",
        provider="cohere",
        access_tier=AccessTier.TEMPORARY_TRIAL,
        rpm_limit=10,
        monthly_limit=1000,
        description="Cohere trial key (10 RPM, 1,000 monthly calls)",
    ),
    "command-light": FreeModelMetadata(
        model_id="command-light",
        provider="cohere",
        access_tier=AccessTier.TEMPORARY_TRIAL,
        rpm_limit=10,
        monthly_limit=1000,
        description="Cohere light model trial key",
    ),
}

# Immutable set of verified model identifiers for fast lookup
VERIFIED_FREE_MODELS: frozenset[str] = frozenset(VERIFIED_FREE_MODELS_REGISTRY.keys())


def get_model_access_tier(model: str) -> AccessTier | None:
    """Return the AccessTier classification for a model, or None if unclassified/unknown."""
    normalized = model.strip().lower()
    if normalized in VERIFIED_FREE_MODELS_REGISTRY:
        return VERIFIED_FREE_MODELS_REGISTRY[normalized].access_tier
    if normalized.endswith(":free"):
        return AccessTier.PERMANENT_FREE
    return None


def get_model_metadata(model: str) -> FreeModelMetadata | None:
    """Retrieve full metadata for a verified free model."""
    normalized = model.strip().lower()
    if normalized in VERIFIED_FREE_MODELS_REGISTRY:
        return VERIFIED_FREE_MODELS_REGISTRY[normalized]
    if normalized.endswith(":free"):
        return FreeModelMetadata(
            model_id=model,
            provider="openrouter",
            access_tier=AccessTier.PERMANENT_FREE,
            rpm_limit=20,
            rpd_limit=200,
            description="OpenRouter community free router endpoint",
        )
    return None


def is_free_tier_model(model: str, custom_allowlist: set[str] | None = None) -> bool:
    """Return True if model is verified free or matches open free provider patterns.

    Unknown models are rejected by default. Entire provider namespaces (like @cf/)
    are NOT treated as free.
    """
    normalized = model.strip().lower()

    # 1. Direct match in verified catalog
    if normalized in VERIFIED_FREE_MODELS:
        return True

    # 2. OpenRouter free endpoints strictly ending in ':free'
    if normalized.endswith(":free"):
        return True

    # 3. User-provided custom free model allowlist
    if custom_allowlist and normalized in custom_allowlist:
        return True

    return False


def parse_custom_allowlist(raw_allowlist: str) -> set[str]:
    """Parse comma-separated free model strings."""
    if not raw_allowlist:
        return set()
    return {item.strip().lower() for item in raw_allowlist.split(",") if item.strip()}


def validate_model_allowed(
    model: str | None,
    provider: str,
    *,
    free_only: bool = True,
    allow_paid_models: bool = False,
    custom_allowlist: str = "",
) -> None:
    """Validate that the requested model is allowed under cost protection policies.

    Raises PaidModelBlockedError if a paid, missing, or unverified model is requested while
    FREE_ONLY is active or ALLOW_PAID_MODELS is false.
    """
    if not model or not str(model).strip():
        raise PaidModelBlockedError(
            f"No model configured for provider '{provider}'. A verified free-tier model is required.",
            provider=provider,
            model=str(model or ""),
        )

    if not free_only and allow_paid_models:
        return  # Explicitly unlocked

    parsed_custom = parse_custom_allowlist(custom_allowlist)
    if not is_free_tier_model(model, parsed_custom):
        logger.error(
            "Cost protection blocked request to unknown/paid model '%s' on provider '%s' (FREE_ONLY=%s, ALLOW_PAID_MODELS=%s)",
            model,
            provider,
            free_only,
            allow_paid_models,
        )
        raise PaidModelBlockedError(
            f"Model '{model}' is rejected. Cost protection is active (FREE_ONLY=true) and '{model}' "
            f"is not in the verified free-tier registry or explicit allowlist.",
            provider=provider,
            model=model,
        )


def validate_token_limit(max_tokens: int | None, max_request_tokens: int = 8192) -> None:
    """Enforce maximum output token budget per request."""
    if max_tokens is not None and max_tokens > max_request_tokens:
        raise QuotaExceededError(
            f"Requested max_tokens ({max_tokens}) exceeds the allowed request limit ({max_request_tokens}).",
            provider="guardrails",
        )


class SearchQuotaTracker:
    """Persistent search query quota tracker for Tavily free tier (1,000 queries/month).

    Preserves consumed counts across backend restarts using an atomic JSON state file
    keyed by year and month. Automatically resets when the calendar month rolls over.
    Does not require database schema modifications.
    """

    _instance: ClassVar[SearchQuotaTracker | None] = None

    def __init__(
        self,
        monthly_budget: int = 1000,
        persistence_path: Path | str | None = None,
    ) -> None:
        self.monthly_budget = monthly_budget
        if persistence_path is not None:
            self._storage_path = Path(persistence_path)
        else:
            self._storage_path = _BACKEND_DIR / ".tavily_quota.json"

        self._consumed: int = 0
        self._current_year_month: str = self._get_current_month()
        self._load_persisted_state()

    @classmethod
    def get_instance(
        cls,
        monthly_budget: int = 1000,
        persistence_path: Path | str | None = None,
    ) -> SearchQuotaTracker:
        if cls._instance is None:
            cls._instance = cls(monthly_budget, persistence_path)
        return cls._instance

    @classmethod
    def _reset_instance_for_testing(cls) -> None:
        """Helper for test suites to clear singleton state."""
        cls._instance = None

    @staticmethod
    def _get_current_month() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    @property
    def consumed(self) -> int:
        self._sync_month_rollover()
        return self._consumed

    @property
    def remaining(self) -> int:
        return max(0, self.monthly_budget - self.consumed)

    def _sync_month_rollover(self) -> None:
        """Check if calendar month rolled over; reset count if so."""
        now_month = self._get_current_month()
        if now_month != self._current_year_month:
            logger.info(
                "Tavily quota month rollover detected (%s -> %s). Resetting consumed count.",
                self._current_year_month,
                now_month,
            )
            self._current_year_month = now_month
            self._consumed = 0
            self._save_persisted_state()

    def _load_persisted_state(self) -> None:
        """Load quota state from local JSON file if present."""
        if not self._storage_path:
            return

        try:
            if self._storage_path.is_file():
                content = self._storage_path.read_text(encoding="utf-8")
                data = json.loads(content)
                persisted_month = data.get("year_month", "")
                if persisted_month == self._current_year_month:
                    self._consumed = int(data.get("consumed", 0))
                    logger.debug(
                        "Restored Tavily quota state: %d/%d queries consumed in %s",
                        self._consumed,
                        self.monthly_budget,
                        self._current_year_month,
                    )
                else:
                    self._consumed = 0
                    self._save_persisted_state()
        except Exception as exc:
            logger.warning("Could not load persisted Tavily quota state (%s). Initializing to 0.", exc)
            self._consumed = 0

    def _save_persisted_state(self) -> None:
        """Atomically persist quota state to local JSON file."""
        if not self._storage_path:
            return

        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "year_month": self._current_year_month,
                "consumed": self._consumed,
                "monthly_budget": self.monthly_budget,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            tmp_path = self._storage_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(self._storage_path)
        except Exception as exc:
            logger.warning("Could not persist Tavily quota state to disk: %s", exc)

    def check_and_increment(self, count: int = 1) -> None:
        """Check if request fits within the remaining monthly budget and consume it."""
        self._sync_month_rollover()

        if self._consumed + count > self.monthly_budget:
            raise QuotaExceededError(
                f"Tavily free-tier monthly query budget ({self.monthly_budget}) reached for month {self._current_year_month}. "
                f"Currently consumed: {self._consumed}. Please retry next billing cycle.",
                provider="tavily",
            )

        self._consumed += count
        self._save_persisted_state()

    def reset(self) -> None:
        """Explicitly reset quota counter (for testing or billing cycle rollover)."""
        self._consumed = 0
        self._current_year_month = self._get_current_month()
        self._save_persisted_state()


class DailyRequestQuotaTracker:
    """Tracks daily request consumption per provider to enforce free-tier RPD limits."""

    def __init__(self, daily_limits: dict[str, int]) -> None:
        self.daily_limits = daily_limits
        self._counts: dict[str, int] = {}
        self._current_day = self._get_current_day()

    @staticmethod
    def _get_current_day() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _sync_day(self) -> None:
        now_day = self._get_current_day()
        if now_day != self._current_day:
            self._current_day = now_day
            self._counts.clear()

    def check_and_increment(self, provider: str, count: int = 1) -> None:
        self._sync_day()
        limit = self.daily_limits.get(provider.lower())
        if limit is None:
            return

        current = self._counts.get(provider.lower(), 0)
        if current + count > limit:
            raise QuotaExceededError(
                f"Daily request limit ({limit} RPD) for provider '{provider}' reached for day {self._current_day}. "
                f"Currently consumed: {current}. Please retry tomorrow.",
                provider=provider,
            )
        self._counts[provider.lower()] = current + count

    def get_count(self, provider: str) -> int:
        self._sync_day()
        return self._counts.get(provider.lower(), 0)

    def reset(self) -> None:
        self._counts.clear()
        self._current_day = self._get_current_day()
