"""Base agent class and execution result structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import logging
import time
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from app.agents.context import AgentContext
from app.agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentProviderError,
    AgentValidationError,
)
from app.config import Settings, get_settings
from app.core.redaction import redact
from app.core.tokens import TokenUsage
from app.enums import AgentName, ExecutionStatus
from app.providers.base import BaseLLMProvider, ChatMessage, LLMRequest
from app.providers.exceptions import (
    AllFreeTiersExhaustedError,
    PaidModelBlockedError,
    ProviderError,
    ProviderRateLimitError,
    QuotaExceededError,
)
from app.providers.guardrails import validate_model_allowed

logger = logging.getLogger(__name__)

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


@dataclass
class AgentExecutionResult(Generic[TOutput]):
    """Standardized result envelope returned by every agent execution."""

    data: TOutput
    agent_name: AgentName | str
    provider: str
    model: str
    status: ExecutionStatus
    latency_ms: int
    usage: TokenUsage
    cost_estimate: Decimal
    started_at: datetime
    finished_at: datetime
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(Generic[TInput, TOutput], ABC):
    """Abstract base class for all AgentFlow AI agents."""

    def __init__(
        self,
        agent_name: AgentName,
        provider: BaseLLMProvider,
        *,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.provider = provider
        self.settings = settings or get_settings()
        self.model = model or provider.default_model

        # Cost-protection guardrail: ensure configured model is explicitly free
        validate_model_allowed(
            self.model,
            self.provider.provider_name,
            free_only=self.settings.free_only,
            allow_paid_models=self.settings.allow_paid_models,
            custom_allowlist=self.settings.free_model_allowlist,
        )

    @abstractmethod
    async def _execute(
        self,
        input_data: TInput,
        context: AgentContext,
    ) -> tuple[TOutput, TokenUsage, dict[str, Any]]:
        """Core execution logic implemented by each specialized agent."""
        ...

    async def run(
        self,
        input_data: TInput,
        context: AgentContext | None = None,
    ) -> AgentExecutionResult[TOutput]:
        """Execute the agent with validation, error isolation, secret redaction, and usage tracking."""
        ctx = context or AgentContext(stage=str(self.agent_name))
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()

        logger.info(
            "[%s] Starting execution (task_id=%s, model=%s, provider=%s)",
            self.agent_name,
            ctx.task_id,
            self.model,
            self.provider.provider_name,
        )

        try:
            # Re-verify guardrail before execution
            validate_model_allowed(
                self.model,
                self.provider.provider_name,
                free_only=self.settings.free_only,
                allow_paid_models=self.settings.allow_paid_models,
                custom_allowlist=self.settings.free_model_allowlist,
            )

            output_data, usage, extra_meta = await self._execute(input_data, ctx)
            latency_ms = max(1, int((time.perf_counter() - start_time) * 1000))
            finished_at = datetime.now(timezone.utc)

            logger.info(
                "[%s] Completed successfully in %dms (tokens=%d, cost=$%s)",
                self.agent_name,
                latency_ms,
                usage.total_tokens,
                usage.cost_estimate,
            )

            return AgentExecutionResult(
                data=output_data,
                agent_name=self.agent_name,
                provider=self.provider.provider_name,
                model=self.model,
                status=ExecutionStatus.SUCCEEDED,
                latency_ms=latency_ms,
                usage=usage,
                cost_estimate=usage.cost_estimate,
                started_at=started_at,
                finished_at=finished_at,
                metadata=extra_meta,
            )

        except (PaidModelBlockedError, QuotaExceededError, AllFreeTiersExhaustedError) as exc:
            latency_ms = max(1, int((time.perf_counter() - start_time) * 1000))
            finished_at = datetime.now(timezone.utc)
            safe_error = redact(str(exc), self.settings.redaction_secrets())
            logger.error("[%s] Cost protection error: %s", self.agent_name, safe_error)
            raise AgentProviderError(
                safe_error,
                agent_name=str(self.agent_name),
                provider=self.provider.provider_name,
                details={"cost_protection": True},
            ) from exc

        except ProviderRateLimitError as exc:
            latency_ms = max(1, int((time.perf_counter() - start_time) * 1000))
            safe_error = redact(exc.message, self.settings.redaction_secrets())
            logger.warning("[%s] Provider rate limit encountered: %s", self.agent_name, safe_error)
            raise AgentProviderError(
                safe_error,
                agent_name=str(self.agent_name),
                provider=self.provider.provider_name,
                status_code=exc.status_code,
                retry_after=exc.retry_after,
            ) from exc

        except ProviderError as exc:
            latency_ms = max(1, int((time.perf_counter() - start_time) * 1000))
            safe_error = redact(exc.message, self.settings.redaction_secrets())
            logger.error("[%s] Provider failure: %s", self.agent_name, safe_error)
            raise AgentProviderError(
                safe_error,
                agent_name=str(self.agent_name),
                provider=self.provider.provider_name,
                status_code=exc.status_code,
            ) from exc

        except ValidationError as exc:
            safe_error = redact(str(exc), self.settings.redaction_secrets())
            logger.error("[%s] Schema validation error: %s", self.agent_name, safe_error)
            raise AgentValidationError(
                f"Validation failed in {self.agent_name}: {safe_error}",
                agent_name=str(self.agent_name),
            ) from exc

        except AgentError:
            raise

        except Exception as exc:
            latency_ms = max(1, int((time.perf_counter() - start_time) * 1000))
            safe_error = redact(str(exc), self.settings.redaction_secrets())
            logger.exception("[%s] Unexpected agent execution error: %s", self.agent_name, safe_error)
            raise AgentExecutionError(
                f"Execution failed in {self.agent_name}: {safe_error}",
                agent_name=str(self.agent_name),
            ) from exc
