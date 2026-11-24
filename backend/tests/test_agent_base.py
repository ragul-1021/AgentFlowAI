"""Tests for shared agent infrastructure, BaseAgent, AgentContext, and error handling."""

from __future__ import annotations

from decimal import Decimal
import pytest

from app.agents import (
    AgentContext,
    AgentExecutionError,
    AgentExecutionResult,
    AgentName,
    AgentProviderError,
    AgentValidationError,
    BaseAgent,
)
from app.config import get_settings
from app.core.tokens import TokenUsage
from app.enums import ExecutionStatus, TaskStage
from app.providers import (
    LLMRequest,
    MockGeminiProvider,
    PaidModelBlockedError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from pydantic import BaseModel, Field


class DummyInput(BaseModel):
    query: str = Field(min_length=2)


class DummyOutput(BaseModel):
    answer: str


class ConcreteDummyAgent(BaseAgent[DummyInput, DummyOutput]):
    def __init__(self, provider, model=None):
        super().__init__(AgentName.PLANNER, provider, model=model or "gemini-2.5-flash")

    async def _execute(self, input_data: DummyInput, context: AgentContext):
        if input_data.query == "trigger_provider_rate_limit":
            raise ProviderRateLimitError("Rate limit reached", provider="gemini", retry_after=45.0)
        if input_data.query == "trigger_provider_down":
            raise ProviderUnavailableError("503 Service Unavailable", provider="gemini")
        if input_data.query == "trigger_leak":
            raise ValueError("Failure with secret AIzaSy" + "A" * 33 + " in stack")

        usage = TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25, cost_estimate=Decimal("0.000100"))
        return DummyOutput(answer=f"Result for {input_data.query}"), usage, {"step": 1}


def test_agent_context_creation_and_fork():
    ctx = AgentContext(stage=TaskStage.PLANNING, metadata={"env": "test"})
    assert ctx.task_id is not None
    assert ctx.stage == TaskStage.PLANNING
    assert ctx.metadata["env"] == "test"

    child = ctx.fork(TaskStage.RESEARCHING, depth="deep")
    assert child.task_id == ctx.task_id
    assert child.stage == TaskStage.RESEARCHING
    assert child.metadata["env"] == "test"
    assert child.metadata["depth"] == "deep"


@pytest.mark.asyncio
async def test_base_agent_successful_run():
    provider = MockGeminiProvider()
    agent = ConcreteDummyAgent(provider)

    result = await agent.run(DummyInput(query="hello agent"))
    assert isinstance(result, AgentExecutionResult)
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.data.answer == "Result for hello agent"
    assert result.provider == "gemini"
    assert result.model == "gemini-2.5-flash"
    assert result.latency_ms > 0
    assert result.usage.total_tokens == 25
    assert result.cost_estimate == Decimal("0.000100")
    assert result.metadata["step"] == 1


@pytest.mark.asyncio
async def test_base_agent_handles_provider_rate_limit():
    provider = MockGeminiProvider()
    agent = ConcreteDummyAgent(provider)

    with pytest.raises(AgentProviderError) as exc_info:
        await agent.run(DummyInput(query="trigger_provider_rate_limit"))

    err = exc_info.value
    assert "Rate limit reached" in str(err)
    assert err.retry_after == 45.0
    assert err.agent_name == "planner"


@pytest.mark.asyncio
async def test_base_agent_handles_provider_unavailable():
    provider = MockGeminiProvider()
    agent = ConcreteDummyAgent(provider)

    with pytest.raises(AgentProviderError) as exc_info:
        await agent.run(DummyInput(query="trigger_provider_down"))

    err = exc_info.value
    assert "Service Unavailable" in str(err)


@pytest.mark.asyncio
async def test_base_agent_redacts_secrets_in_unexpected_errors():
    provider = MockGeminiProvider()
    agent = ConcreteDummyAgent(provider)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run(DummyInput(query="trigger_leak"))

    err_str = str(exc_info.value)
    assert "AIzaSySecretKey12345" not in err_str
    assert "[REDACTED]" in err_str or "Failure with secret" in err_str


def test_base_agent_blocks_paid_model_at_init():
    provider = MockGeminiProvider()
    with pytest.raises(PaidModelBlockedError):
        ConcreteDummyAgent(provider, model="gpt-4o")
