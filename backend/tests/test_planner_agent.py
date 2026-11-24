"""Tests for PlannerAgent."""

from __future__ import annotations

import pytest

from app.agents import AgentValidationError, PlannerAgent, PlannerInput, PlannerOutput
from app.enums import ExecutionStatus, ResearchDepth
from app.providers import MockGeminiProvider, PaidModelBlockedError


@pytest.mark.asyncio
async def test_planner_agent_successful_generation():
    provider = MockGeminiProvider()
    planner = PlannerAgent(provider)

    payload = PlannerInput(
        topic="PostgreSQL Vector Indexes with pgvector",
        objective="Analyze performance tradeoffs between HNSW and IVFFlat indexes",
        depth=ResearchDepth.STANDARD,
        max_questions=4,
        max_queries=6,
    )

    result = await planner.run(payload)
    assert result.status == ExecutionStatus.SUCCEEDED
    assert isinstance(result.data, PlannerOutput)
    assert result.data.title
    assert result.data.refined_objective
    assert len(result.data.research_questions) <= 4
    assert len(result.data.search_queries) <= 6
    assert result.usage.total_tokens > 0
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_planner_agent_rejects_empty_topic():
    provider = MockGeminiProvider()
    planner = PlannerAgent(provider)

    # Empty/whitespace topic raises AgentValidationError
    with pytest.raises(AgentValidationError):
        await planner.run(PlannerInput.model_construct(topic="   ", objective="Test"))


def test_planner_agent_blocks_paid_model_configuration():
    provider = MockGeminiProvider()
    with pytest.raises(PaidModelBlockedError):
        PlannerAgent(provider, model="claude-3-opus")
