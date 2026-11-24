"""Tests for pipeline failures, cancellation, and error handling in AgentFlow AI."""

from __future__ import annotations

import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock

from app.config import Settings
from app.enums import TaskStatus, TaskStage, AgentName
from app.models import ResearchTask
from app.orchestration.state import PipelineState
from app.orchestration.exceptions import PipelineCancelledError, PipelineError
from app.orchestration.orchestrator import ResearchOrchestrator
from app.agents.schemas import PlannerInput, PlannerOutput
from tests.fakes import FakeSessionmaker


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        ai_mock_mode=True,
        gemini_model="gemini-2.0-flash",
        gemini_api_key="AIzaSyMockKeyForTestingPurposes123456",
        free_only=True,
        allow_paid_models=False,
    )


@pytest.fixture
def fake_db():
    return FakeSessionmaker()


@pytest.mark.asyncio
async def test_cancellation_between_stages(test_settings, fake_db):
    """Test cancellation between pipeline stages halts execution and marks task as CANCELLED."""
    from app.schemas import TaskCreate
    orchestrator = ResearchOrchestrator(
        fake_db,
        settings=test_settings,
    )

    request = TaskCreate(
        topic="Cancelled Task",
        description="Testing cancellation flow",
    )
    task_id = uuid.uuid4()

    # Pre-emptively request cancellation on pipeline execute by hooking planner
    original_planner_run = orchestrator.planner_agent.run
    async def cancelling_planner_run(*args, **kwargs):
        orchestrator.cancel(task_id)
        return await original_planner_run(*args, **kwargs)

    orchestrator.planner_agent.run = cancelling_planner_run

    with pytest.raises(PipelineCancelledError):
        await orchestrator.run(request, task_id=task_id)

    # Verify task in DB is marked CANCELLED, never COMPLETED
    async with fake_db() as session:
        db_task = await session.get(ResearchTask, task_id)
        assert db_task is not None
        assert db_task.status == TaskStatus.CANCELLED.value
        assert db_task.status != TaskStatus.APPROVED.value


@pytest.mark.asyncio
async def test_planner_failure_marks_task_failed(test_settings, fake_db):
    """Test that if the planner agent fails, the task is marked as FAILED and error stored."""
    from app.schemas import TaskCreate
    orchestrator = ResearchOrchestrator(
        fake_db,
        settings=test_settings,
    )

    request = TaskCreate(
        topic="Failing Task",
        description="Testing planner failure",
    )
    task_id = uuid.uuid4()

    # Mock planner agent to raise an unhandled exception
    orchestrator.planner_agent.run = AsyncMock(
        side_effect=RuntimeError("Simulated catastrophic planner error")
    )

    with pytest.raises(PipelineError, match="Pipeline execution failed at"):
        await orchestrator.run(request, task_id=task_id)

    async with fake_db() as session:
        db_task = await session.get(ResearchTask, task_id)
        assert db_task is not None
        assert db_task.status == TaskStatus.FAILED.value
        assert "Simulated catastrophic planner error" in db_task.error_message


@pytest.mark.asyncio
async def test_secret_redaction_in_pipeline_failure(fake_db):
    """Test that API secrets are redacted in error messages persisted to database."""
    from app.schemas import TaskCreate
    secret_key = "AIzaSySecretTestingKeyThatMustBeRedacted999"
    settings = Settings(
        ai_mock_mode=True,
        gemini_model="gemini-2.0-flash",
        gemini_api_key=secret_key,
        free_only=True,
        allow_paid_models=False,
    )

    orchestrator = ResearchOrchestrator(
        fake_db,
        settings=settings,
    )

    request = TaskCreate(
        topic="Secret Redaction Task",
        description="Testing secret redaction",
    )
    task_id = uuid.uuid4()

    # Planner throws error containing the secret key
    orchestrator.planner_agent.run = AsyncMock(
        side_effect=RuntimeError(f"Connection failed using key {secret_key}")
    )

    with pytest.raises(PipelineError):
        await orchestrator.run(request, task_id=task_id)

    async with fake_db() as session:
        db_task = await session.get(ResearchTask, task_id)
        assert db_task is not None
        assert secret_key not in db_task.error_message
        assert "[redacted]" in db_task.error_message.lower()
