"""Tests for orchestrator configuration, free-tier guardrails, and human approval preservation."""

from __future__ import annotations

import uuid
import pytest

from app.config import Settings
from app.enums import TaskStatus, TaskStage
from app.models import ResearchTask, ReportApproval
from app.providers.guardrails import PaidModelBlockedError
from app.orchestration.orchestrator import ResearchOrchestrator
from tests.fakes import FakeSessionmaker


@pytest.fixture
def fake_db():
    return FakeSessionmaker()


def test_orchestrator_rejects_unverified_model(fake_db):
    """Test that setting an unverified/paid model in GEMINI_MODEL fails safely at initialization."""
    settings = Settings(
        ai_mock_mode=False,
        gemini_model="gemini-3.8-flash",  # Not in verified free allowlist
        gemini_api_key="AIzaSyMockKeyForTestingPurposes123456",
        free_only=True,
        allow_paid_models=False,
    )

    with pytest.raises(PaidModelBlockedError) as exc_info:
        ResearchOrchestrator(
            fake_db,
            settings=settings,
        )

    assert "gemini-3.8-flash" in str(exc_info.value)
    assert "not in the verified free-tier registry" in str(exc_info.value)


def test_orchestrator_rejects_paid_provider_model(fake_db):
    """Test that orchestrator rejects known paid models like gpt-4o under free_only=True."""
    settings = Settings(
        ai_mock_mode=False,
        gemini_model="openai/gpt-4o",
        gemini_api_key="AIzaSyMockKeyForTestingPurposes123456",
        free_only=True,
        allow_paid_models=False,
    )

    with pytest.raises(PaidModelBlockedError):
        ResearchOrchestrator(
            fake_db,
            settings=settings,
        )


def test_orchestrator_does_not_silently_substitute_model(fake_db):
    """Test that configured gemini_model is preserved exactly and not silently substituted."""
    settings = Settings(
        ai_mock_mode=True,
        gemini_model="gemini-2.0-flash",
        gemini_api_key="AIzaSyMockKeyForTestingPurposes123456",
        free_only=True,
        allow_paid_models=False,
    )

    orchestrator = ResearchOrchestrator(
        fake_db,
        settings=settings,
    )

    assert orchestrator.gemini_model == "gemini-2.0-flash"


@pytest.mark.asyncio
async def test_human_approval_is_preserved_and_never_auto_approved(fake_db):
    """
    Test that after pipeline completion:
    1. TaskStatus is strictly PENDING_APPROVAL.
    2. No ReportApproval record is created automatically by the reviewer.
    """
    settings = Settings(
        ai_mock_mode=True,
        gemini_model="gemini-2.0-flash",
        gemini_api_key="AIzaSyMockKeyForTestingPurposes123456",
        free_only=True,
        allow_paid_models=False,
    )

    orchestrator = ResearchOrchestrator(
        fake_db,
        settings=settings,
    )

    from app.schemas import TaskCreate

    request = TaskCreate(
        topic="Human Approval Verification Task",
        description="Testing that reviewer recommendation never auto-approves",
    )
    task_id = uuid.uuid4()
    result = await orchestrator.run(request, task_id=task_id)

    # 1. State and Task must be PENDING_APPROVAL
    assert result.status == TaskStatus.PENDING_APPROVAL

    async with fake_db() as session:
        db_task = await session.get(ResearchTask, task_id)
        assert db_task is not None
        assert db_task.status == TaskStatus.PENDING_APPROVAL.value
        assert db_task.status != TaskStatus.APPROVED.value

        # 2. Verify no ReportApproval rows exist
        approvals = [
            entity for entity in session._store.values()
            if isinstance(entity, ReportApproval)
        ]
        assert len(approvals) == 0, "Reviewer must never create a ReportApproval record; only humans can approve"
