"""End-to-end tests for the 7-step research pipeline flow and database persistence."""

from __future__ import annotations

import uuid
import pytest

from app.config import Settings
from app.enums import (
    AgentName,
    ReportStatus,
    ResearchDepth,
    ReviewStatus,
    TaskStage,
    TaskStatus,
)
from app.models import (
    AgentExecution,
    FindingSource,
    ReportCitation,
    ReportReview,
    ResearchFinding,
    ResearchPlan,
    ResearchReport,
    ResearchSource,
    ResearchTask,
    RunEvent,
)
from app.orchestration import PipelineResult, ResearchOrchestrator
from app.providers import MockGeminiProvider, MockTavilyProvider
from app.schemas import TaskCreate
from tests.fakes import FakeSessionmaker


@pytest.mark.asyncio
async def test_full_pipeline_mock_execution(settings: Settings):
    sessionmaker = FakeSessionmaker()
    orchestrator = ResearchOrchestrator(
        sessionmaker=sessionmaker,
        settings=settings,
        llm_provider=MockGeminiProvider(default_model="gemini-2.5-flash"),
        search_provider=MockTavilyProvider(),
    )

    request = TaskCreate(
        topic="Vector Indexing Architectures in Production Databases",
        description="Comprehensive evaluation of HNSW and IVFFlat trade-offs",
        depth=ResearchDepth.STANDARD,
        preferred_domains=["postgresql.org", "github.com"],
        excluded_domains=["spam-site.com"],
    )

    task_id = uuid.uuid4()
    result = await orchestrator.run(request, task_id=task_id)

    # 1. Verify PipelineResult envelope
    assert isinstance(result, PipelineResult)
    assert result.task_id == task_id
    assert result.stage == TaskStage.COMPLETED
    # Mandatory requirement 6: Final status remains PENDING_APPROVAL
    assert result.status == TaskStatus.PENDING_APPROVAL
    assert result.human_approval_required is True
    assert result.report_id is not None
    assert result.report_title
    assert result.total_tokens > 0
    assert result.review_score is not None
    assert result.review_score >= 0

    # 2. Verify Database Persistence across all models in session history
    history = sessionmaker.history

    # ResearchTask
    tasks = [o for o in history if isinstance(o, ResearchTask)]
    assert len(tasks) >= 1
    final_task = tasks[0]
    assert final_task.id == task_id
    assert final_task.status == TaskStatus.PENDING_APPROVAL.value
    assert final_task.stage == TaskStage.COMPLETED.value
    assert final_task.progress == 100
    assert final_task.total_tokens > 0

    # ResearchPlan
    plans = [o for o in history if isinstance(o, ResearchPlan)]
    assert len(plans) == 1
    assert plans[0].task_id == task_id
    assert plans[0].objective
    assert len(plans[0].questions) > 0
    assert len(plans[0].queries) > 0

    # ResearchSource
    sources = [o for o in history if isinstance(o, ResearchSource)]
    assert len(sources) > 0
    for src in sources:
        assert src.task_id == task_id
        assert src.url.startswith("http")
        assert src.domain != "spam-site.com"

    # ResearchFinding
    findings = [o for o in history if isinstance(o, ResearchFinding)]
    assert len(findings) > 0
    for f in findings:
        assert f.task_id == task_id
        assert f.statement

    # ResearchReport
    reports = [o for o in history if isinstance(o, ResearchReport)]
    assert len(reports) >= 1
    final_report = reports[-1]
    assert final_report.task_id == task_id
    assert final_report.status == ReportStatus.PENDING_APPROVAL.value
    assert final_report.word_count > 0

    # ReportReview
    reviews = [o for o in history if isinstance(o, ReportReview)]
    assert len(reviews) >= 1
    assert reviews[-1].quality_score >= 0

    # ReportCitation
    citations = [o for o in history if isinstance(o, ReportCitation)]
    assert len(citations) >= 1

    # AgentExecution: records exist for planner, researcher, analyst, writer, reviewer
    executions = [o for o in history if isinstance(o, AgentExecution)]
    agents_executed = {e.agent for e in executions}
    assert AgentName.PLANNER.value in agents_executed
    assert AgentName.RESEARCHER.value in agents_executed
    assert AgentName.ANALYST.value in agents_executed
    assert AgentName.WRITER.value in agents_executed
    assert AgentName.REVIEWER.value in agents_executed

    # RunEvents
    events = [o for o in history if isinstance(o, RunEvent)]
    assert len(events) >= 5
