"""Tests for pipeline revision cycles, bounds, and feedback passing."""

from __future__ import annotations

import uuid
import pytest

from app.agents import ReviewerOutput
from app.config import Settings
from app.enums import TaskStage, TaskStatus
from app.models import ResearchReport, ResearchTask
from app.orchestration import ResearchOrchestrator
from app.providers import LLMRequest, LLMResponse, MockGeminiProvider, MockTavilyProvider
from app.schemas import TaskCreate
from tests.fakes import FakeSessionmaker


class ImperfectReviewMockGemini(MockGeminiProvider):
    """Mock Gemini provider that gives low quality scores to force revision rounds."""

    def __init__(self, target_low_scores: int = 1):
        super().__init__(default_model="gemini-2.5-flash")
        self.review_calls = 0
        self.target_low_scores = target_low_scores

    async def generate_structured(self, request: LLMRequest, response_schema: type, model=None):
        data, res = await super().generate_structured(request, response_schema, model=model)
        if response_schema is ReviewerOutput:
            self.review_calls += 1
            if self.review_calls <= self.target_low_scores:
                # Low score and critical concern triggers revision
                data.quality_score = 55
                data.accuracy_concerns = ["Critical empirical claim lacks supporting data."]
                data.revision_suggestions = ["Incorporate quantitative benchmarking numbers."]
                data.approval_recommendation = False
            else:
                data.quality_score = 88
                data.accuracy_concerns = []
                data.approval_recommendation = True
        return data, res


@pytest.mark.asyncio
async def test_pipeline_triggers_revision_and_passes_feedback(settings: Settings):
    sessionmaker = FakeSessionmaker()
    llm = ImperfectReviewMockGemini(target_low_scores=1)
    orchestrator = ResearchOrchestrator(
        sessionmaker=sessionmaker,
        settings=settings,
        llm_provider=llm,
        search_provider=MockTavilyProvider(),
    )

    request = TaskCreate(
        topic="Database Engine Architecture and Indexing",
        description="Detailed study of vector indexing and cache behavior",
    )

    result = await orchestrator.run(request)

    # Verify revision was executed
    assert result.state.revision_count == 1
    assert result.status == TaskStatus.PENDING_APPROVAL
    assert result.stage == TaskStage.COMPLETED

    # Verify multiple report versions exist in history (v1 draft, v2 revised)
    reports = [o for o in sessionmaker.history if isinstance(o, ResearchReport)]
    assert len(reports) == 2
    assert reports[0].version == 1
    assert reports[1].version == 2


@pytest.mark.asyncio
async def test_pipeline_revision_strictly_bounded_at_max_revisions(settings: Settings):
    sessionmaker = FakeSessionmaker()
    # Always returns low scores; must stop at max_revisions (2) without infinite loop
    llm = ImperfectReviewMockGemini(target_low_scores=10)
    orchestrator = ResearchOrchestrator(
        sessionmaker=sessionmaker,
        settings=settings,
        llm_provider=llm,
        search_provider=MockTavilyProvider(),
    )

    request = TaskCreate(
        topic="Database Engine Architecture and Indexing",
        description="Detailed study of vector indexing and cache behavior",
    )

    result = await orchestrator.run(request)

    # Strictly capped at max_revisions = 2
    assert result.state.revision_count == 2
    # Task still reaches PENDING_APPROVAL for human inspection
    assert result.status == TaskStatus.PENDING_APPROVAL
    assert result.stage == TaskStage.COMPLETED

    reports = [o for o in sessionmaker.history if isinstance(o, ResearchReport)]
    # v1 (initial) + v2 (revision 1) + v3 (revision 2) = 3 reports max
    assert len(reports) == 3
