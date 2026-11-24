"""Tests for research report retrieval REST endpoint."""

from __future__ import annotations

import uuid
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.enums import ApprovalDecision, CitationVerdict, ReportStatus, ReviewStatus, TaskStage, TaskStatus
from app.main import create_app
from app.models import ReportApproval, ReportCitation, ReportReview, ResearchReport, ResearchSource, ResearchTask
from tests.fakes import FakeDatabase, FakeSessionmaker


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        app_name="agentflow",
        app_env="development",
        ai_mock_mode=True,
        gemini_model="gemini-2.5-flash",
        gemini_api_key="AIzaSyMockKeyForTestingPurposes123456",
        free_only=True,
        allow_paid_models=False,
    )


@pytest.fixture
def fake_db():
    fdb = FakeDatabase(status="up")
    fdb.sessionmaker = FakeSessionmaker()
    return fdb


@pytest.fixture
def app_with_db(test_settings, fake_db):
    return create_app(test_settings, database=fake_db)


@pytest.mark.asyncio
async def test_get_report_success(app_with_db, fake_db):
    """Test GET /api/tasks/{task_id}/report returns synthesized report with citations and reviews."""
    task_id = uuid.uuid4()
    report_id = uuid.uuid4()
    source_id = uuid.uuid4()

    async with fake_db.sessionmaker() as session:
        task = ResearchTask(
            id=task_id,
            topic="Formal Verification of Smart Contracts",
            status=TaskStatus.PENDING_APPROVAL.value,
            stage=TaskStage.COMPLETED.value,
        )
        session.add(task)

        source = ResearchSource(
            id=source_id,
            task_id=task_id,
            url="https://arxiv.org/abs/2101.00001",
            title="SMT Solving in Smart Contracts",
            domain="arxiv.org",
        )
        session.add(source)

        report = ResearchReport(
            id=report_id,
            task_id=task_id,
            version=1,
            title="Formal Verification of Smart Contracts: State of the Art",
            executive_summary="This survey evaluates symbolic execution and SMT solvers.",
            content=[{"heading": "Executive Summary", "content": "Full summary text here"}],
            word_count=1200,
            status=ReportStatus.PENDING_APPROVAL.value,
            review_status=ReviewStatus.PASSED.value,
        )
        session.add(report)

        citation = ReportCitation(
            id=uuid.uuid4(),
            report_id=report_id,
            source_id=source_id,
            marker=1,
            section_key="body",
            claim_text="Symbolic execution accurately identifies re-entrancy bugs.",
            verdict=CitationVerdict.SUPPORTED.value,
            review_note="Verified against paper benchmarks.",
        )
        citation.source = source
        report.citations.append(citation)

        review = ReportReview(
            id=uuid.uuid4(),
            report_id=report_id,
            revision_no=0,
            passed=True,
            quality_score=90,
            citation_coverage=1.0,
            unsupported_count=0,
            feedback="Exemplary report with rigorous verification.",
        )
        report.reviews.append(review)
        await session.commit()

    transport = ASGITransport(app=app_with_db)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/tasks/{task_id}/report")
        assert response.status_code == 200

        data = response.json()
        assert data["report_id"] == str(report_id)
        assert data["task_id"] == str(task_id)
        assert data["version"] == 1
        assert data["title"] == "Formal Verification of Smart Contracts: State of the Art"
        assert data["word_count"] == 1200
        assert data["status"] == "pending_approval"
        assert data["review_status"] == "passed"

        # Citations
        assert len(data["citations"]) == 1
        cit = data["citations"][0]
        assert cit["marker"] == 1
        assert cit["verdict"] == "supported"
        assert cit["source_url"] == "https://arxiv.org/abs/2101.00001"

        # Review
        assert data["latest_review"] is not None
        assert data["latest_review"]["quality_score"] == 90
        assert data["latest_review"]["passed"] is True

        # Approval: human approval must be False by default
        assert data["approval"]["is_approved"] is False


@pytest.mark.asyncio
async def test_get_report_not_generated_yet(app_with_db, fake_db):
    """Test GET /api/tasks/{task_id}/report returns 404 when report draft has not been synthesized."""
    task_id = uuid.uuid4()
    async with fake_db.sessionmaker() as session:
        task = ResearchTask(
            id=task_id,
            topic="Pending Report Task",
            status=TaskStatus.PLANNING.value,
            stage=TaskStage.PLANNING.value,
        )
        session.add(task)
        await session.commit()

    transport = ASGITransport(app=app_with_db)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/tasks/{task_id}/report")
        assert response.status_code == 404
        assert "not been generated yet" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_report_task_not_found(app_with_db):
    """Test GET /api/tasks/{task_id}/report returns 404 when task does not exist."""
    non_existent = uuid.uuid4()
    transport = ASGITransport(app=app_with_db)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/tasks/{non_existent}/report")
        assert response.status_code == 404
