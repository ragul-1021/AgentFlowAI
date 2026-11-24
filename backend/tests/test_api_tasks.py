"""Tests for research task creation, retrieval, and cancellation REST endpoints."""

from __future__ import annotations

import uuid
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.enums import TaskStage, TaskStatus
from app.main import create_app
from app.models import ResearchTask
from app.orchestration.orchestrator import ResearchOrchestrator
from app.providers import MockGeminiProvider, MockTavilyProvider
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
def app_with_mocks(test_settings, fake_db):
    app = create_app(test_settings, database=fake_db)
    # Configure mock orchestrator with in-memory double
    orchestrator = ResearchOrchestrator(
        sessionmaker=fake_db.sessionmaker,
        settings=test_settings,
        llm_provider=MockGeminiProvider(default_model="gemini-2.5-flash"),
        search_provider=MockTavilyProvider(),
    )
    app.state.orchestrator = orchestrator
    return app


@pytest.mark.asyncio
async def test_create_task_success(app_with_mocks, fake_db):
    """Test POST /api/tasks queues a task, returns 202, and starts background execution."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "topic": "Evaluating Multi-Agent Consensus Algorithms in Distributed Systems",
            "description": "Comparison of Paxos and Raft under high latency networks",
            "depth": "standard",
            "preferred_domains": ["acm.org", "ieee.org"],
            "excluded_domains": ["unverified-blog.com"],
        }
        response = await client.post("/api/tasks", json=payload)
        assert response.status_code == 202

        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["stage"] == "queued"
        assert data["topic"] == payload["topic"]
        assert data["stream_url"] == f"/api/tasks/{data['task_id']}/stream"
        assert data["status_url"] == f"/api/tasks/{data['task_id']}"

        # Verify task row was created in database
        task_uuid = uuid.UUID(data["task_id"])
        async with fake_db.sessionmaker() as session:
            task = await session.get(ResearchTask, task_uuid)
            assert task is not None
            assert task.topic == payload["topic"]
            assert task.depth == "standard"


@pytest.mark.asyncio
async def test_create_task_validation_error(app_with_mocks):
    """Test POST /api/tasks with overlapping domains or short topic returns 422."""
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Topic too short
        res1 = await client.post("/api/tasks", json={"topic": "Short"})
        assert res1.status_code == 422

        # 2. Overlapping domains
        res2 = await client.post(
            "/api/tasks",
            json={
                "topic": "Comprehensive analysis of distributed consensus mechanisms",
                "preferred_domains": ["arxiv.org"],
                "excluded_domains": ["arxiv.org"],
            },
        )
        assert res2.status_code == 422


@pytest.mark.asyncio
async def test_get_task_status_success(app_with_mocks, fake_db):
    """Test GET /api/tasks/{task_id} returns existing task status."""
    task_id = uuid.uuid4()
    async with fake_db.sessionmaker() as session:
        task = ResearchTask(
            id=task_id,
            topic="Benchmarking LSM Trees vs B-Trees",
            status=TaskStatus.ANALYZING.value,
            stage=TaskStage.ANALYZING.value,
            progress=45,
            current_agent="analyst",
            total_tokens=1500,
            total_calls=3,
            estimated_cost=0,
        )
        session.add(task)
        await session.commit()

    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/tasks/{task_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["task_id"] == str(task_id)
        assert data["status"] == "analyzing"
        assert data["stage"] == "analyzing"
        assert data["progress"] == 45
        assert data["current_agent"] == "analyst"
        assert data["total_tokens"] == 1500


@pytest.mark.asyncio
async def test_get_task_status_not_found(app_with_mocks):
    """Test GET /api/tasks/{task_id} returns 404 for non-existent task."""
    non_existent = uuid.uuid4()
    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/tasks/{non_existent}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_cancel_task_success(app_with_mocks, fake_db):
    """Test POST /api/tasks/{task_id}/cancel cancels an active task."""
    task_id = uuid.uuid4()
    async with fake_db.sessionmaker() as session:
        task = ResearchTask(
            id=task_id,
            topic="Active Task To Cancel",
            status=TaskStatus.RESEARCHING.value,
            stage=TaskStage.RESEARCHING.value,
            progress=20,
        )
        session.add(task)
        await session.commit()

    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/api/tasks/{task_id}/cancel")
        assert response.status_code == 200

        data = response.json()
        assert data["task_id"] == str(task_id)
        assert data["status"] == "cancelled"
        assert data["cancellation_requested"] is True

        # Verify task is updated in DB
        async with fake_db.sessionmaker() as session:
            db_task = await session.get(ResearchTask, task_id)
            assert db_task.status == TaskStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_cancel_task_already_completed(app_with_mocks, fake_db):
    """Test POST /api/tasks/{task_id}/cancel returns appropriate message if already finished."""
    task_id = uuid.uuid4()
    async with fake_db.sessionmaker() as session:
        task = ResearchTask(
            id=task_id,
            topic="Completed Task",
            status=TaskStatus.PENDING_APPROVAL.value,
            stage=TaskStage.COMPLETED.value,
            progress=100,
        )
        session.add(task)
        await session.commit()

    transport = ASGITransport(app=app_with_mocks)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/api/tasks/{task_id}/cancel")
        assert response.status_code == 200

        data = response.json()
        assert data["cancellation_requested"] is False
        assert "already completed" in data["message"].lower()
