"""Tests for Server-Sent Events (SSE) streaming endpoint."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.enums import EventType, TaskStage, TaskStatus
from app.main import create_app
from app.models import ResearchTask, RunEvent
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
async def test_sse_stream_delivers_events(app_with_db, fake_db):
    """Test GET /api/tasks/{task_id}/stream streams events and terminates cleanly on done."""
    task_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    async with fake_db.sessionmaker() as session:
        task = ResearchTask(
            id=task_id,
            topic="Streaming Test Task",
            status=TaskStatus.PENDING_APPROVAL.value,
            stage=TaskStage.COMPLETED.value,
        )
        session.add(task)

        event1 = RunEvent(
            id=1,
            task_id=task_id,
            type=EventType.STAGE.value,
            stage=TaskStage.PLANNING.value,
            message="Planning stage initiated",
            payload={"progress": 10},
            created_at=now,
        )
        event2 = RunEvent(
            id=2,
            task_id=task_id,
            type=EventType.DONE.value,
            stage=TaskStage.COMPLETED.value,
            message="Research pipeline finished",
            payload={"status": "completed"},
            created_at=now,
        )
        session.add(event1)
        session.add(event2)
        await session.commit()

    transport = ASGITransport(app=app_with_db)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/tasks/{task_id}/stream")
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body_text = response.text
        assert "id: 1" in body_text
        assert "event: stage" in body_text
        assert "id: 2" in body_text
        assert "event: done" in body_text


@pytest.mark.asyncio
async def test_sse_stream_last_event_id_reconnection(app_with_db, fake_db):
    """Test client reconnection with Last-Event-ID only delivers events past the cursor."""
    task_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    async with fake_db.sessionmaker() as session:
        task = ResearchTask(
            id=task_id,
            topic="Reconnection Test Task",
            status=TaskStatus.PENDING_APPROVAL.value,
            stage=TaskStage.COMPLETED.value,
        )
        session.add(task)

        event1 = RunEvent(
            id=10,
            task_id=task_id,
            type=EventType.STAGE.value,
            stage=TaskStage.PLANNING.value,
            message="Already received event",
            payload={},
            created_at=now,
        )
        event2 = RunEvent(
            id=11,
            task_id=task_id,
            type=EventType.STAGE.value,
            stage=TaskStage.RESEARCHING.value,
            message="New event after reconnect",
            payload={},
            created_at=now,
        )
        event3 = RunEvent(
            id=12,
            task_id=task_id,
            type=EventType.DONE.value,
            stage=TaskStage.COMPLETED.value,
            message="Finished",
            payload={},
            created_at=now,
        )
        session.add(event1)
        session.add(event2)
        session.add(event3)
        await session.commit()

    transport = ASGITransport(app=app_with_db)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Client presents Last-Event-ID: 10
        headers = {"Last-Event-ID": "10"}
        response = await client.get(f"/api/tasks/{task_id}/stream", headers=headers)
        assert response.status_code == 200

        body_text = response.text
        # Event 10 should be skipped
        assert "id: 10" not in body_text
        # Events 11 and 12 should be present
        assert "id: 11" in body_text
        assert "id: 12" in body_text


@pytest.mark.asyncio
async def test_sse_stream_task_not_found(app_with_db):
    """Test SSE endpoint returns 404 if task does not exist."""
    non_existent = uuid.uuid4()
    transport = ASGITransport(app=app_with_db)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/tasks/{non_existent}/stream")
        assert response.status_code == 404
