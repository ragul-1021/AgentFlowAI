from __future__ import annotations

import asyncio
import ssl
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import (
    Database,
    build_engine,
    get_session,
    is_transient_db_error,
    prepare_url,
    retry_on_transient,
    session_scope,
    sqlstate,
)
from app.models import ResearchTask

REMOTE_URL = "postgresql+asyncpg://user:secret@db.example.com:5432/postgres?ssl=require"


class FakeDBError(SQLAlchemyError):
    def __init__(self, code: str) -> None:
        super().__init__("database error")
        self.orig = SimpleNamespace(sqlstate=code)


def test_prepare_url_moves_ssl_to_connect_args() -> None:
    url, connect_args = prepare_url(REMOTE_URL)

    assert "ssl" not in url.query
    assert isinstance(connect_args["ssl"], ssl.SSLContext)


def test_prepare_url_accepts_sslmode() -> None:
    url, connect_args = prepare_url(
        "postgresql+asyncpg://user:secret@db.example.com:5432/postgres"
        "?sslmode=verify-full"
    )

    assert "sslmode" not in url.query
    assert "ssl" in connect_args


def test_prepare_url_without_tls() -> None:
    url, connect_args = prepare_url(
        "postgresql+asyncpg://agentflow@127.0.0.1:55432/agentflow_test"
    )

    assert url.host == "127.0.0.1"
    assert connect_args == {}


def test_prepare_url_rejects_unknown_ssl_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported TLS mode"):
        prepare_url(
            "postgresql+asyncpg://user:secret@db.example.com:5432/postgres?ssl=banana"
        )


@pytest.mark.asyncio
async def test_engine_uses_configured_pool(settings: Settings) -> None:
    configured = settings.model_copy(update={"db_pool_size": 3, "db_max_overflow": 2})
    engine = build_engine(configured, poolclass=None)
    try:
        assert engine.pool.size() == 3
        assert engine.pool._max_overflow == 2
    finally:
        await engine.dispose()


def test_sqlstate_reads_asyncpg_style_errors() -> None:
    assert sqlstate(FakeDBError("40001")) == "40001"
    assert sqlstate(SimpleNamespace(orig=SimpleNamespace(pgcode="23505"))) == "23505"
    assert sqlstate(SimpleNamespace()) is None


def test_transient_error_detection() -> None:
    assert is_transient_db_error(FakeDBError("40001"))
    assert is_transient_db_error(FakeDBError("40P01"))
    assert is_transient_db_error(FakeDBError("08006"))
    assert not is_transient_db_error(FakeDBError("23505"))
    assert not is_transient_db_error(FakeDBError("42501"))


@pytest.mark.asyncio
async def test_retry_on_transient_succeeds_after_failures() -> None:
    attempts = {"count": 0}

    async def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeDBError("40001")
        return "ok"

    assert await retry_on_transient(operation, base_delay=0.001) == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_retry_on_transient_gives_up() -> None:
    attempts = {"count": 0}

    async def operation() -> str:
        attempts["count"] += 1
        raise FakeDBError("40001")

    with pytest.raises(FakeDBError):
        await retry_on_transient(operation, attempts=2, base_delay=0.001)
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_retry_does_not_retry_non_transient_error() -> None:
    attempts = {"count": 0}

    async def operation() -> None:
        attempts["count"] += 1
        raise FakeDBError("23505")

    with pytest.raises(FakeDBError):
        await retry_on_transient(operation, base_delay=0.001)
    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_probe_reports_up_and_revision(db_database: Database) -> None:
    probe = await db_database.probe()

    assert probe.status == "up"
    assert probe.revision == "0001"
    assert probe.latency_ms >= 0


@pytest.mark.asyncio
async def test_probe_reports_down_without_raising(settings: Settings) -> None:
    down_settings = settings.model_copy(
        update={
            "database_url": "postgresql+asyncpg://agentflow@127.0.0.1:1/agentflow",
            "db_connect_timeout": 1,
            "db_command_timeout": 1,
        }
    )
    database = Database(down_settings, engine=build_engine(down_settings, poolclass=NullPool))
    try:
        probe = await database.probe()
    finally:
        await database.dispose()

    assert probe.status == "down"
    assert probe.revision is None
    assert probe.error_code in {"unavailable", "timeout"}


@pytest.mark.asyncio
async def test_session_scope_commits_and_rolls_back(db_database: Database, db_session) -> None:
    maker: async_sessionmaker = async_sessionmaker(
        bind=db_database.engine, expire_on_commit=False, autoflush=False
    )
    topic = "session scope commit and rollback test"
    try:
        async with session_scope(maker) as session:
            session.add(ResearchTask(topic=topic))

        async with session_scope(maker) as session:
            committed = await session.scalar(
                select(func.count()).select_from(ResearchTask).where(
                    ResearchTask.topic == topic
                )
            )
        assert committed == 1

        with pytest.raises(RuntimeError):
            async with session_scope(maker) as session:
                session.add(ResearchTask(topic="session scope rollback test"))
                raise RuntimeError("boom")

        async with session_scope(maker) as session:
            rolled_back = await session.scalar(
                select(func.count()).select_from(ResearchTask).where(
                    ResearchTask.topic == "session scope rollback test"
                )
            )
        assert rolled_back == 0
    finally:
        async with session_scope(maker) as session:
            await session.execute(
                text("DELETE FROM research_tasks WHERE topic = :topic"), {"topic": topic}
            )


@pytest.mark.asyncio
async def test_session_scope_lifecycle_with_mock() -> None:
    from unittest.mock import AsyncMock, MagicMock

    mock_session = AsyncMock()
    mock_maker = MagicMock(return_value=mock_session)

    # Success case: commits and closes
    async with session_scope(mock_maker) as s:
        assert s is mock_session
    mock_session.commit.assert_awaited_once()
    mock_session.close.assert_awaited_once()

    # Error case: rolls back and closes
    mock_session.reset_mock()
    with pytest.raises(ValueError):
        async with session_scope(mock_maker) as s:
            raise ValueError("simulated error")
    mock_session.rollback.assert_awaited_once()
    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_session_lifecycle_with_mock() -> None:
    from unittest.mock import AsyncMock, MagicMock

    mock_session = AsyncMock()
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_session
    mock_context.__aexit__.return_value = None

    mock_db = MagicMock()
    mock_db.sessionmaker.return_value = mock_context

    mock_request = MagicMock()
    mock_request.app.state.database = mock_db

    # Clean yield
    generator = get_session(mock_request)
    session = await anext(generator)
    assert session is mock_session
    with pytest.raises(StopAsyncIteration):
        await anext(generator)

    # Rollback on exception thrown into generator
    generator = get_session(mock_request)
    session = await anext(generator)
    assert session is mock_session
    with pytest.raises(RuntimeError):
        await generator.athrow(RuntimeError("pipeline failure"))
    mock_session.rollback.assert_awaited_once()

