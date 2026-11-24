"""Shared test configuration.

Unit tests never require a network or database. Database tests are opt-in via
``TEST_DATABASE_URL`` and use an isolated database that is migrated with
Alembic; they skip with a clear reason when the variable is absent.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://agentflow:agentflow@localhost:5432/agentflow?ssl=require",
)
os.environ.setdefault("AI_MOCK_MODE", "true")

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import Database, build_engine
from app.main import create_app
from tests.fakes import FakeDatabase

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "").strip()
PRODUCTION_DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
ALEMBIC_INI = os.path.join(os.path.dirname(__file__), "..", "alembic.ini")


def alembic_config(database_url: str) -> Config:
    config = Config(ALEMBIC_INI)
    # configparser treats % specially.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url=(
            TEST_DATABASE_URL
            or "postgresql+asyncpg://agentflow:test-secret@localhost:5432/agentflow"
        ),
        gemini_api_key="test-gemini-key",
        groq_api_key="",
        tavily_api_key="",
        ai_mock_mode=True,
        log_level="WARNING",
    )


@pytest.fixture()
def database() -> FakeDatabase:
    return FakeDatabase()


@pytest.fixture()
async def client(settings: Settings, database: FakeDatabase):
    app = create_app(settings, database=database)  # type: ignore[arg-type]
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://agentflow.test",
    ) as http_client:
        yield http_client


@pytest.fixture(scope="session")
def test_database_url() -> str:
    if not TEST_DATABASE_URL:
        pytest.skip(
            "TEST_DATABASE_URL is not set; start a local PostgreSQL or use a "
            "dedicated Supabase test project, then re-run with the variable set."
        )
    if TEST_DATABASE_URL == PRODUCTION_DATABASE_URL:
        pytest.fail(
            "TEST_DATABASE_URL must not equal DATABASE_URL; refusing to run tests "
            "against the application database."
        )
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def migrated_database(test_database_url: str) -> str:
    """Apply migrations once for the whole test session and assert no drift."""
    config = alembic_config(test_database_url)
    command.upgrade(config, "head")
    command.check(config)
    return test_database_url


@pytest.fixture()
def db_settings(test_database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        database_url=test_database_url,
        ai_mock_mode=True,
        log_level="WARNING",
        db_pool_size=2,
        db_max_overflow=0,
    )


@pytest.fixture()
async def db_database(migrated_database: str, db_settings: Settings) -> Database:
    database = Database(
        db_settings,
        engine=build_engine(db_settings, raw_url=migrated_database, poolclass=NullPool),
    )
    try:
        yield database
    finally:
        await database.dispose()


@pytest.fixture()
async def db_session(db_database: Database):
    """Session bound to an outer transaction rolled back after each test."""
    async with db_database.engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()


@pytest.fixture()
def sessionmaker(db_database: Database) -> async_sessionmaker[AsyncSession]:
    return db_database.sessionmaker
