from __future__ import annotations

import asyncio
import io
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import Database, prepare_url, sqlstate
from tests.conftest import ALEMBIC_INI, alembic_config

EXPECTED_TABLES = {
    "agent_executions",
    "finding_sources",
    "report_approvals",
    "report_citations",
    "report_reviews",
    "research_findings",
    "research_plans",
    "research_reports",
    "research_sources",
    "research_tasks",
    "run_events",
}


def test_offline_upgrade_renders_all_tables() -> None:
    buffer = io.StringIO()
    config = Config(ALEMBIC_INI, output_buffer=buffer)
    config.set_main_option(
        "sqlalchemy.url",
        "postgresql+asyncpg://agentflow:placeholder@localhost:5432/agentflow",
    )

    command.upgrade(config, "head", sql=True)

    rendered = buffer.getvalue()
    assert "CREATE TABLE research_tasks" in rendered
    for table in EXPECTED_TABLES:
        assert f"CREATE TABLE {table}" in rendered
    assert "fk_report_citations_source_id_research_sources" in rendered
    assert "uq_report_approvals_approved" in rendered


def test_alembic_models_have_no_drift(migrated_database: str) -> None:
    command.check(alembic_config(migrated_database))


async def test_database_reports_head_revision(db_database: Database) -> None:
    probe = await db_database.probe()

    assert probe.status == "up"
    assert probe.revision == "0001"


async def test_expected_tables_exist(db_database: Database) -> None:
    async with db_database.engine.connect() as connection:
        rows = await connection.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        )
    tables = {row[0] for row in rows} - {"alembic_version"}
    assert tables == EXPECTED_TABLES


async def test_upgrade_and_downgrade_on_scratch_database(
    test_database_url: str, db_settings: Settings
) -> None:
    url, connect_args = prepare_url(test_database_url)
    admin_engine = create_async_engine(
        url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        poolclass=NullPool,
        connect_args=connect_args,
    )
    scratch_name = f"agentflow_mig_{uuid4().hex[:10]}"
    scratch_url = url.set(database=scratch_name)

    try:
        async with admin_engine.connect() as connection:
            try:
                await connection.execute(text(f'CREATE DATABASE "{scratch_name}"'))
            except DBAPIError as exc:
                if sqlstate(exc) == "42501":
                    pytest.skip(
                        "test role lacks CREATEDB; scratch-database migration test skipped"
                    )
                raise

        config = alembic_config(scratch_url.render_as_string(hide_password=False))
        await asyncio.to_thread(command.upgrade, config, "head")

        scratch_engine = create_async_engine(
            scratch_url, poolclass=NullPool, connect_args=connect_args
        )
        try:
            async with scratch_engine.connect() as connection:
                count = await connection.scalar(
                    text(
                        "SELECT count(*) FROM information_schema.tables "
                        "WHERE table_schema = 'public' "
                        "AND table_name <> 'alembic_version'"
                    )
                )
                revision = await connection.scalar(
                    text("SELECT version_num FROM alembic_version")
                )
            assert count == len(EXPECTED_TABLES)
            assert revision == "0001"

            await asyncio.to_thread(command.downgrade, config, "base")

            async with scratch_engine.connect() as connection:
                remaining = await connection.scalar(
                    text(
                        "SELECT count(*) FROM information_schema.tables "
                        "WHERE table_schema = 'public' "
                        "AND table_name <> 'alembic_version'"
                    )
                )
            assert remaining == 0
        finally:
            await scratch_engine.dispose()
    finally:
        try:
            async with admin_engine.connect() as connection:
                await connection.execute(
                    text(f'DROP DATABASE IF EXISTS "{scratch_name}" WITH (FORCE)')
                )
        finally:
            await admin_engine.dispose()
