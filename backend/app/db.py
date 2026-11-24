"""Async database engine, session management, health probing, and retries.

The backend is the only component that receives ``DATABASE_URL``. Connections
use SQLAlchemy 2.0 + asyncpg with TLS, pool pre-ping/recycling, and server-side
statement/idle timeouts.
"""

from __future__ import annotations

import asyncio
import logging
import random
import ssl as ssl_module
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import Pool

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TLS_MODES = {"require", "true", "verify-ca", "verify-full", "1", "yes", "on"}
_TRANSIENT_SQLSTATES = {
    "40001",  # serialization_failure
    "40P01",  # deadlock_detected
    "08000",  # connection_exception
    "08001",  # sqlclient_unable_to_establish_sqlconnection
    "08003",  # connection_does_not_exist
    "08004",  # sqlserver_rejected_establishment_of_sqlconnection
    "08006",  # connection_failure
    "57P01",  # admin_shutdown
    "53300",  # too_many_connections
}


@dataclass(frozen=True)
class DatabaseProbe:
    """Result of a lightweight database connectivity check."""

    status: str  # "up" | "down"
    revision: str | None
    latency_ms: int
    error_code: str | None = None


def prepare_url(raw_url: str) -> tuple[URL, dict[str, Any]]:
    """Split a SQLAlchemy URL into a driver URL and asyncpg connect args.

    ``ssl``/``sslmode`` move from the query string into an SSLContext so the
    driver receives exactly one, correctly typed value.
    """
    url = make_url(raw_url)
    query = dict(url.query)
    ssl_mode = query.pop("ssl", None) or query.pop("sslmode", None)

    if "ssl" in dict(url.query) or "sslmode" in dict(url.query):
        url = url.difference_update_query(["ssl", "sslmode"])

    connect_args: dict[str, Any] = {}
    if ssl_mode:
        mode = str(ssl_mode).strip().lower()
        if mode not in _TLS_MODES:
            raise ValueError(
                "Unsupported TLS mode in DATABASE_URL; use ssl=require "
                "(or sslmode=require)"
            )
        ctx = ssl_module.create_default_context()
        if mode in {"require", "true", "1", "yes", "on"}:
            ctx.check_hostname = False
            ctx.verify_mode = ssl_module.CERT_NONE
        connect_args["ssl"] = ctx
    return url, connect_args


def build_engine(
    settings: Settings,
    *,
    raw_url: str | None = None,
    poolclass: type[Pool] | None = None,
) -> AsyncEngine:
    """Create the async engine without opening a connection."""
    url, connect_args = prepare_url(raw_url or settings.database_url)
    connect_args.setdefault("timeout", settings.db_connect_timeout)
    connect_args.setdefault("command_timeout", settings.db_command_timeout)
    connect_args.setdefault(
        "server_settings",
        {
            "application_name": settings.app_name,
            "statement_timeout": str(settings.db_statement_timeout_ms),
            "idle_in_transaction_session_timeout": str(
                settings.db_idle_in_tx_timeout_ms
            ),
        },
    )

    kwargs: dict[str, Any] = {
        "pool_pre_ping": True,
        "pool_recycle": settings.db_pool_recycle,
        "connect_args": connect_args,
    }
    if poolclass is not None:
        kwargs["poolclass"] = poolclass
    else:
        kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_use_lifo=True,
        )

    return create_async_engine(url, **kwargs)


class Database:
    """Owns the engine and session factory for the application process."""

    def __init__(self, settings: Settings, *, engine: AsyncEngine | None = None) -> None:
        self.settings = settings
        self.engine = engine or build_engine(settings)
        self.sessionmaker = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            autoflush=False,
        )

    async def probe(self) -> DatabaseProbe:
        """Run ``SELECT 1`` and read the Alembic revision. Never raises."""
        started = time.perf_counter()
        timeout = self.settings.db_connect_timeout + 5
        try:
            async with asyncio.timeout(timeout):
                async with self.engine.connect() as connection:
                    await connection.execute(text("SELECT 1"))
                    revision = await self._revision(connection)
        except TimeoutError:
            logger.warning("database probe timed out")
            return DatabaseProbe(
                status="down",
                revision=None,
                latency_ms=_elapsed_ms(started),
                error_code="timeout",
            )
        except Exception as exc:  # noqa: BLE001 - probe must never crash health
            from app.core.redaction import redact

            logger.warning(
                "database probe failed: %s",
                redact(f"{type(exc).__name__}: {exc}", self.settings.redaction_secrets()),
            )
            return DatabaseProbe(
                status="down",
                revision=None,
                latency_ms=_elapsed_ms(started),
                error_code="unavailable",
            )

        return DatabaseProbe(
            status="up",
            revision=revision,
            latency_ms=_elapsed_ms(started),
        )

    async def _revision(self, connection: Any) -> str | None:
        try:
            result = await connection.execute(
                text("SELECT version_num FROM alembic_version LIMIT 1")
            )
        except DBAPIError as exc:
            sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
            if sqlstate == "42P01":  # undefined_table: migrations not applied
                return None
            raise
        return result.scalar_one_or_none()

    async def dispose(self) -> None:
        await self.engine.dispose()


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


@asynccontextmanager
async def session_scope(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Short transaction: commit on success, roll back on error, always close."""
    session = sessionmaker()
    try:
        yield session
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session; callers manage commit/rollback."""
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        try:
            yield session
        except BaseException:
            await session.rollback()
            raise


def sqlstate(exc: BaseException) -> str | None:
    """Extract a PostgreSQL SQLSTATE from SQLAlchemy/asyncpg exceptions."""
    original = getattr(exc, "orig", exc)
    return getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)


def is_transient_db_error(exc: BaseException) -> bool:
    """True for failures that are safe to retry at a transaction boundary."""
    return sqlstate(exc) in _TRANSIENT_SQLSTATES


async def retry_on_transient(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.25,
) -> T:
    """Run an operation, retrying transient DB errors with jittered backoff.

    The operation must be idempotent or perform a single transaction, because a
    failed transaction is rolled back before the retry.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except SQLAlchemyError as exc:
            if attempt >= attempts or not is_transient_db_error(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            await asyncio.sleep(delay + random.uniform(0, base_delay))
    raise AssertionError("unreachable")  # pragma: no cover
