"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pydantic import ValidationError

from app import __version__
from app.api.health import router as health_router
from app.api.tasks import router as tasks_router
from app.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db import Database
from app.orchestration.orchestrator import ResearchOrchestrator
from app.providers.factory import (
    create_gemini_provider,
    create_groq_provider,
    create_tavily_provider,
)
from app.security import configure_security

logger = logging.getLogger(__name__)


def _format_validation_error(exc: ValidationError) -> str:
    """Summarize validation errors without echoing secret input values."""
    details: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "settings"
        details.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "; ".join(details)


def load_settings() -> Settings:
    try:
        return get_settings()
    except ValidationError as exc:
        raise RuntimeError(
            "AgentFlow configuration is invalid. Copy backend/.env.example to "
            "backend/.env and set DATABASE_URL to a postgresql+asyncpg:// URL "
            f"(TLS required for remote hosts). Details: {_format_validation_error(exc)}"
        ) from exc


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level, secrets=settings.redaction_secrets())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db: Database = app.state.database
        client: httpx.AsyncClient | None = app.state.http_client
        owns_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=30.0)
            app.state.http_client = client
            owns_client = True

        app.state.gemini_provider = create_gemini_provider(settings, client=client)
        app.state.groq_provider = create_groq_provider(settings, client=client)
        app.state.tavily_provider = create_tavily_provider(settings, client=client)

        if not hasattr(app.state, "orchestrator") or app.state.orchestrator is None:
            app.state.orchestrator = ResearchOrchestrator(
                sessionmaker=db.sessionmaker,
                settings=settings,
                llm_provider=app.state.gemini_provider,
                search_provider=app.state.tavily_provider,
                http_client=client,
            )

        logger.info(
            "AgentFlow API starting",
            extra={
                "environment": settings.app_env,
                "mock_mode": settings.ai_mock_mode,
            },
        )
        probe = await db.probe()
        if probe.status == "up":
            logger.info(
                "database ready",
                extra={"stage": "startup"},
            )
        else:
            # Do not kill the process: /api/health must be able to report the
            # failure, and a paused Supabase project can recover without a
            # restart. Configuration errors fail fast before this point.
            logger.error(
                "database unavailable at startup (%s); /api/health will report 503",
                probe.error_code,
                extra={"stage": "startup"},
            )
        yield
        if owns_client and client is not None:
            await client.aclose()
        await db.dispose()
        logger.info("AgentFlow API stopped")

    app = FastAPI(
        title="AgentFlow AI API",
        version=__version__,
        description="Source-grounded multi-agent research API.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database or Database(settings)
    app.state.http_client = http_client
    app.state.orchestrator = None
    configure_security(app, settings)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "AgentFlow AI API",
            "health": "/api/health",
            "docs": "/docs",
        }

    app.include_router(health_router, prefix="/api")
    app.include_router(tasks_router, prefix="/api/tasks")
    return app


app = create_app()
