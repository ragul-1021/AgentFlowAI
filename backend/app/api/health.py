"""Health endpoint: process liveness plus database readiness."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from app import __version__
from app.db import DatabaseProbe

router = APIRouter(tags=["health"])


class ProviderStatus(BaseModel):
    gemini: bool
    groq: bool
    tavily: bool


class DatabaseHealth(BaseModel):
    status: Literal["up", "down"]
    revision: str | None
    latency_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    environment: str
    version: str
    mock_mode: bool
    providers: ProviderStatus
    database: DatabaseHealth


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"description": "Database is unavailable"}},
)
async def health(request: Request, response: Response) -> HealthResponse:
    settings = request.app.state.settings
    database = getattr(request.app.state, "database", None)

    if database is None:
        probe = DatabaseProbe(
            status="down", revision=None, latency_ms=0, error_code="not_initialized"
        )
    else:
        probe = await database.probe()

    if probe.status != "up":
        response.status_code = 503

    return HealthResponse(
        status="ok" if probe.status == "up" else "degraded",
        service=settings.app_name,
        environment=settings.app_env,
        version=__version__,
        mock_mode=settings.ai_mock_mode,
        providers=ProviderStatus(**settings.providers_configured),
        database=DatabaseHealth(
            status=probe.status,
            revision=probe.revision,
            latency_ms=probe.latency_ms,
        ),
    )
