"""FastAPI dependency injection utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING
from fastapi import Request

from app.config import Settings
from app.db import Database

if TYPE_CHECKING:
    from app.orchestration.orchestrator import ResearchOrchestrator


def get_settings(request: Request) -> Settings:
    """Retrieve application settings from application state."""
    return request.app.state.settings


def get_database(request: Request) -> Database:
    """Retrieve database manager from application state."""
    return request.app.state.database


def get_orchestrator(request: Request) -> ResearchOrchestrator:
    """Retrieve research orchestrator singleton from application state."""
    return request.app.state.orchestrator
