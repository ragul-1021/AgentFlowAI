"""Exceptions for the research pipeline and orchestrator."""

from __future__ import annotations

from typing import Any
import uuid


class PipelineError(Exception):
    """Base exception for all pipeline and orchestration failures."""

    def __init__(
        self,
        message: str,
        *,
        task_id: uuid.UUID | str | None = None,
        stage: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.task_id = str(task_id) if task_id else None
        self.stage = stage
        self.details = details or {}

    def __str__(self) -> str:
        ctx = []
        if self.task_id:
            ctx.append(f"task={self.task_id}")
        if self.stage:
            ctx.append(f"stage={self.stage}")
        prefix = f"[{', '.join(ctx)}] " if ctx else ""
        return f"{prefix}{self.message}"


class PipelineStageError(PipelineError):
    """Raised when an error occurs during the execution of a specific pipeline stage."""


class PipelineValidationError(PipelineError):
    """Raised when input parameters or intermediate outputs fail validation."""


class PipelineCancelledError(PipelineError):
    """Raised when pipeline execution is interrupted due to a cancellation request."""


class MaxRevisionsExceededError(PipelineError):
    """Raised when the report fails quality review and exceeds the maximum revision limit."""
