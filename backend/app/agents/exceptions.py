"""Exceptions for the multi-agent system."""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base exception for all agent-related failures."""

    def __init__(
        self,
        message: str,
        *,
        agent_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.agent_name = agent_name
        self.details = details or {}

    def __str__(self) -> str:
        prefix = f"[{self.agent_name}] " if self.agent_name else ""
        return f"{prefix}{self.message}"


class AgentValidationError(AgentError):
    """Raised when an agent receives invalid input or produces malformed output."""


class AgentExecutionError(AgentError):
    """Raised when an agent fails to complete its assigned workflow."""


class AgentProviderError(AgentError):
    """Raised when an underlying AI provider fails during an agent run."""

    def __init__(
        self,
        message: str,
        *,
        agent_name: str | None = None,
        provider: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, agent_name=agent_name, details=details)
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
