"""Execution context and metadata passed across agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

from app.enums import TaskStage


@dataclass
class AgentContext:
    """Carries execution state, tracing identifiers, and runtime metadata for an agent call."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attempt: int = 1
    stage: TaskStage | str = TaskStage.PLANNING
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def fork(self, new_stage: TaskStage | str, **extra_meta: Any) -> AgentContext:
        """Create a child context for downstream agents in the pipeline."""
        merged_meta = {**self.metadata, **extra_meta}
        return AgentContext(
            task_id=self.task_id,
            attempt=self.attempt,
            stage=new_stage,
            metadata=merged_meta,
            created_at=self.created_at,
        )
