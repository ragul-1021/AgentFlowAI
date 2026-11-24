"""Pipeline state management and stage transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
import uuid

from app.agents.schemas import (
    AnalystOutput,
    PlannerOutput,
    ResearcherOutput,
    ReviewerOutput,
    WriterOutput,
)
from app.enums import TaskStage, TaskStatus


@dataclass
class PipelineState:
    """Carries the end-to-end mutable runtime state of a research task."""

    task_id: uuid.UUID
    current_stage: TaskStage = TaskStage.QUEUED
    task_status: TaskStatus = TaskStatus.QUEUED
    progress: int = 0  # 0 to 100

    # Intermediate agent outputs
    planner_output: PlannerOutput | None = None
    researcher_output: ResearcherOutput | None = None
    analyst_output: AnalystOutput | None = None
    writer_output: WriterOutput | None = None
    reviewer_output: ReviewerOutput | None = None

    # Telemetry and bounds
    retry_count: int = 0
    revision_count: int = 0
    max_revisions: int = 2
    total_tokens: int = 0
    total_calls: int = 0
    estimated_cost: Decimal = field(default_factory=lambda: Decimal("0.000000"))

    # Timing and cancellation
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error_message: str | None = None
    error_code: str | None = None
    cancellation_requested: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition_to(
        self,
        stage: TaskStage,
        status: TaskStatus,
        progress: int | None = None,
    ) -> None:
        """Apply a validated state and progress transition."""
        self.current_stage = stage
        self.task_status = status
        if progress is not None:
            self.progress = max(0, min(100, progress))

    def request_cancellation(self) -> None:
        """Flag the pipeline to halt cleanly before beginning the next stage."""
        self.cancellation_requested = True

    def check_cancellation(self) -> None:
        """Raise PipelineCancelledError if cancellation was requested."""
        if self.cancellation_requested:
            from app.orchestration.exceptions import PipelineCancelledError

            raise PipelineCancelledError(
                f"Pipeline execution for task {self.task_id} was cancelled.",
                task_id=self.task_id,
                stage=str(self.current_stage),
            )

    def accumulate_usage(self, prompt_tokens: int, completion_tokens: int, cost: Decimal) -> None:
        """Increment aggregate telemetry counters."""
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_calls += 1
        self.estimated_cost += cost
