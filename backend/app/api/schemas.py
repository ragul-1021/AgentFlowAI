"""Pydantic request and response schemas for AgentFlow REST API and SSE events."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.enums import (
    ApprovalDecision,
    CitationVerdict,
    ReportFormat,
    ReportLength,
    ReportStatus,
    ResearchDepth,
    ReviewStatus,
    TaskStage,
    TaskStatus,
)


class TaskCreateResponse(BaseModel):
    """Initial response returned immediately when a research task is queued."""

    model_config = ConfigDict(extra="forbid")

    task_id: uuid.UUID
    status: TaskStatus
    stage: TaskStage
    topic: str
    created_at: datetime
    status_url: str
    stream_url: str


class TaskStatusResponse(BaseModel):
    """Detailed lifecycle status and resource consumption metrics for a research task."""

    model_config = ConfigDict(extra="forbid")

    task_id: uuid.UUID
    topic: str
    status: TaskStatus
    stage: TaskStage
    progress: int = Field(ge=0, le=100)
    current_agent: str | None = None
    revision_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_calls: int = Field(ge=0)
    estimated_cost: Decimal
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None


class ReportSectionItem(BaseModel):
    """Section or chapter in the synthesized report."""

    model_config = ConfigDict(extra="forbid")

    heading: str
    content: str
    subsections: list[ReportSectionItem] = Field(default_factory=list)


class ReportCitationItem(BaseModel):
    """In-text citation linked to external research evidence."""

    model_config = ConfigDict(extra="forbid")

    marker: int
    source_url: str | None = None
    source_title: str | None = None
    claim_text: str
    verdict: CitationVerdict
    review_note: str | None = None


class ReportReviewSummary(BaseModel):
    """Audit evaluation score and feedback from the Reviewer agent."""

    model_config = ConfigDict(extra="forbid")

    quality_score: int
    passed: bool
    citation_coverage: float
    unsupported_count: int
    feedback: str


class ReportApprovalSummary(BaseModel):
    """Human approval record if the report has undergone review."""

    model_config = ConfigDict(extra="forbid")

    is_approved: bool = False
    decision: ApprovalDecision | None = None
    actor: str | None = None
    approved_at: datetime | None = None
    notes: str | None = None


class ReportDetailResponse(BaseModel):
    """Full research report payload with citations, audit review, and approval state."""

    model_config = ConfigDict(extra="forbid")

    report_id: uuid.UUID
    task_id: uuid.UUID
    version: int
    title: str
    executive_summary: str
    content: list[dict[str, Any]]
    word_count: int
    status: ReportStatus
    review_status: ReviewStatus
    created_at: datetime
    updated_at: datetime
    approved_at: datetime | None = None
    citations: list[ReportCitationItem] = Field(default_factory=list)
    latest_review: ReportReviewSummary | None = None
    approval: ReportApprovalSummary


class TaskCancelResponse(BaseModel):
    """Response returned upon task cancellation request."""

    model_config = ConfigDict(extra="forbid")

    task_id: uuid.UUID
    status: TaskStatus
    cancellation_requested: bool
    message: str


class SSEEventData(BaseModel):
    """Standard payload structure for streamed progress events."""

    model_config = ConfigDict(extra="ignore")

    task_id: uuid.UUID
    type: str
    stage: str | None = None
    agent: str | None = None
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
