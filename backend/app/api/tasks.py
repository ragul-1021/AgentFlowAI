"""REST API endpoints and SSE stream for research tasks and reports."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import selectinload

from app.api.deps import get_database, get_orchestrator, get_settings
from app.api.schemas import (
    ReportApprovalSummary,
    ReportCitationItem,
    ReportDetailResponse,
    ReportReviewSummary,
    TaskCancelResponse,
    TaskCreateResponse,
    TaskStatusResponse,
)
from app.config import Settings
from app.core.redaction import redact
from app.db import Database, session_scope
from app.enums import (
    ApprovalDecision,
    CitationVerdict,
    EventType,
    ReportStatus,
    ReviewStatus,
    TaskStage,
    TaskStatus,
)
from app.models import (
    ReportApproval,
    ReportCitation,
    ReportReview,
    ResearchReport,
    ResearchSource,
    ResearchTask,
    RunEvent,
)
from app.orchestration.orchestrator import ResearchOrchestrator
from app.schemas import TaskCreate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])


async def _run_task_background(
    orchestrator: ResearchOrchestrator,
    request: TaskCreate,
    task_id: uuid.UUID,
    sessionmaker,
    settings: Settings,
) -> None:
    """Execute pipeline in background, ensuring task status is updated if fatal crash occurs."""
    try:
        await orchestrator.run(request, task_id=task_id)
    except Exception as exc:
        safe_msg = redact(str(exc), settings.redaction_secrets())
        logger.exception("[api.tasks] Background execution for task %s failed: %s", task_id, safe_msg)
        async with session_scope(sessionmaker) as session:
            task = await session.get(ResearchTask, task_id)
            if task and task.status not in (TaskStatus.FAILED.value, TaskStatus.CANCELLED.value):
                task.status = TaskStatus.FAILED.value
                task.stage = TaskStage.FAILED.value
                task.error_code = exc.__class__.__name__
                task.error_message = safe_msg
                event = RunEvent(
                    task_id=task_id,
                    type=EventType.ERROR.value,
                    stage=TaskStage.FAILED.value,
                    message="Background execution failed unexpectedly",
                    payload={"error": safe_msg},
                )
                session.add(event)


@router.post(
    "",
    response_model=TaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a new research task",
)
async def create_task(
    payload: TaskCreate,
    database: Database = Depends(get_database),
    orchestrator: ResearchOrchestrator = Depends(get_orchestrator),
    settings: Settings = Depends(get_settings),
) -> TaskCreateResponse:
    """Create a new research task, persist initial row, and schedule pipeline execution."""
    task_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    async with session_scope(database.sessionmaker) as session:
        task = ResearchTask(
            id=task_id,
            topic=payload.topic,
            description=payload.description,
            depth=payload.depth.value,
            source_target=payload.resolved_source_target,
            format=payload.format.value,
            length=payload.length.value,
            preferred_domains=payload.preferred_domains,
            excluded_domains=payload.excluded_domains,
            status=TaskStatus.QUEUED.value,
            stage=TaskStage.QUEUED.value,
            progress=0,
            started_at=now,
        )
        session.add(task)

        event = RunEvent(
            task_id=task_id,
            type=EventType.STAGE.value,
            stage=TaskStage.QUEUED.value,
            message="Task received and queued for execution",
            payload={"topic": payload.topic, "depth": payload.depth.value},
        )
        session.add(event)

    # Launch pipeline background execution
    asyncio.create_task(
        _run_task_background(
            orchestrator=orchestrator,
            request=payload,
            task_id=task_id,
            sessionmaker=database.sessionmaker,
            settings=settings,
        )
    )

    return TaskCreateResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        stage=TaskStage.QUEUED,
        topic=payload.topic,
        created_at=now,
        status_url=f"/api/tasks/{task_id}",
        stream_url=f"/api/tasks/{task_id}/stream",
    )


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get status and progress of a research task",
)
async def get_task_status(
    task_id: uuid.UUID,
    database: Database = Depends(get_database),
) -> TaskStatusResponse:
    """Retrieve task status, current active stage, progress, and resource consumption."""
    async with session_scope(database.sessionmaker) as session:
        task = await session.get(ResearchTask, task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research task '{task_id}' not found.",
            )

        now = datetime.now(timezone.utc)
        return TaskStatusResponse(
            task_id=task.id,
            topic=task.topic,
            status=TaskStatus(task.status),
            stage=TaskStage(task.stage),
            progress=task.progress or 0,
            current_agent=task.current_agent,
            revision_count=task.revision_count or 0,
            total_tokens=task.total_tokens or 0,
            total_calls=task.total_calls or 0,
            estimated_cost=task.estimated_cost or 0,
            created_at=task.created_at or now,
            updated_at=task.updated_at or now,
            started_at=task.started_at,
            completed_at=task.completed_at,
            error_code=task.error_code,
            error_message=task.error_message,
        )


@router.post(
    "/{task_id}/cancel",
    response_model=TaskCancelResponse,
    summary="Cancel a running research task",
)
async def cancel_task(
    task_id: uuid.UUID,
    database: Database = Depends(get_database),
    orchestrator: ResearchOrchestrator = Depends(get_orchestrator),
) -> TaskCancelResponse:
    """Request cooperative cancellation of a research task."""
    async with session_scope(database.sessionmaker) as session:
        task = await session.get(ResearchTask, task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research task '{task_id}' not found.",
            )

        current_status = TaskStatus(task.status)
        if current_status in (TaskStatus.PENDING_APPROVAL, TaskStatus.APPROVED, TaskStatus.REJECTED):
            return TaskCancelResponse(
                task_id=task_id,
                status=current_status,
                cancellation_requested=False,
                message=f"Task already completed with status '{current_status.value}'. Cannot cancel.",
            )

        if current_status in (TaskStatus.CANCELLED, TaskStatus.FAILED):
            return TaskCancelResponse(
                task_id=task_id,
                status=current_status,
                cancellation_requested=False,
                message=f"Task is already {current_status.value}.",
            )

        # Trigger cancellation in active orchestrator state
        signaled = orchestrator.cancel(task_id)

        # If not active in memory (e.g. still in queued state), update database directly
        if not signaled:
            task.status = TaskStatus.CANCELLED.value
            task.stage = TaskStage.CANCELLED.value
            event = RunEvent(
                task_id=task_id,
                type=EventType.STAGE.value,
                stage=TaskStage.CANCELLED.value,
                message="Task cancelled by user request",
                payload={"status": "cancelled"},
            )
            session.add(event)

        return TaskCancelResponse(
            task_id=task_id,
            status=TaskStatus.CANCELLED,
            cancellation_requested=True,
            message="Cancellation signal delivered to pipeline.",
        )


@router.get(
    "/{task_id}/report",
    response_model=ReportDetailResponse,
    summary="Get generated research report",
)
async def get_task_report(
    task_id: uuid.UUID,
    version: int | None = Query(default=None, description="Report revision version number (defaults to latest)"),
    database: Database = Depends(get_database),
) -> ReportDetailResponse:
    """Retrieve full synthesized report, verified citations, reviewer evaluation, and approval state."""
    async with session_scope(database.sessionmaker) as session:
        # Check task existence
        task = await session.get(ResearchTask, task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research task '{task_id}' not found.",
            )

        # Query report with relationships
        stmt = (
            select(ResearchReport)
            .where(ResearchReport.task_id == task_id)
            .options(
                selectinload(ResearchReport.citations).selectinload(ReportCitation.source),
                selectinload(ResearchReport.reviews),
                selectinload(ResearchReport.approvals),
            )
        )
        if version is not None:
            stmt = stmt.where(ResearchReport.version == version)
        else:
            stmt = stmt.order_by(desc(ResearchReport.version)).limit(1)

        result = await session.execute(stmt)
        report = result.scalars().first()

        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report for task '{task_id}' has not been generated yet.",
            )

        # Build citations list
        citation_items: list[ReportCitationItem] = []
        for cit in sorted(report.citations, key=lambda c: c.marker):
            citation_items.append(
                ReportCitationItem(
                    marker=cit.marker,
                    source_url=cit.source.url if cit.source else None,
                    source_title=cit.source.title if cit.source else None,
                    claim_text=cit.claim_text,
                    verdict=CitationVerdict(cit.verdict),
                    review_note=cit.review_note,
                )
            )

        # Build latest review summary
        latest_review_summary: ReportReviewSummary | None = None
        if report.reviews:
            latest_rev = max(report.reviews, key=lambda r: r.created_at)
            latest_review_summary = ReportReviewSummary(
                quality_score=latest_rev.quality_score,
                passed=latest_rev.passed,
                citation_coverage=latest_rev.citation_coverage,
                unsupported_count=latest_rev.unsupported_count,
                feedback=latest_rev.feedback,
            )

        # Build human approval summary
        approval_summary = ReportApprovalSummary(is_approved=False)
        if report.approvals:
            approved_records = [a for a in report.approvals if a.decision == ApprovalDecision.APPROVED.value]
            if approved_records:
                latest_app = approved_records[-1]
                approval_summary = ReportApprovalSummary(
                    is_approved=True,
                    decision=ApprovalDecision(latest_app.decision),
                    actor=latest_app.actor,
                    approved_at=latest_app.created_at,
                    notes=latest_app.notes,
                )

        now = datetime.now(timezone.utc)
        return ReportDetailResponse(
            report_id=report.id,
            task_id=task_id,
            version=report.version,
            title=report.title,
            executive_summary=report.executive_summary,
            content=report.content if isinstance(report.content, list) else [],
            word_count=report.word_count,
            status=ReportStatus(report.status),
            review_status=ReviewStatus(report.review_status),
            created_at=report.created_at or now,
            updated_at=report.updated_at or now,
            approved_at=report.approved_at,
            citations=citation_items,
            latest_review=latest_review_summary,
            approval=approval_summary,
        )


@router.get(
    "/{task_id}/stream",
    summary="Stream real-time task progress via Server-Sent Events (SSE)",
)
async def stream_task_events(
    task_id: uuid.UUID,
    request: Request,
    database: Database = Depends(get_database),
) -> StreamingResponse:
    """Stream Server-Sent Events (SSE) tracking lifecycle events, stage transitions, and agent runs."""
    async with session_scope(database.sessionmaker) as session:
        task = await session.get(ResearchTask, task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research task '{task_id}' not found.",
            )

    # Determine starting cursor from Last-Event-ID header if client reconnected
    last_event_id_header = request.headers.get("Last-Event-ID")
    initial_last_id = 0
    if last_event_id_header and last_event_id_header.isdigit():
        initial_last_id = int(last_event_id_header)

    async def event_generator() -> AsyncGenerator[str, None]:
        last_id = initial_last_id
        heartbeat_interval = 15.0
        poll_interval = 0.5
        time_since_heartbeat = 0.0

        while True:
            # Check client disconnection
            if await request.is_disconnected():
                logger.info("[sse] Client disconnected from task %s stream", task_id)
                break

            events_dispatched = 0
            is_terminal = False

            # Poll batch of new events with short-lived session
            async with session_scope(database.sessionmaker) as session:
                stmt = (
                    select(RunEvent)
                    .where(RunEvent.task_id == task_id, RunEvent.id > last_id)
                    .order_by(RunEvent.id.asc())
                    .limit(50)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()

                for row in rows:
                    last_id = row.id
                    events_dispatched += 1

                    data_payload = {
                        "task_id": str(task_id),
                        "type": row.type,
                        "stage": row.stage,
                        "agent": row.agent,
                        "message": row.message,
                        "payload": row.payload or {},
                        "created_at": row.created_at.isoformat(),
                    }

                    yield f"id: {row.id}\nevent: {row.type}\ndata: {json.dumps(data_payload)}\n\n"

                    if row.type == EventType.DONE.value or row.stage in (
                        TaskStage.COMPLETED.value,
                        TaskStage.FAILED.value,
                        TaskStage.CANCELLED.value,
                    ):
                        is_terminal = True

            if is_terminal:
                logger.info("[sse] Terminal event reached for task %s, closing stream cleanly", task_id)
                break

            if events_dispatched > 0:
                time_since_heartbeat = 0.0
                await asyncio.sleep(0.05)
                continue

            # No new events: sleep and emit heartbeat periodically
            await asyncio.sleep(poll_interval)
            time_since_heartbeat += poll_interval

            if time_since_heartbeat >= heartbeat_interval:
                yield ": heartbeat\n\n"
                time_since_heartbeat = 0.0

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
