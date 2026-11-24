"""Core research pipeline execution with database persistence and revision cycles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import logging
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents import (
    AgentContext,
    AnalystAgent,
    AnalystInput,
    PlannerAgent,
    PlannerInput,
    ResearcherAgent,
    ResearcherInput,
    ReviewerAgent,
    ReviewerInput,
    WriterAgent,
    WriterInput,
)
from app.config import Settings, get_settings
from app.core.redaction import redact
from app.core.tokens import TokenUsage
from app.db import session_scope
from app.enums import (
    AgentName,
    CitationVerdict,
    EventType,
    ExecutionStatus,
    ReportFormat,
    ReportLength,
    ReportStatus,
    ResearchDepth,
    ReviewStatus,
    TaskStage,
    TaskStatus,
)
from app.models import (
    AgentExecution,
    FindingSource,
    ReportCitation,
    ReportReview,
    ResearchFinding,
    ResearchPlan,
    ResearchReport,
    ResearchSource,
    ResearchTask,
    RunEvent,
)
from app.orchestration.exceptions import (
    PipelineCancelledError,
    PipelineError,
    PipelineStageError,
    PipelineValidationError,
)
from app.orchestration.retry import execute_with_pipeline_retry
from app.orchestration.state import PipelineState
from app.schemas import TaskCreate

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Consolidated result returned after a pipeline execution finishes or pauses for approval."""

    task_id: uuid.UUID
    status: TaskStatus
    stage: TaskStage
    state: PipelineState
    report_id: uuid.UUID | None = None
    report_title: str = ""
    review_score: int | None = None
    approval_recommendation: bool = False
    human_approval_required: bool = True
    total_tokens: int = 0
    estimated_cost: Decimal = Decimal("0.000000")
    error: str | None = None


class ResearchPipeline:
    """Coordinates the 7-step multi-agent research pipeline with Supabase PostgreSQL persistence."""

    def __init__(
        self,
        planner_agent: PlannerAgent,
        researcher_agent: ResearcherAgent,
        analyst_agent: AnalystAgent,
        writer_agent: WriterAgent,
        reviewer_agent: ReviewerAgent,
        sessionmaker: async_sessionmaker[AsyncSession],
        settings: Settings | None = None,
    ) -> None:
        self.planner = planner_agent
        self.researcher = researcher_agent
        self.analyst = analyst_agent
        self.writer = writer_agent
        self.reviewer = reviewer_agent
        self.sessionmaker = sessionmaker
        self.settings = settings or get_settings()

    async def _emit_event(
        self,
        session: AsyncSession,
        task_id: uuid.UUID,
        event_type: EventType,
        stage: TaskStage,
        agent: AgentName | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record an immutable lifecycle event in the run_events table."""
        event = RunEvent(
            task_id=task_id,
            type=event_type.value,
            stage=stage.value,
            agent=agent.value if agent else None,
            message="",
            payload=payload or {},
        )
        session.add(event)

    async def _record_execution(
        self,
        session: AsyncSession,
        task_id: uuid.UUID,
        stage: TaskStage,
        agent: AgentName,
        provider: str,
        model: str,
        status: ExecutionStatus,
        started_at: datetime,
        finished_at: datetime,
        latency_ms: int,
        usage: TokenUsage,
        output_excerpt: str | None = None,
        error: str | None = None,
    ) -> None:
        """Persist an agent execution record."""
        exec_row = AgentExecution(
            task_id=task_id,
            stage=stage.value,
            agent=agent.value,
            provider=provider,
            model=model,
            status=status.value,
            attempt=1,
            started_at=started_at,
            finished_at=finished_at,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_estimate=usage.cost_estimate,
            output_excerpt=output_excerpt[:500] if output_excerpt else None,
            error=redact(error, self.settings.redaction_secrets()) if error else None,
        )
        session.add(exec_row)

    async def execute(
        self,
        request: TaskCreate,
        state: PipelineState | None = None,
    ) -> PipelineResult:
        """Run the complete 7-step research pipeline."""
        task_id = state.task_id if state else uuid.uuid4()
        pipeline_state = state or PipelineState(task_id=task_id)

        try:
            # -----------------------------------------------------------------
            # STEP 1: CREATE TASK
            # -----------------------------------------------------------------
            async with session_scope(self.sessionmaker) as session:
                task_row = ResearchTask(
                    id=task_id,
                    topic=request.topic,
                    description=request.description,
                    depth=request.depth.value,
                    source_target=request.resolved_source_target,
                    format=request.format.value,
                    length=request.length.value,
                    preferred_domains=request.preferred_domains,
                    excluded_domains=request.excluded_domains,
                    status=TaskStatus.QUEUED.value,
                    stage=TaskStage.QUEUED.value,
                    progress=0,
                    started_at=datetime.now(timezone.utc),
                )
                session.add(task_row)
                await self._emit_event(
                    session,
                    task_id,
                    EventType.STAGE,
                    TaskStage.QUEUED,
                    payload={"topic": request.topic, "depth": request.depth.value},
                )

            pipeline_state.transition_to(TaskStage.QUEUED, TaskStatus.QUEUED, progress=0)
            pipeline_state.check_cancellation()

            # -----------------------------------------------------------------
            # STEP 2: PLANNER
            # -----------------------------------------------------------------
            pipeline_state.transition_to(TaskStage.PLANNING, TaskStatus.PLANNING, progress=15)
            async with session_scope(self.sessionmaker) as session:
                task = await session.get(ResearchTask, task_id)
                if task:
                    task.status = TaskStatus.PLANNING.value
                    task.stage = TaskStage.PLANNING.value
                    task.current_agent = AgentName.PLANNER.value
                    task.progress = 15
                await self._emit_event(session, task_id, EventType.AGENT, TaskStage.PLANNING, AgentName.PLANNER)

            async def _run_planner():
                p_in = PlannerInput(
                    topic=request.topic,
                    objective=request.description or request.topic,
                    depth=request.depth,
                    max_questions=5 if request.depth == ResearchDepth.QUICK else 8,
                    max_queries=6 if request.depth == ResearchDepth.QUICK else 10,
                )
                return await self.planner.run(p_in, context=AgentContext(task_id=str(task_id), stage=TaskStage.PLANNING))

            plan_res = await execute_with_pipeline_retry(
                _run_planner,
                stage_name="planner",
                max_attempts=self.settings.retry_max_attempts,
                base_delay=self.settings.retry_base_delay,
                secrets=self.settings.redaction_secrets(),
            )
            pipeline_state.planner_output = plan_res.data
            pipeline_state.accumulate_usage(plan_res.usage.prompt_tokens, plan_res.usage.completion_tokens, plan_res.cost_estimate)

            async with session_scope(self.sessionmaker) as session:
                await self._record_execution(
                    session,
                    task_id,
                    TaskStage.PLANNING,
                    AgentName.PLANNER,
                    plan_res.provider,
                    plan_res.model,
                    ExecutionStatus.SUCCEEDED,
                    plan_res.started_at,
                    plan_res.finished_at,
                    plan_res.latency_ms,
                    plan_res.usage,
                    output_excerpt=plan_res.data.refined_objective,
                )
                plan_row = ResearchPlan(
                    task_id=task_id,
                    objective=plan_res.data.refined_objective,
                    questions=plan_res.data.research_questions,
                    queries=plan_res.data.search_queries,
                    subtopics=plan_res.data.subtopics,
                    outline=plan_res.data.suggested_outline,
                    model=plan_res.model,
                )
                session.add(plan_row)

            pipeline_state.check_cancellation()

            # -----------------------------------------------------------------
            # STEP 3: RESEARCHER
            # -----------------------------------------------------------------
            pipeline_state.transition_to(TaskStage.RESEARCHING, TaskStatus.RESEARCHING, progress=35)
            async with session_scope(self.sessionmaker) as session:
                task = await session.get(ResearchTask, task_id)
                if task:
                    task.status = TaskStatus.RESEARCHING.value
                    task.stage = TaskStage.RESEARCHING.value
                    task.current_agent = AgentName.RESEARCHER.value
                    task.progress = 35
                await self._emit_event(session, task_id, EventType.AGENT, TaskStage.RESEARCHING, AgentName.RESEARCHER)

            async def _run_researcher():
                r_in = ResearcherInput(
                    research_questions=pipeline_state.planner_output.research_questions,
                    search_queries=pipeline_state.planner_output.search_queries,
                    max_sources=request.resolved_source_target,
                    preferred_domains=request.preferred_domains,
                    excluded_domains=request.excluded_domains,
                    extract_content=True,
                )
                return await self.researcher.run(r_in, context=AgentContext(task_id=str(task_id), stage=TaskStage.RESEARCHING))

            res_res = await execute_with_pipeline_retry(
                _run_researcher,
                stage_name="researcher",
                max_attempts=self.settings.retry_max_attempts,
                base_delay=self.settings.retry_base_delay,
                secrets=self.settings.redaction_secrets(),
            )
            pipeline_state.researcher_output = res_res.data
            pipeline_state.accumulate_usage(res_res.usage.prompt_tokens, res_res.usage.completion_tokens, res_res.cost_estimate)

            # Persist sources into database
            source_uuid_map: dict[str, uuid.UUID] = {}
            async with session_scope(self.sessionmaker) as session:
                await self._record_execution(
                    session,
                    task_id,
                    TaskStage.RESEARCHING,
                    AgentName.RESEARCHER,
                    res_res.provider,
                    res_res.model,
                    ExecutionStatus.SUCCEEDED,
                    res_res.started_at,
                    res_res.finished_at,
                    res_res.latency_ms,
                    res_res.usage,
                    output_excerpt=f"Found {len(res_res.data.sources)} sources",
                )
                for src in res_res.data.sources:
                    source_id = uuid.uuid4()
                    source_uuid_map[src.source_id] = source_id
                    source_row = ResearchSource(
                        id=source_id,
                        task_id=task_id,
                        url=src.url,
                        canonical_url=src.url,
                        domain=src.domain,
                        title=src.title,
                        snippet=src.summary[:500] if src.summary else "",
                        summary=src.summary,
                        search_score=src.search_score,
                        status="ok",
                    )
                    session.add(source_row)

            pipeline_state.check_cancellation()

            # -----------------------------------------------------------------
            # STEP 4: ANALYST
            # -----------------------------------------------------------------
            pipeline_state.transition_to(TaskStage.ANALYZING, TaskStatus.ANALYZING, progress=55)
            async with session_scope(self.sessionmaker) as session:
                task = await session.get(ResearchTask, task_id)
                if task:
                    task.status = TaskStatus.ANALYZING.value
                    task.stage = TaskStage.ANALYZING.value
                    task.current_agent = AgentName.ANALYST.value
                    task.progress = 55
                await self._emit_event(session, task_id, EventType.AGENT, TaskStage.ANALYZING, AgentName.ANALYST)

            async def _run_analyst():
                a_in = AnalystInput(
                    plan=pipeline_state.planner_output,
                    sources=pipeline_state.researcher_output.sources,
                )
                return await self.analyst.run(a_in, context=AgentContext(task_id=str(task_id), stage=TaskStage.ANALYZING))

            ana_res = await execute_with_pipeline_retry(
                _run_analyst,
                stage_name="analyst",
                max_attempts=self.settings.retry_max_attempts,
                base_delay=self.settings.retry_base_delay,
                secrets=self.settings.redaction_secrets(),
            )
            pipeline_state.analyst_output = ana_res.data
            pipeline_state.accumulate_usage(ana_res.usage.prompt_tokens, ana_res.usage.completion_tokens, ana_res.cost_estimate)

            async with session_scope(self.sessionmaker) as session:
                await self._record_execution(
                    session,
                    task_id,
                    TaskStage.ANALYZING,
                    AgentName.ANALYST,
                    ana_res.provider,
                    ana_res.model,
                    ExecutionStatus.SUCCEEDED,
                    ana_res.started_at,
                    ana_res.finished_at,
                    ana_res.latency_ms,
                    ana_res.usage,
                    output_excerpt=f"Generated {len(ana_res.data.findings)} findings",
                )
                for finding in ana_res.data.findings:
                    f_id = uuid.uuid4()
                    finding_row = ResearchFinding(
                        id=f_id,
                        task_id=task_id,
                        category=finding.category.value,
                        statement=finding.statement,
                        evidence_strength=finding.evidence_strength.value,
                    )
                    session.add(finding_row)
                    # Link supporting sources in finding_sources join table
                    for sid in finding.supporting_source_ids:
                        if sid in source_uuid_map:
                            session.add(FindingSource(finding_id=f_id, source_id=source_uuid_map[sid]))

            pipeline_state.check_cancellation()

            # -----------------------------------------------------------------
            # STEP 5 & 6: WRITER & REVIEWER (WITH REVISION LOOP)
            # -----------------------------------------------------------------
            revision_feedback: list[str] = []
            latest_report_id: uuid.UUID | None = None

            while True:
                # STEP 5: WRITER
                stage_val = TaskStage.REVISING if pipeline_state.revision_count > 0 else TaskStage.WRITING
                status_val = TaskStatus.REVISING if pipeline_state.revision_count > 0 else TaskStatus.WRITING
                progress_val = min(85, 70 + (pipeline_state.revision_count * 5))
                pipeline_state.transition_to(stage_val, status_val, progress=progress_val)

                async with session_scope(self.sessionmaker) as session:
                    task = await session.get(ResearchTask, task_id)
                    if task:
                        task.status = status_val.value
                        task.stage = stage_val.value
                        task.current_agent = AgentName.WRITER.value
                        task.progress = progress_val
                        task.revision_count = pipeline_state.revision_count
                    await self._emit_event(session, task_id, EventType.AGENT, stage_val, AgentName.WRITER)

                async def _run_writer():
                    w_in = WriterInput(
                        plan=pipeline_state.planner_output,
                        analyst_output=pipeline_state.analyst_output,
                        sources=pipeline_state.researcher_output.sources,
                        format=request.format,
                        length=request.length,
                        depth=request.depth,
                        revision_feedback=revision_feedback,
                        previous_report=pipeline_state.writer_output,
                    )
                    return await self.writer.run(w_in, context=AgentContext(task_id=str(task_id), stage=stage_val))

                writer_res = await execute_with_pipeline_retry(
                    _run_writer,
                    stage_name="writer",
                    max_attempts=self.settings.retry_max_attempts,
                    base_delay=self.settings.retry_base_delay,
                    secrets=self.settings.redaction_secrets(),
                )
                pipeline_state.writer_output = writer_res.data
                pipeline_state.accumulate_usage(writer_res.usage.prompt_tokens, writer_res.usage.completion_tokens, writer_res.cost_estimate)

                latest_report_id = uuid.uuid4()
                async with session_scope(self.sessionmaker) as session:
                    await self._record_execution(
                        session,
                        task_id,
                        stage_val,
                        AgentName.WRITER,
                        writer_res.provider,
                        writer_res.model,
                        ExecutionStatus.SUCCEEDED,
                        writer_res.started_at,
                        writer_res.finished_at,
                        writer_res.latency_ms,
                        writer_res.usage,
                        output_excerpt=writer_res.data.title,
                    )
                    report_row = ResearchReport(
                        id=latest_report_id,
                        task_id=task_id,
                        version=pipeline_state.revision_count + 1,
                        status=ReportStatus.DRAFT.value,
                        title=writer_res.data.title,
                        executive_summary=writer_res.data.executive_summary,
                        content=[sec.model_dump() for sec in writer_res.data.main_sections],
                        word_count=writer_res.data.word_count,
                        review_status=ReviewStatus.PENDING.value,
                    )
                    session.add(report_row)

                pipeline_state.check_cancellation()

                # STEP 6: REVIEWER
                pipeline_state.transition_to(TaskStage.REVIEWING, TaskStatus.REVIEWING, progress=88)
                async with session_scope(self.sessionmaker) as session:
                    task = await session.get(ResearchTask, task_id)
                    if task:
                        task.status = TaskStatus.REVIEWING.value
                        task.stage = TaskStage.REVIEWING.value
                        task.current_agent = AgentName.REVIEWER.value
                        task.progress = 88
                    await self._emit_event(session, task_id, EventType.AGENT, TaskStage.REVIEWING, AgentName.REVIEWER)

                async def _run_reviewer():
                    rev_in = ReviewerInput(
                        plan=pipeline_state.planner_output,
                        sources=pipeline_state.researcher_output.sources,
                        analyst_output=pipeline_state.analyst_output,
                        report=pipeline_state.writer_output,
                    )
                    return await self.reviewer.run(rev_in, context=AgentContext(task_id=str(task_id), stage=TaskStage.REVIEWING))

                reviewer_res = await execute_with_pipeline_retry(
                    _run_reviewer,
                    stage_name="reviewer",
                    max_attempts=self.settings.retry_max_attempts,
                    base_delay=self.settings.retry_base_delay,
                    secrets=self.settings.redaction_secrets(),
                )
                pipeline_state.reviewer_output = reviewer_res.data
                pipeline_state.accumulate_usage(reviewer_res.usage.prompt_tokens, reviewer_res.usage.completion_tokens, reviewer_res.cost_estimate)

                review_passed = (
                    reviewer_res.data.quality_score >= 70
                    and not reviewer_res.data.accuracy_concerns
                )

                async with session_scope(self.sessionmaker) as session:
                    await self._record_execution(
                        session,
                        task_id,
                        TaskStage.REVIEWING,
                        AgentName.REVIEWER,
                        reviewer_res.provider,
                        reviewer_res.model,
                        ExecutionStatus.SUCCEEDED,
                        reviewer_res.started_at,
                        reviewer_res.finished_at,
                        reviewer_res.latency_ms,
                        reviewer_res.usage,
                        output_excerpt=f"Score: {reviewer_res.data.quality_score}",
                    )
                    # Compute metrics from citation and evidence validations
                    total_citations = len(reviewer_res.data.citation_validations)
                    supported_citations = sum(
                        1 for cit in reviewer_res.data.citation_validations
                        if cit.verdict == CitationVerdict.SUPPORTED
                    )
                    citation_coverage = (
                        float(supported_citations) / float(total_citations)
                        if total_citations > 0
                        else 1.0
                    )
                    unsupported_count = sum(
                        1 for cit in reviewer_res.data.citation_validations
                        if cit.verdict == CitationVerdict.UNSUPPORTED
                    )

                    review_row = ReportReview(
                        report_id=latest_report_id,
                        revision_no=pipeline_state.revision_count,
                        model=reviewer_res.model,
                        passed=review_passed,
                        quality_score=reviewer_res.data.quality_score,
                        citation_coverage=citation_coverage,
                        unsupported_count=unsupported_count,
                        feedback="\n".join(reviewer_res.data.revision_suggestions) if reviewer_res.data.revision_suggestions else "No specific suggestions",
                        findings=reviewer_res.data.model_dump(),
                    )
                    session.add(review_row)

                    # Persist individual citation validation rows
                    for idx, cit in enumerate(reviewer_res.data.citation_validations, start=1):
                        db_source_id = source_uuid_map.get(cit.source_id)
                        citation_row = ReportCitation(
                            report_id=latest_report_id,
                            source_id=db_source_id,
                            marker=idx,
                            section_key="body",
                            claim_text=f"Claim source {cit.source_id}",
                            verdict=cit.verdict.value,
                            review_note=cit.explanation,
                        )
                        session.add(citation_row)

                # Check if revision is required and permitted
                needs_revision = (
                    not review_passed
                    or bool(reviewer_res.data.accuracy_concerns)
                )
                if needs_revision and pipeline_state.revision_count < pipeline_state.max_revisions:
                    pipeline_state.revision_count += 1
                    revision_feedback = (
                        reviewer_res.data.revision_suggestions
                        + reviewer_res.data.accuracy_concerns
                    )
                    logger.info(
                        "[pipeline] Report score (%d) requires revision (attempt %d/%d). Looping to Writer...",
                        reviewer_res.data.quality_score,
                        pipeline_state.revision_count,
                        pipeline_state.max_revisions,
                    )
                    continue

                # Review cycle complete
                break

            # -----------------------------------------------------------------
            # STEP 7: COMPLETE (AWAITING HUMAN APPROVAL)
            # -----------------------------------------------------------------
            pipeline_state.transition_to(TaskStage.COMPLETED, TaskStatus.PENDING_APPROVAL, progress=100)
            pipeline_state.completed_at = datetime.now(timezone.utc)

            async with session_scope(self.sessionmaker) as session:
                task = await session.get(ResearchTask, task_id)
                if task:
                    task.stage = TaskStage.COMPLETED.value
                    # Mandatory requirement 6: Final status remains PENDING_APPROVAL
                    task.status = TaskStatus.PENDING_APPROVAL.value
                    task.progress = 100
                    task.completed_at = pipeline_state.completed_at
                    task.total_tokens = pipeline_state.total_tokens
                    task.total_calls = pipeline_state.total_calls
                    task.estimated_cost = pipeline_state.estimated_cost
                    task.revision_count = pipeline_state.revision_count

                if latest_report_id:
                    report = await session.get(ResearchReport, latest_report_id)
                    if report:
                        report.status = ReportStatus.PENDING_APPROVAL.value
                        report.review_status = (
                            ReviewStatus.PASSED.value if review_passed else ReviewStatus.FAILED.value
                        )

                await self._emit_event(
                    session,
                    task_id,
                    EventType.DONE,
                    TaskStage.COMPLETED,
                    payload={"status": "pending_approval", "score": pipeline_state.reviewer_output.quality_score},
                )

            return PipelineResult(
                task_id=task_id,
                status=TaskStatus.PENDING_APPROVAL,
                stage=TaskStage.COMPLETED,
                state=pipeline_state,
                report_id=latest_report_id,
                report_title=pipeline_state.writer_output.title if pipeline_state.writer_output else "",
                review_score=pipeline_state.reviewer_output.quality_score if pipeline_state.reviewer_output else None,
                approval_recommendation=pipeline_state.reviewer_output.approval_recommendation if pipeline_state.reviewer_output else False,
                human_approval_required=True,
                total_tokens=pipeline_state.total_tokens,
                estimated_cost=pipeline_state.estimated_cost,
            )

        except PipelineCancelledError as exc:
            pipeline_state.transition_to(TaskStage.CANCELLED, TaskStatus.CANCELLED)
            safe_msg = redact(str(exc), self.settings.redaction_secrets())
            logger.warning("[pipeline] Task %s was cancelled: %s", task_id, safe_msg)
            async with session_scope(self.sessionmaker) as session:
                task = await session.get(ResearchTask, task_id)
                if task:
                    task.stage = TaskStage.CANCELLED.value
                    task.status = TaskStatus.CANCELLED.value
                await self._emit_event(session, task_id, EventType.STAGE, TaskStage.CANCELLED, payload={"status": "cancelled"})
            raise

        except BaseException as exc:
            safe_msg = redact(str(exc), self.settings.redaction_secrets())
            logger.exception("[pipeline] Task %s failed: %s", task_id, safe_msg)
            pipeline_state.transition_to(TaskStage.FAILED, TaskStatus.FAILED)
            pipeline_state.error_message = safe_msg
            pipeline_state.error_code = exc.__class__.__name__

            async with session_scope(self.sessionmaker) as session:
                task = await session.get(ResearchTask, task_id)
                if task:
                    task.stage = TaskStage.FAILED.value
                    task.status = TaskStatus.FAILED.value
                    task.error_code = pipeline_state.error_code
                    task.error_message = safe_msg
                await self._emit_event(
                    session,
                    task_id,
                    EventType.ERROR,
                    pipeline_state.current_stage,
                    payload={"error_code": pipeline_state.error_code, "error": safe_msg},
                )
            raise PipelineStageError(
                f"Pipeline execution failed at {pipeline_state.current_stage.value}: {safe_msg}",
                task_id=task_id,
                stage=pipeline_state.current_stage.value,
            ) from exc
