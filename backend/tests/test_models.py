from __future__ import annotations

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Base,
    AgentExecution,
    FindingSource,
    ReportApproval,
    ReportCitation,
    ReportReview,
    ResearchFinding,
    ResearchPlan,
    ResearchReport,
    ResearchSource,
    ResearchTask,
    RunEvent,
)


def make_task(topic: str = "A source-grounded research task", **overrides) -> ResearchTask:
    values = {
        "topic": topic,
        "description": "Objective and scope",
        "depth": "standard",
        "source_target": 12,
        "format": "general",
        "length": "standard",
    }
    values.update(overrides)
    return ResearchTask(**values)


async def count(session: AsyncSession, model) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def test_task_defaults_are_applied(db_session: AsyncSession) -> None:
    task = make_task(preferred_domains=["example.com"])
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    assert task.id is not None
    assert task.status == "queued"
    assert task.stage == "queued"
    assert task.progress == 0
    assert task.revision_count == 0
    assert task.total_tokens == 0
    assert task.total_calls == 0
    assert task.preferred_domains == ["example.com"]
    assert task.excluded_domains == []
    assert task.created_at is not None
    assert task.updated_at is not None


async def test_full_relationship_graph(db_session: AsyncSession) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()

    plan = ResearchPlan(
        task_id=task.id,
        objective="Explain the landscape",
        questions=["What changed?", "Who leads?"],
        queries=["query one"],
        subtopics=["subtopic"],
        outline=[{"key": "introduction", "title": "Introduction"}],
        model="gemini-test",
    )
    source = ResearchSource(
        task_id=task.id,
        url="https://example.com/report?utm_source=x",
        canonical_url="https://example.com/report",
        domain="example.com",
        title="Example report",
        content_hash="abc123",
        quality_score=0.8,
        quality={"credibility": 0.9},
    )
    finding = ResearchFinding(
        task_id=task.id,
        category="finding",
        statement="Market grew 20% year over year",
        evidence_strength="high",
    )
    execution = AgentExecution(
        task_id=task.id,
        stage="planning",
        agent="planner",
        provider="gemini",
        model="gemini-test",
        status="succeeded",
        prompt_tokens=100,
        completion_tokens=50,
    )
    report = ResearchReport(
        task_id=task.id,
        version=1,
        status="pending_approval",
        title="Draft report",
        executive_summary="Summary",
        content=[{"key": "introduction", "body": "Body [1]"}],
        word_count=1200,
        review_status="passed",
    )
    db_session.add_all([plan, source, finding, execution, report])
    await db_session.flush()

    db_session.add(FindingSource(finding_id=finding.id, source_id=source.id))
    db_session.add(
        ReportCitation(
            report_id=report.id,
            source_id=source.id,
            marker=1,
            section_key="introduction",
            claim_text="Market grew 20% [1]",
            verdict="supported",
        )
    )
    db_session.add(
        ReportReview(
            report_id=report.id,
            revision_no=0,
            model="gemini-test",
            passed=True,
            quality_score=88,
            citation_coverage=0.95,
            unsupported_count=0,
            feedback="Good",
            findings={"missing_sections": []},
        )
    )
    db_session.add(
        ReportApproval(
            report_id=report.id,
            decision="approved",
            actor="reviewer@example.com",
        )
    )
    db_session.add(
        RunEvent(task_id=task.id, type="stage", stage="reviewing", message="review done")
    )
    await db_session.commit()

    assert await count(db_session, ResearchTask) == 1
    assert await count(db_session, ResearchPlan) == 1
    assert await count(db_session, ResearchSource) == 1
    assert await count(db_session, ResearchFinding) == 1
    assert await count(db_session, FindingSource) == 1
    assert await count(db_session, AgentExecution) == 1
    assert await count(db_session, ResearchReport) == 1
    assert await count(db_session, ReportCitation) == 1
    assert await count(db_session, ReportReview) == 1
    assert await count(db_session, ReportApproval) == 1
    assert await count(db_session, RunEvent) == 1

    linked = await db_session.scalar(
        select(func.count())
        .select_from(FindingSource)
        .where(
            FindingSource.finding_id == finding.id,
            FindingSource.source_id == source.id,
        )
    )
    assert linked == 1


async def test_cascade_delete_removes_all_children(db_session: AsyncSession) -> None:
    task = make_task(topic="Cascade delete research task")
    db_session.add(task)
    await db_session.flush()

    source = ResearchSource(
        task_id=task.id,
        url="https://example.com/a",
        canonical_url="https://example.com/a",
    )
    finding = ResearchFinding(task_id=task.id, statement="Statement")
    report = ResearchReport(task_id=task.id, version=1)
    db_session.add_all(
        [
            ResearchPlan(task_id=task.id, objective="Objective"),
            source,
            finding,
            AgentExecution(
                task_id=task.id, stage="planning", agent="planner", provider="gemini"
            ),
            report,
            RunEvent(task_id=task.id, type="log", message="event"),
        ]
    )
    await db_session.flush()
    db_session.add(FindingSource(finding_id=finding.id, source_id=source.id))
    db_session.add(
        ReportCitation(report_id=report.id, source_id=source.id, marker=1)
    )
    db_session.add(ReportReview(report_id=report.id))
    db_session.add(ReportApproval(report_id=report.id, decision="rejected", notes="no"))
    await db_session.commit()

    await db_session.execute(delete(ResearchTask).where(ResearchTask.id == task.id))
    await db_session.commit()

    for model in (
        ResearchTask,
        ResearchPlan,
        ResearchSource,
        ResearchFinding,
        FindingSource,
        AgentExecution,
        ResearchReport,
        ReportCitation,
        ReportReview,
        ReportApproval,
        RunEvent,
    ):
        assert await count(db_session, model) == 0, model.__tablename__


async def test_canonical_url_is_unique_per_task(db_session: AsyncSession) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()
    db_session.add_all(
        [
            ResearchSource(
                task_id=task.id,
                url="https://example.com/x",
                canonical_url="https://example.com/x",
            ),
            ResearchSource(
                task_id=task.id,
                url="https://example.com/x?utm=1",
                canonical_url="https://example.com/x",
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_one_plan_per_task(db_session: AsyncSession) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()
    db_session.add_all(
        [
            ResearchPlan(task_id=task.id, objective="One"),
            ResearchPlan(task_id=task.id, objective="Two"),
        ]
    )

    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_report_version_is_unique_per_task(db_session: AsyncSession) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()
    db_session.add_all(
        [
            ResearchReport(task_id=task.id, version=1),
            ResearchReport(task_id=task.id, version=1),
        ]
    )

    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_only_one_approved_approval_per_report(db_session: AsyncSession) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()
    report = ResearchReport(task_id=task.id, version=1)
    db_session.add(report)
    await db_session.flush()
    db_session.add(
        ReportApproval(report_id=report.id, decision="approved", actor="first")
    )
    await db_session.commit()

    db_session.add(
        ReportApproval(report_id=report.id, decision="approved", actor="second")
    )
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_rejection_and_approval_can_coexist(db_session: AsyncSession) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()
    report = ResearchReport(task_id=task.id, version=1)
    db_session.add(report)
    await db_session.flush()
    db_session.add_all(
        [
            ReportApproval(report_id=report.id, decision="rejected", notes="Needs work"),
            ReportApproval(report_id=report.id, decision="approved", actor="owner"),
        ]
    )
    await db_session.commit()

    assert await count(db_session, ReportApproval) == 2


async def test_progress_check_constraint(db_session: AsyncSession) -> None:
    db_session.add(make_task(progress=150))

    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_source_quality_score_check_constraint(db_session: AsyncSession) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()
    db_session.add(
        ResearchSource(
            task_id=task.id,
            url="https://example.com/q",
            canonical_url="https://example.com/q",
            quality_score=1.5,
        )
    )

    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_review_score_check_constraint(db_session: AsyncSession) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()
    report = ResearchReport(task_id=task.id, version=1)
    db_session.add(report)
    await db_session.flush()
    db_session.add(ReportReview(report_id=report.id, quality_score=101))

    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_deleting_source_nulls_citation_but_keeps_report(
    db_session: AsyncSession,
) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()
    source = ResearchSource(
        task_id=task.id,
        url="https://example.com/s",
        canonical_url="https://example.com/s",
    )
    report = ResearchReport(task_id=task.id, version=1, content=[{"body": "text [1]"}])
    db_session.add_all([source, report])
    await db_session.flush()
    citation = ReportCitation(report_id=report.id, source_id=source.id, marker=1)
    db_session.add(citation)
    await db_session.commit()

    await db_session.execute(
        delete(ResearchSource).where(ResearchSource.id == source.id)
    )
    await db_session.commit()
    await db_session.refresh(citation)

    assert citation.source_id is None
    assert await count(db_session, ResearchReport) == 1


async def test_run_events_are_ordered(db_session: AsyncSession) -> None:
    task = make_task()
    db_session.add(task)
    await db_session.flush()
    db_session.add_all(
        [
            RunEvent(task_id=task.id, type="stage", message="one"),
            RunEvent(task_id=task.id, type="log", message="two"),
            RunEvent(task_id=task.id, type="done", message="three"),
        ]
    )
    await db_session.commit()

    events = (
        await db_session.scalars(
            select(RunEvent).where(RunEvent.task_id == task.id).order_by(RunEvent.id)
        )
    ).all()
    assert [event.message for event in events] == ["one", "two", "three"]
    assert events[0].id < events[1].id < events[2].id


def test_metadata_registers_all_11_models() -> None:
    expected_tables = {
        "research_tasks",
        "research_plans",
        "agent_executions",
        "research_sources",
        "research_findings",
        "finding_sources",
        "research_reports",
        "report_citations",
        "report_reviews",
        "report_approvals",
        "run_events",
    }
    registered = set(Base.metadata.tables.keys())
    assert expected_tables.issubset(registered)


def test_metadata_foreign_keys_and_cascades() -> None:
    tables = Base.metadata.tables

    # research_plans -> research_tasks (CASCADE)
    plan_fks = list(tables["research_plans"].foreign_keys)
    assert len(plan_fks) == 1
    assert plan_fks[0].target_fullname == "research_tasks.id"
    assert plan_fks[0].ondelete == "CASCADE"

    # agent_executions -> research_tasks (CASCADE)
    exec_fks = list(tables["agent_executions"].foreign_keys)
    assert len(exec_fks) == 1
    assert exec_fks[0].target_fullname == "research_tasks.id"
    assert exec_fks[0].ondelete == "CASCADE"

    # research_sources -> research_tasks (CASCADE)
    src_fks = list(tables["research_sources"].foreign_keys)
    assert len(src_fks) == 1
    assert src_fks[0].target_fullname == "research_tasks.id"
    assert src_fks[0].ondelete == "CASCADE"

    # research_findings -> research_tasks (CASCADE)
    find_fks = list(tables["research_findings"].foreign_keys)
    assert len(find_fks) == 1
    assert find_fks[0].target_fullname == "research_tasks.id"
    assert find_fks[0].ondelete == "CASCADE"

    # finding_sources -> findings & sources (CASCADE)
    fs_fks = {fk.target_fullname: fk.ondelete for fk in tables["finding_sources"].foreign_keys}
    assert fs_fks == {
        "research_findings.id": "CASCADE",
        "research_sources.id": "CASCADE",
    }

    # research_reports -> research_tasks (CASCADE)
    rep_fks = list(tables["research_reports"].foreign_keys)
    assert len(rep_fks) == 1
    assert rep_fks[0].target_fullname == "research_tasks.id"
    assert rep_fks[0].ondelete == "CASCADE"

    # report_citations -> reports (CASCADE) and sources (SET NULL)
    cit_fks = {fk.target_fullname: fk.ondelete for fk in tables["report_citations"].foreign_keys}
    assert cit_fks == {
        "research_reports.id": "CASCADE",
        "research_sources.id": "SET NULL",
    }

    # report_reviews -> reports (CASCADE)
    rev_fks = list(tables["report_reviews"].foreign_keys)
    assert len(rev_fks) == 1
    assert rev_fks[0].target_fullname == "research_reports.id"
    assert rev_fks[0].ondelete == "CASCADE"

    # report_approvals -> reports (CASCADE)
    app_fks = list(tables["report_approvals"].foreign_keys)
    assert len(app_fks) == 1
    assert app_fks[0].target_fullname == "research_reports.id"
    assert app_fks[0].ondelete == "CASCADE"

    # run_events -> research_tasks (CASCADE)
    ev_fks = list(tables["run_events"].foreign_keys)
    assert len(ev_fks) == 1
    assert ev_fks[0].target_fullname == "research_tasks.id"
    assert ev_fks[0].ondelete == "CASCADE"


def test_metadata_unique_constraints_and_indexes() -> None:
    tables = Base.metadata.tables

    # research_sources: prevents duplicate source records within task
    src_uqs = [
        [c.name for c in uq.columns]
        for uq in tables["research_sources"].constraints
        if uq.__class__.__name__ == "UniqueConstraint"
    ]
    assert ["task_id", "canonical_url"] in src_uqs

    # research_plans: one plan per task
    plan_uqs = [
        [c.name for c in uq.columns]
        for uq in tables["research_plans"].constraints
        if uq.__class__.__name__ == "UniqueConstraint"
    ]
    assert ["task_id"] in plan_uqs

    # research_reports: unique task_id + version
    rep_uqs = [
        [c.name for c in uq.columns]
        for uq in tables["research_reports"].constraints
        if uq.__class__.__name__ == "UniqueConstraint"
    ]
    assert ["task_id", "version"] in rep_uqs

    # report_citations: unique report_id + marker + section_key
    cit_uqs = [
        [c.name for c in uq.columns]
        for uq in tables["report_citations"].constraints
        if uq.__class__.__name__ == "UniqueConstraint"
    ]
    assert ["report_id", "marker", "section_key"] in cit_uqs

    # report_approvals: partial unique index for approved decision
    app_indices = {idx.name: idx for idx in tables["report_approvals"].indexes}
    assert "uq_report_approvals_approved" in app_indices
    assert app_indices["uq_report_approvals_approved"].unique is True


def test_in_memory_model_instantiation() -> None:
    task = ResearchTask(topic="Offline Test Topic")
    assert task.topic == "Offline Test Topic"

    plan = ResearchPlan(objective="Discover insights", questions=["Q1"])
    assert plan.objective == "Discover insights"
    assert plan.questions == ["Q1"]

    source = ResearchSource(url="https://test.com", canonical_url="https://test.com")
    assert source.canonical_url == "https://test.com"

    finding = ResearchFinding(statement="Key finding statement")
    assert finding.statement == "Key finding statement"

    execution = AgentExecution(
        stage="planning", agent="planner", provider="gemini", model="gemini-1.5-pro"
    )
    assert execution.stage == "planning"

    report = ResearchReport(version=1, title="Test Report")
    assert report.version == 1

    citation = ReportCitation(marker=1, claim_text="Claim [1]")
    assert citation.marker == 1

    review = ReportReview(quality_score=95, passed=True)
    assert review.passed is True

    approval = ReportApproval(decision="approved", actor="lead_reviewer")
    assert approval.decision == "approved"

    event = RunEvent(type="log", message="Task started")
    assert event.message == "Task started"

