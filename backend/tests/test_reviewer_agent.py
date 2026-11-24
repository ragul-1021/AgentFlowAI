"""Tests for ReviewerAgent."""

from __future__ import annotations

import pytest

from app.agents import (
    AnalystOutput,
    CitationVerdict,
    FindingCategory,
    FindingItem,
    PlannerOutput,
    ReferenceItem,
    ReportSection,
    ReviewerAgent,
    ReviewerInput,
    ReviewerOutput,
    SourceItem,
    WriterOutput,
)
from app.enums import ExecutionStatus
from app.providers import MockGeminiProvider


@pytest.mark.asyncio
async def test_reviewer_agent_successful_audit():
    provider = MockGeminiProvider()
    reviewer = ReviewerAgent(provider)

    mock_plan = PlannerOutput(
        title="Vector Performance Study",
        refined_objective="Evaluate indexing trade-offs",
        research_questions=["What is the recall vs build time ratio?"],
        subtopics=["Recall", "Build Time"],
        search_queries=["pgvector recall"],
        suggested_outline=["Intro", "Body", "Conclusion"],
    )

    mock_sources = [
        SourceItem(
            source_id="src_01",
            title="Database Vector Benchmarks",
            url="https://postgresql.org/benchmarks",
            domain="postgresql.org",
            summary="Extensive study on indexing recall and latency curves.",
        )
    ]

    mock_analyst = AnalystOutput(
        executive_findings=["HNSW provides superior recall at the cost of build time."],
        key_insights=["Memory usage dominates operational costs."],
        findings=[
            FindingItem(
                finding_id="find_01",
                category=FindingCategory.FINDING,
                statement="Recall remains above 98% under high concurrency.",
                explanation="Graph connections maintain high path diversity.",
                confidence_score=0.9,
                confidence_explanation="Validated via benchmark run results.",
                supporting_source_ids=["src_01"],
            )
        ],
    )

    mock_report = WriterOutput(
        title="Comprehensive Index Analysis",
        executive_summary="This report evaluates HNSW vs IVFFlat in production databases.",
        introduction="Vector search requires balancing recall and hardware expenditure.",
        methodology="Empirical performance testing using standardized datasets.",
        main_sections=[ReportSection(heading="Performance", content="Detailed discussion citing [src_01].")],
        findings="Findings demonstrate substantial recall advantages.",
        discussion="Discussion on tradeoffs.",
        limitations="Limited to 1536-dimension embeddings.",
        conclusion="HNSW is recommended for low-latency critical pathways.",
        references=[ReferenceItem(source_id="src_01", title="Database Vector Benchmarks", url="https://postgresql.org/benchmarks")],
        word_count=520,
    )

    payload = ReviewerInput(
        plan=mock_plan,
        sources=mock_sources,
        analyst_output=mock_analyst,
        report=mock_report,
    )

    result = await reviewer.run(payload)
    assert result.status == ExecutionStatus.SUCCEEDED
    assert isinstance(result.data, ReviewerOutput)
    assert 0 <= result.data.quality_score <= 100
    assert 0 <= result.data.completeness_score <= 100
    assert len(result.data.citation_validations) >= 1
    assert result.data.citation_validations[0].verdict == CitationVerdict.SUPPORTED

    # Ensure human review reminder is explicitly injected
    assert "Final approval is reserved for the human user" in result.data.recommendation_notes


@pytest.mark.asyncio
async def test_reviewer_detects_hallucinated_citations():
    provider = MockGeminiProvider()
    reviewer = ReviewerAgent(provider)

    mock_plan = PlannerOutput(
        title="Title",
        refined_objective="Thorough research objective on target",
        research_questions=["Q1"],
        subtopics=["S1"],
        search_queries=["q1"],
        suggested_outline=["S1", "S2", "S3"],
    )

    # Source list only has src_01
    mock_sources = [
        SourceItem(
            source_id="src_01",
            title="Real Source",
            url="https://real.org",
            domain="real.org",
            summary="Real data.",
        )
    ]

    mock_analyst = AnalystOutput(executive_findings=["F1"], key_insights=["I1"], findings=[])

    # Report cites a hallucinated source ID src_99 that does NOT exist
    mock_report = WriterOutput(
        title="Title",
        executive_summary="Executive summary content exceeding 30 characters.",
        introduction="Introduction content exceeding 20 chars.",
        methodology="Methodology content exceeding 20 chars.",
        main_sections=[ReportSection(heading="Section", content="Content citing fake source.")],
        findings="Findings content exceeding 20 chars.",
        discussion="Discussion content exceeding 20 chars.",
        limitations="Limitations content.",
        conclusion="Conclusion content exceeding 20 chars.",
        references=[
            ReferenceItem(source_id="src_99", title="Fake Hallucinated Source", url="https://fake.org")
        ],
        word_count=100,
    )

    payload = ReviewerInput(
        plan=mock_plan,
        sources=mock_sources,
        analyst_output=mock_analyst,
        report=mock_report,
    )

    result = await reviewer.run(payload)
    assert result.status == ExecutionStatus.SUCCEEDED
    # Hallucinated citation should be caught and marked UNSUPPORTED
    unsupported = [v for v in result.data.citation_validations if v.verdict == CitationVerdict.UNSUPPORTED]
    assert len(unsupported) == 1
    assert unsupported[0].source_id == "src_99"
    assert result.data.approval_recommendation is False
