"""Tests for WriterAgent."""

from __future__ import annotations

import pytest

from app.agents import (
    AnalystOutput,
    FindingCategory,
    FindingItem,
    PlannerOutput,
    SourceItem,
    WriterAgent,
    WriterInput,
    WriterOutput,
)
from app.enums import ExecutionStatus, ReportFormat, ReportLength
from app.providers import MockGeminiProvider


@pytest.mark.asyncio
async def test_writer_agent_successful_report_generation():
    provider = MockGeminiProvider()
    writer = WriterAgent(provider)

    mock_plan = PlannerOutput(
        title="Comparative Evaluation of Vector Indexes",
        refined_objective="Produce exhaustive technical report comparing HNSW and IVFFlat",
        research_questions=["What are the memory and latency tradeoffs?"],
        subtopics=["Architecture", "Performance"],
        search_queries=["pgvector benchmark"],
        suggested_outline=["Introduction", "Methodology", "Performance Tradeoffs", "Conclusion"],
    )

    mock_sources = [
        SourceItem(
            source_id="src_01",
            title="Vector Index Performance",
            url="https://postgresql.org/benchmarks/vectors",
            domain="postgresql.org",
            summary="HNSW reaches 99% recall at 10x higher memory cost.",
        )
    ]

    mock_analyst = AnalystOutput(
        executive_findings=["HNSW outperforms in latency; IVFFlat excels in build speed."],
        key_insights=["Memory footprint is the deciding deployment factor."],
        findings=[
            FindingItem(
                finding_id="find_01",
                category=FindingCategory.FINDING,
                statement="HNSW consumes 3x-5x more RAM than IVFFlat.",
                explanation="Graph-based neighbor lists require substantial in-memory buffers.",
                confidence_score=0.95,
                confidence_explanation="Confirmed across multiple empirical benchmarks.",
                supporting_source_ids=["src_01"],
            )
        ],
    )

    payload = WriterInput(
        plan=mock_plan,
        analyst_output=mock_analyst,
        sources=mock_sources,
        format=ReportFormat.TECHNICAL,
        length=ReportLength.STANDARD,
    )

    result = await writer.run(payload)
    assert result.status == ExecutionStatus.SUCCEEDED
    assert isinstance(result.data, WriterOutput)
    assert result.data.title
    assert result.data.executive_summary
    assert result.data.word_count > 0
    assert len(result.data.references) >= 1

    # Verify cited references preserve original source URLs
    for ref in result.data.references:
        assert ref.source_id == "src_01"
        assert ref.url == "https://postgresql.org/benchmarks/vectors"
