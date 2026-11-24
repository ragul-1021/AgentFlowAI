"""Tests for AnalystAgent."""

from __future__ import annotations

import pytest

from app.agents import (
    AnalystAgent,
    AnalystInput,
    AnalystOutput,
    FindingCategory,
    PlannerOutput,
    SourceItem,
)
from app.enums import ExecutionStatus
from app.providers import MockGeminiProvider


@pytest.mark.asyncio
async def test_analyst_agent_successful_analysis():
    provider = MockGeminiProvider()
    analyst = AnalystAgent(provider)

    mock_plan = PlannerOutput(
        title="Vector Index Performance",
        refined_objective="Compare HNSW and IVFFlat index construction and query cost",
        research_questions=["How does build time compare?"],
        subtopics=["Build time", "Memory"],
        search_queries=["pgvector build time comparison"],
        suggested_outline=["Intro", "Results", "Conclusion"],
    )

    mock_sources = [
        SourceItem(
            source_id="src_01",
            title="Benchmarking pgvector at scale",
            url="https://postgresql.org/blog/benchmarking-pgvector",
            domain="postgresql.org",
            summary="HNSW offers higher QPS but requires significantly longer indexing build times and RAM.",
        ),
        SourceItem(
            source_id="src_02",
            title="IVFFlat deep dive",
            url="https://github.com/pgvector/pgvector",
            domain="github.com",
            summary="IVFFlat builds fast and uses less memory, but recall degrades under large vector counts.",
        ),
    ]

    payload = AnalystInput(plan=mock_plan, sources=mock_sources)
    result = await analyst.run(payload)

    assert result.status == ExecutionStatus.SUCCEEDED
    assert isinstance(result.data, AnalystOutput)
    assert len(result.data.executive_findings) >= 1
    assert len(result.data.findings) >= 1

    # Verify that all findings reference valid source IDs
    valid_ids = {"src_01", "src_02"}
    for finding in result.data.findings:
        assert finding.finding_id
        assert finding.confidence_explanation
        assert all(sid in valid_ids for sid in finding.supporting_source_ids)
