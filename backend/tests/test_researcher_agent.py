"""Tests for ResearcherAgent."""

from __future__ import annotations

import pytest

from app.agents import ResearcherAgent, ResearcherInput, ResearcherOutput, SourceItem
from app.enums import ExecutionStatus
from app.providers import MockGeminiProvider, MockTavilyProvider


@pytest.mark.asyncio
async def test_researcher_agent_successful_search_and_deduplication():
    search_provider = MockTavilyProvider()
    llm_provider = MockGeminiProvider(default_model="gemini-2.5-flash-lite")
    researcher = ResearcherAgent(search_provider, llm_provider)

    payload = ResearcherInput(
        research_questions=["What is the query latency of HNSW vs IVFFlat?"],
        search_queries=["pgvector HNSW benchmark latency", "pgvector index comparison"],
        max_sources=5,
        preferred_domains=["postgresql.org", "github.com"],
        excluded_domains=["unwanted-spam.org"],
    )

    result = await researcher.run(payload)
    assert result.status == ExecutionStatus.SUCCEEDED
    assert isinstance(result.data, ResearcherOutput)
    assert len(result.data.sources) <= 5
    assert result.data.total_sources_found > 0
    assert len(result.data.queries_executed) == 2

    # Verify source item properties
    for src in result.data.sources:
        assert src.source_id.startswith("src_")
        assert src.url.startswith("http")
        assert src.summary
        assert src.domain != "unwanted-spam.org"


@pytest.mark.asyncio
async def test_researcher_agent_empty_search_queries_raises_validation():
    search_provider = MockTavilyProvider()
    llm_provider = MockGeminiProvider(default_model="gemini-2.5-flash-lite")
    researcher = ResearcherAgent(search_provider, llm_provider)

    with pytest.raises(Exception):
        await researcher.run(ResearcherInput.model_construct(research_questions=["Q1"], search_queries=[]))
