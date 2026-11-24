"""Tests for agent input and output Pydantic contracts and constraints."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents import (
    AnalystInput,
    AnalystOutput,
    CitationValidation,
    CitationVerdict,
    EvidenceStrength,
    FindingCategory,
    FindingItem,
    PlannerInput,
    PlannerOutput,
    ReferenceItem,
    ReportSection,
    ResearcherInput,
    ResearcherOutput,
    ReviewerInput,
    ReviewerOutput,
    SourceItem,
    WriterInput,
    WriterOutput,
)
from app.enums import ResearchDepth


def test_planner_input_validation():
    # Valid input
    p_in = PlannerInput(topic="Autonomous Multi-Agent AI Systems", depth=ResearchDepth.DEEP)
    assert p_in.topic == "Autonomous Multi-Agent AI Systems"
    assert p_in.max_questions == 5

    # Short topic raises ValidationError
    with pytest.raises(ValidationError):
        PlannerInput(topic="hi")


def test_planner_output_query_deduplication():
    # Duplicate search queries and questions should be cleanly deduplicated by validator
    out = PlannerOutput(
        title="AI Evaluation Research",
        refined_objective="Thorough study on multi-agent evaluation frameworks",
        research_questions=["What is X?", "what is x? ", "How does Y work?"],
        subtopics=["Evaluation", "Frameworks"],
        search_queries=["Multi agent evaluation", "multi agent evaluation", "Agent metrics"],
        suggested_outline=["Intro", "Methods", "Results"],
    )
    assert len(out.research_questions) == 2
    assert len(out.search_queries) == 2


def test_researcher_input_domain_normalization():
    r_in = ResearcherInput(
        research_questions=["Q1"],
        search_queries=["query 1"],
        preferred_domains=["https://arxiv.org/abs/123", "github.com/"],
        excluded_domains=["spam.com"],
    )
    assert r_in.preferred_domains == ["arxiv.org", "github.com"]
    assert r_in.excluded_domains == ["spam.com"]


def test_source_item_validation():
    src = SourceItem(
        source_id="src_01",
        title="Deep Analysis of Agents",
        url="https://arxiv.org/abs/2401.0001",
        domain="arxiv.org",
        summary="A paper exploring coordination patterns in agent swarms.",
    )
    assert src.source_id == "src_01"
    assert src.extracted_content is None

    # Invalid URL format raises
    with pytest.raises(ValidationError):
        SourceItem(
            source_id="src_02",
            title="Bad Source",
            url="not-a-valid-url",
            domain="bad",
            summary="Invalid",
        )


def test_analyst_finding_item_validation():
    f = FindingItem(
        finding_id="find_01",
        category=FindingCategory.FINDING,
        statement="Agents coordinate efficiently with structured message envelopes.",
        explanation="Multi-agent benchmarking shows structured schemas reduce parsing errors by 90%.",
        evidence_strength=EvidenceStrength.HIGH,
        confidence_score=0.92,
        confidence_explanation="Supported by 3 empirical benchmark studies in source documents.",
        supporting_source_ids=["src_01", "src_02"],
    )
    assert f.confidence_score == 0.92
    assert len(f.supporting_source_ids) == 2

    # Confidence score > 1.0 raises ValidationError
    with pytest.raises(ValidationError):
        FindingItem(
            finding_id="find_02",
            statement="Statement",
            explanation="Explanation",
            confidence_score=1.5,
            confidence_explanation="Invalid",
            supporting_source_ids=["src_01"],
        )


def test_writer_report_section_hierarchy():
    sub = ReportSection(heading="Latency Tradeoffs", content="Detailed discussion on token latency.")
    sec = ReportSection(heading="Performance Evaluation", content="Overview of speed and cost.", subsections=[sub])
    assert len(sec.subsections) == 1
    assert sec.subsections[0].heading == "Latency Tradeoffs"


def test_reviewer_citation_validation():
    val = CitationValidation(
        source_id="src_01",
        verdict=CitationVerdict.SUPPORTED,
        explanation="Source URL matches and supports conclusion.",
    )
    assert val.verdict == CitationVerdict.SUPPORTED
