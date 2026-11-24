"""Reviewer Agent for AgentFlow AI.

Audits report quality, verifies citations against collected source evidence,
detects missing coverage, and provides recommendations for human review.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.agents.context import AgentContext
from app.agents.schemas import (
    CitationValidation,
    EvidenceValidation,
    ReviewerInput,
    ReviewerOutput,
)
from app.config import Settings, get_settings
from app.core.tokens import TokenUsage
from app.enums import AgentName, CitationVerdict
from app.providers.base import BaseLLMProvider, ChatMessage, LLMRequest

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are the Senior Audit and Verification Reviewer for AgentFlow AI.
Your responsibility is to critically evaluate a drafted research report against the research plan and source evidence.

Evaluation Rules:
1. Verify Citations: Check that every cited source ID exists in the provided source records.
2. Verify Evidence: Evaluate if claims are supported by available evidence summaries.
3. Check Completeness: Score how thoroughly the report answered each research question (0-100).
4. Score Overall Quality: Assign a rigorous quality score (0-100).
5. Highlight Concerns: Explicitly identify missing answers, unsupported conclusions, or gaps.
6. Recommendation: Provide revision guidance. Remember that final approval belongs to the human user."""


class ReviewerAgent(BaseAgent[ReviewerInput, ReviewerOutput]):
    """Agent responsible for report verification, evidence checking, and quality scoring."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        target_model = model or cfg.reviewer_model
        super().__init__(AgentName.REVIEWER, provider, model=target_model, settings=cfg)

    async def _execute(
        self,
        input_data: ReviewerInput,
        context: AgentContext,
    ) -> tuple[ReviewerOutput, TokenUsage, dict[str, Any]]:
        known_sources = {s.source_id: s for s in input_data.sources}
        cited_in_report = {ref.source_id for ref in input_data.report.references}

        # Deterministic citation audit before LLM evaluation
        citation_validations: list[CitationValidation] = []
        hallucinated_citations: list[str] = []

        for ref in input_data.report.references:
            if ref.source_id in known_sources:
                citation_validations.append(
                    CitationValidation(
                        source_id=ref.source_id,
                        verdict=CitationVerdict.SUPPORTED,
                        explanation=f"Valid source record found ({known_sources[ref.source_id].domain}).",
                    )
                )
            else:
                hallucinated_citations.append(ref.source_id)
                citation_validations.append(
                    CitationValidation(
                        source_id=ref.source_id,
                        verdict=CitationVerdict.UNSUPPORTED,
                        explanation=f"Cited source ID '{ref.source_id}' does not exist in collected research sources.",
                    )
                )

        user_content = (
            f"Original Objective: {input_data.plan.refined_objective}\n"
            f"Original Questions:\n- " + "\n- ".join(input_data.plan.research_questions) + "\n\n"
            f"Report Title: {input_data.report.title}\n"
            f"Report Word Count: {input_data.report.word_count}\n"
            f"Report Executive Summary:\n{input_data.report.executive_summary}\n\n"
            f"Report Findings Excerpt:\n{input_data.report.findings[:2000]}\n\n"
            f"Known Source IDs ({len(known_sources)}): {sorted(list(known_sources.keys()))}\n"
            f"Report Cited Source IDs: {sorted(list(cited_in_report))}\n\n"
            f"Evaluate report quality, coverage, citation fidelity, and provide recommendations."
        )

        request = LLMRequest(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=user_content)],
            temperature=0.2,
            response_schema=ReviewerOutput,
        )

        output, response = await self.provider.generate_structured(
            request,
            ReviewerOutput,
            model=self.model,
        )

        # Merge deterministic citation validations
        if citation_validations:
            output.citation_validations = citation_validations

        # If hallucinated citations were found, flag accuracy concern and adjust score
        if hallucinated_citations:
            output.accuracy_concerns.append(
                f"Report cited {len(hallucinated_citations)} unverified or missing source IDs: {hallucinated_citations}"
            )
            output.quality_score = min(output.quality_score, 65)
            output.approval_recommendation = False

        # Guardrail on approval: final approval strictly belongs to human user
        output.recommendation_notes = (
            f"{output.recommendation_notes.strip()} "
            "[Human Review Note: Final approval is reserved for the human user. "
            "The Reviewer does not grant autonomous approval.]"
        ).strip()

        meta = {
            "quality_score": output.quality_score,
            "completeness_score": output.completeness_score,
            "hallucinated_citations_count": len(hallucinated_citations),
            "approval_recommendation": output.approval_recommendation,
        }

        return output, response.usage, meta
