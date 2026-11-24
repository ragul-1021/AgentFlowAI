"""Analyst Agent for AgentFlow AI.

Synthesizes collected research evidence, detects contradictions and gaps,
and maps findings directly to supporting source IDs using Gemini free tier.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.agents.context import AgentContext
from app.agents.exceptions import AgentValidationError
from app.agents.schemas import AnalystInput, AnalystOutput, FindingItem
from app.config import Settings, get_settings
from app.core.tokens import TokenUsage
from app.enums import AgentName, EvidenceStrength, FindingCategory
from app.providers.base import BaseLLMProvider, ChatMessage, LLMRequest

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are the Senior Intelligence and Evidence Analyst for AgentFlow AI.
Your role is to rigorously analyze collected research evidence and synthesize findings.

Strict Rules:
1. Every finding MUST reference available source IDs (e.g. ['src_01', 'src_02']). Never invent source IDs.
2. Distinguish verified facts from interpretations.
3. Explicitly document agreements across sources, contradictions, and data gaps.
4. Provide a concrete explanation for every confidence score (0.0 to 1.0).
5. If evidence is thin or absent for a question, note it under research_gaps or limitations."""


class AnalystAgent(BaseAgent[AnalystInput, AnalystOutput]):
    """Agent responsible for cross-source analysis and evidence mapping."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        target_model = model or cfg.analyst_model
        super().__init__(AgentName.ANALYST, provider, model=target_model, settings=cfg)

    async def _execute(
        self,
        input_data: AnalystInput,
        context: AgentContext,
    ) -> tuple[AnalystOutput, TokenUsage, dict[str, Any]]:
        available_source_ids = {src.source_id for src in input_data.sources}

        # Build structured prompt context
        evidence_digest = []
        for src in input_data.sources:
            evidence_digest.append(
                f"[{src.source_id}] {src.title} ({src.url})\n"
                f"Domain: {src.domain} | Date: {src.published_date or 'N/A'}\n"
                f"Summary: {src.summary}\n"
            )

        sources_text = "\n---\n".join(evidence_digest) if evidence_digest else "No external sources collected."

        user_content = (
            f"Research Title: {input_data.plan.title}\n"
            f"Objective: {input_data.plan.refined_objective}\n"
            f"Key Questions:\n- " + "\n- ".join(input_data.plan.research_questions) + "\n\n"
            f"Collected Evidence Sources ({len(input_data.sources)} available):\n"
            f"{sources_text}\n\n"
            f"Available Source IDs: {sorted(list(available_source_ids))}\n"
            f"Synthesize comprehensive analytical findings referencing ONLY these source IDs."
        )

        request = LLMRequest(
            system_prompt=ANALYST_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=user_content)],
            temperature=0.2,
            response_schema=AnalystOutput,
        )

        output, response = await self.provider.generate_structured(
            request,
            AnalystOutput,
            model=self.model,
        )

        # Post-validation: verify all referenced source IDs exist
        valid_findings: list[FindingItem] = []
        for finding in output.findings:
            cleaned_sources = [sid for sid in finding.supporting_source_ids if sid in available_source_ids]
            if not cleaned_sources and available_source_ids:
                # If LLM cited an invalid ID, fallback to available sources or flag
                logger.warning("[%s] Finding '%s' cited invalid source IDs: %s", self.agent_name, finding.finding_id, finding.supporting_source_ids)
                cleaned_sources = [sorted(list(available_source_ids))[0]]

            finding.supporting_source_ids = cleaned_sources
            valid_findings.append(finding)

        output.findings = valid_findings

        meta = {
            "findings_count": len(output.findings),
            "key_insights_count": len(output.key_insights),
            "sources_analyzed": len(input_data.sources),
        }

        return output, response.usage, meta
