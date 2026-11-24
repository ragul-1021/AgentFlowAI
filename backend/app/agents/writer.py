"""Writer Agent for AgentFlow AI.

Drafts structured, comprehensive research reports based on analyst findings and source evidence,
using verified free-tier models (default: gemini-2.5-flash).
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.agents.context import AgentContext
from app.agents.schemas import ReferenceItem, ReportSection, WriterInput, WriterOutput
from app.config import Settings, get_settings
from app.core.tokens import TokenUsage
from app.enums import AgentName
from app.providers.base import BaseLLMProvider, ChatMessage, LLMRequest

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are the Senior Scientific and Technical Report Writer for AgentFlow AI.
Your role is to write a comprehensive, authoritative, and publication-ready research report.

Citation & Reference Rules:
1. Every factual statement must cite supporting sources using their exact ID (e.g., [src_01], [src_02]).
2. Use ONLY source IDs and URLs provided in the input. Never invent citations or external URLs.
3. Organize the report into logical, detailed sections following the approved outline.
4. Conclude with a rigorous discussion of limitations and next steps."""


def _calculate_total_word_count(output: WriterOutput) -> int:
    """Calculate total words across all narrative sections and subsections."""
    text_blocks = [
        output.title,
        output.executive_summary,
        output.introduction,
        output.methodology,
        output.findings,
        output.discussion,
        output.limitations,
        output.conclusion,
    ]
    for section in output.main_sections:
        text_blocks.append(section.content)
        for sub in section.subsections:
            text_blocks.append(sub.content)

    full_text = " ".join(text_blocks)
    words = [w for w in full_text.split() if w.strip()]
    return len(words)


class WriterAgent(BaseAgent[WriterInput, WriterOutput]):
    """Agent responsible for writing the complete research report with verified citations."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        target_model = model or cfg.writer_model
        super().__init__(AgentName.WRITER, provider, model=target_model, settings=cfg)

    async def _execute(
        self,
        input_data: WriterInput,
        context: AgentContext,
    ) -> tuple[WriterOutput, TokenUsage, dict[str, Any]]:
        source_map = {src.source_id: src for src in input_data.sources}

        # Format findings for writer prompt
        findings_text = []
        for f in input_data.analyst_output.findings:
            sources_str = ", ".join(f.supporting_source_ids)
            findings_text.append(f"- [{f.category.value.upper()}] {f.statement} (Sources: {sources_str})\n  Explanation: {f.explanation}")

        sources_ref_text = []
        for src in input_data.sources:
            sources_ref_text.append(f"[{src.source_id}] {src.title} | URL: {src.url}")

        user_content = (
            f"Report Title Proposal: {input_data.plan.title}\n"
            f"Format: {input_data.format.value} | Length Target: {input_data.length.value}\n"
            f"Research Objective: {input_data.plan.refined_objective}\n\n"
            f"Recommended Outline:\n- " + "\n- ".join(input_data.plan.suggested_outline) + "\n\n"
            f"Key Insights from Analyst:\n- " + "\n- ".join(input_data.analyst_output.key_insights) + "\n\n"
            f"Detailed Findings with Sources:\n" + "\n".join(findings_text) + "\n\n"
            f"Available Sources for Reference:\n" + "\n".join(sources_ref_text) + "\n\n"
        )
        if input_data.revision_feedback:
            user_content += (
                "CRITICAL REVISION INSTRUCTIONS FROM REVIEWER:\n"
                + "\n- ".join(input_data.revision_feedback)
                + "\nPlease address each critique thoroughly in this revised edition.\n\n"
            )

        user_content += "Write the complete structured report. Ensure references strictly match the available sources."

        request = LLMRequest(
            system_prompt=WRITER_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=user_content)],
            temperature=0.4,
            response_schema=WriterOutput,
        )

        report, response = await self.provider.generate_structured(
            request,
            WriterOutput,
            model=self.model,
        )

        # Ensure references preserve canonical URLs from researcher sources
        verified_references: list[ReferenceItem] = []
        for ref in report.references:
            if ref.source_id in source_map:
                src = source_map[ref.source_id]
                verified_references.append(
                    ReferenceItem(
                        source_id=src.source_id,
                        title=src.title,
                        url=src.url,
                        citation_label=ref.citation_label or f"[{src.source_id}]",
                    )
                )

        # If LLM omitted references list, construct from available sources used
        if not verified_references and input_data.sources:
            for idx, src in enumerate(input_data.sources[:10], start=1):
                verified_references.append(
                    ReferenceItem(
                        source_id=src.source_id,
                        title=src.title,
                        url=src.url,
                        citation_label=f"[{idx}]",
                    )
                )

        report.references = verified_references
        report.word_count = _calculate_total_word_count(report)

        meta = {
            "word_count": report.word_count,
            "sections_count": len(report.main_sections),
            "references_count": len(report.references),
        }

        return report, response.usage, meta
