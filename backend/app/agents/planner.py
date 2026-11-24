"""Planner Agent for AgentFlow AI.

Converts a user's research objective into a structured, validated research plan
using Google Gemini free-tier models.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.agents.context import AgentContext
from app.agents.exceptions import AgentValidationError
from app.agents.schemas import PlannerInput, PlannerOutput
from app.config import Settings, get_settings
from app.core.tokens import TokenUsage
from app.enums import AgentName
from app.providers.base import BaseLLMProvider, ChatMessage, LLMRequest

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the Senior Research Planning Agent for AgentFlow AI.
Your responsibility is to convert a user's research objective into a rigorous, well-scoped, and highly structured research plan.

Instructions:
1. Title: Create a concise, professional title.
2. Refined Objective: Formulate an actionable, comprehensive research objective statement.
3. Research Questions: Formulate focused, critical research questions (non-overlapping and clear).
4. Subtopics: Define specific subtopics necessary to answer the questions.
5. Search Queries: Generate precise, non-duplicated search queries designed for web retrieval.
6. Scope: Define inclusions and boundaries.
7. Exclusions: Define explicit out-of-scope topics or tangents to prevent scope creep.
8. Suggested Outline: Propose a logical section outline for the final report.

Constraint: Return ONLY valid structured JSON matching the requested schema. Do not execute searches yourself."""


class PlannerAgent(BaseAgent[PlannerInput, PlannerOutput]):
    """Agent responsible for structuring research objectives into actionable plans."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        target_model = model or cfg.planner_model
        super().__init__(AgentName.PLANNER, provider, model=target_model, settings=cfg)

    async def _execute(
        self,
        input_data: PlannerInput,
        context: AgentContext,
    ) -> tuple[PlannerOutput, TokenUsage, dict[str, Any]]:
        # Validate input constraints
        if not input_data.topic.strip():
            raise AgentValidationError("Research topic cannot be blank.", agent_name=str(self.agent_name))

        user_content = (
            f"Research Topic: {input_data.topic}\n"
            f"Stated Objective: {input_data.objective or 'Comprehensive investigation'}\n"
            f"Depth Tier: {input_data.depth.value}\n"
            f"Max Research Questions: {input_data.max_questions}\n"
            f"Max Search Queries: {input_data.max_queries}\n"
        )
        if input_data.user_instructions:
            user_content += f"User Instructions / Constraints: {input_data.user_instructions}\n"

        request = LLMRequest(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=user_content)],
            temperature=0.3,
            response_schema=PlannerOutput,
        )

        plan_output, response = await self.provider.generate_structured(
            request,
            PlannerOutput,
            model=self.model,
        )

        # Enforce bounds on generated questions and queries
        if len(plan_output.research_questions) > input_data.max_questions:
            plan_output.research_questions = plan_output.research_questions[: input_data.max_questions]

        if len(plan_output.search_queries) > input_data.max_queries:
            plan_output.search_queries = plan_output.search_queries[: input_data.max_queries]

        meta = {
            "questions_count": len(plan_output.research_questions),
            "queries_count": len(plan_output.search_queries),
            "depth": input_data.depth.value,
        }

        return plan_output, response.usage, meta
