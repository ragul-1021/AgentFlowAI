"""High-level research orchestrator coordinating agents, sessions, and runs."""

from __future__ import annotations

import logging
from typing import Any
import uuid

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents import (
    AnalystAgent,
    PlannerAgent,
    ResearcherAgent,
    ReviewerAgent,
    WriterAgent,
)
from app.config import Settings, get_settings
from app.orchestration.exceptions import PipelineError, PipelineValidationError
from app.orchestration.pipeline import PipelineResult, ResearchPipeline
from app.orchestration.state import PipelineState
from app.providers.base import BaseLLMProvider, BaseSearchProvider
from app.providers.factory import (
    create_gemini_provider,
    create_tavily_provider,
)
from app.providers.guardrails import validate_model_allowed
from app.schemas import TaskCreate

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Coordinates agent instantiation, model verification, and pipeline runs."""

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        settings: Settings | None = None,
        llm_provider: BaseLLMProvider | None = None,
        search_provider: BaseSearchProvider | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.settings = settings or get_settings()
        self._active_runs: dict[uuid.UUID, PipelineState] = {}

        # 1. Resolve providers: use configured Gemini provider for all five agents
        # (Groq is not required; Tavily handles search and extraction)
        self.llm_provider = llm_provider or create_gemini_provider(
            self.settings,
            client=http_client,
        )
        self.search_provider = search_provider or create_tavily_provider(
            self.settings,
            client=http_client,
        )

        # 2. Strict model validation (Mandatory correction 1)
        # Read model ID from GEMINI_MODEL (settings.gemini_model).
        # Verify actual model availability and free-tier eligibility.
        # Fail safely if the configured model is unverified or paid; never silently substitute.
        configured_gemini_model = self.settings.gemini_model
        self.gemini_model = configured_gemini_model
        validate_model_allowed(
            configured_gemini_model,
            "gemini",
            free_only=self.settings.free_only,
            allow_paid_models=self.settings.allow_paid_models,
            custom_allowlist=self.settings.free_model_allowlist,
        )

        # 3. Initialize the five agents using the verified free Gemini provider
        self.planner_agent = PlannerAgent(
            self.llm_provider,
            model=configured_gemini_model,
            settings=self.settings,
        )
        self.researcher_agent = ResearcherAgent(
            search_provider=self.search_provider,
            llm_provider=self.llm_provider,
            model=self.settings.researcher_model or configured_gemini_model,
            settings=self.settings,
        )
        self.analyst_agent = AnalystAgent(
            self.llm_provider,
            model=self.settings.analyst_model or configured_gemini_model,
            settings=self.settings,
        )
        self.writer_agent = WriterAgent(
            self.llm_provider,
            model=self.settings.writer_model or configured_gemini_model,
            settings=self.settings,
        )
        self.reviewer_agent = ReviewerAgent(
            self.llm_provider,
            model=self.settings.reviewer_model or configured_gemini_model,
            settings=self.settings,
        )

        # 4. Assemble the research pipeline
        self.pipeline = ResearchPipeline(
            planner_agent=self.planner_agent,
            researcher_agent=self.researcher_agent,
            analyst_agent=self.analyst_agent,
            writer_agent=self.writer_agent,
            reviewer_agent=self.reviewer_agent,
            sessionmaker=self.sessionmaker,
            settings=self.settings,
        )

    def get_state(self, task_id: uuid.UUID | str) -> PipelineState | None:
        """Retrieve the in-flight state of a task by ID."""
        tid = uuid.UUID(str(task_id))
        return self._active_runs.get(tid)

    def cancel(self, task_id: uuid.UUID | str) -> bool:
        """Signal an active pipeline to halt between stages."""
        tid = uuid.UUID(str(task_id))
        state = self._active_runs.get(tid)
        if state:
            state.request_cancellation()
            return True
        return False

    async def run(
        self,
        request: TaskCreate,
        task_id: uuid.UUID | None = None,
    ) -> PipelineResult:
        """Execute a full research task through the pipeline."""
        tid = task_id or uuid.uuid4()
        state = PipelineState(task_id=tid)
        self._active_runs[tid] = state

        try:
            result = await self.pipeline.execute(request, state=state)
            return result
        finally:
            self._active_runs.pop(tid, None)
