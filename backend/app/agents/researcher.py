"""Researcher Agent for AgentFlow AI.

Executes search queries across external web sources via Tavily, extracts content,
removes duplicate URLs, filters by domain, and summarizes evidence with Groq.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from app.agents.base import BaseAgent
from app.agents.context import AgentContext
from app.agents.exceptions import AgentValidationError
from app.agents.schemas import ResearcherInput, ResearcherOutput, SourceItem
from app.config import Settings, get_settings
from app.core.tokens import TokenUsage
from app.enums import AgentName
from app.providers.base import (
    BaseLLMProvider,
    BaseSearchProvider,
    ChatMessage,
    ExtractRequest,
    LLMRequest,
    SearchRequest,
    SearchResult,
)

logger = logging.getLogger(__name__)

SUMMARIZER_PROMPT = """You are a research analyst summarizer.
Summarize the factual evidence in the provided webpage text clearly and concisely.
- Do NOT invent facts or external citations.
- Focus strictly on verified data, findings, and technical claims.
- Keep the summary between 2 and 4 paragraphs."""


def _canonical_url(url: str) -> str:
    """Normalize URL by removing trailing slashes, fragments, and common query trackers."""
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/")
    # Build clean canonical string
    clean = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"
    return clean


class ResearcherAgent(BaseAgent[ResearcherInput, ResearcherOutput]):
    """Agent responsible for gathering and synthesizing external research evidence."""

    def __init__(
        self,
        search_provider: BaseSearchProvider,
        llm_provider: BaseLLMProvider,
        *,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        target_model = model or cfg.researcher_model
        super().__init__(AgentName.RESEARCHER, llm_provider, model=target_model, settings=cfg)
        self.search_provider = search_provider

    async def _execute(
        self,
        input_data: ResearcherInput,
        context: AgentContext,
    ) -> tuple[ResearcherOutput, TokenUsage, dict[str, Any]]:
        if not input_data.search_queries:
            raise AgentValidationError(
                "ResearcherInput requires at least one search query.",
                agent_name=str(self.agent_name),
            )

        total_usage = TokenUsage()
        queries_to_run = input_data.search_queries[:15]  # Safety bound
        raw_results: list[SearchResult] = []
        queries_executed: list[str] = []

        # 1. Execute search queries via Tavily
        for query in queries_to_run:
            search_req = SearchRequest(
                query=query,
                max_results=min(input_data.max_sources, 10),
                include_domains=input_data.preferred_domains,
                exclude_domains=input_data.excluded_domains,
            )
            try:
                search_resp = await self.search_provider.search(search_req)
                queries_executed.append(query)
                raw_results.extend(search_resp.results)
            except Exception as exc:
                logger.warning("[%s] Search query '%s' failed: %s", self.agent_name, query, exc)
                continue

        # 2. Deduplicate by canonical URL
        seen_canonical: set[str] = set()
        deduped_results: list[SearchResult] = []
        excluded_domains_set = {d.lower() for d in input_data.excluded_domains}

        for hit in raw_results:
            canon = _canonical_url(hit.url)
            domain = hit.domain.lower() if hit.domain else urlparse(hit.url).netloc.lower()

            if domain in excluded_domains_set:
                continue

            if canon not in seen_canonical:
                seen_canonical.add(canon)
                deduped_results.append(hit)

        filtered_count = len(raw_results) - len(deduped_results)
        target_slice = deduped_results[: input_data.max_sources]

        # 3. Optional content extraction for deep text
        extracted_content_map: dict[str, str] = {}
        if input_data.extract_content and target_slice:
            urls_to_extract = [item.url for item in target_slice[:10]]
            try:
                extract_resp = await self.search_provider.extract(ExtractRequest(urls=urls_to_extract))
                for ext in extract_resp.results:
                    if ext.extracted_text and len(ext.extracted_text.strip()) > 50:
                        extracted_content_map[ext.url] = ext.extracted_text.strip()
            except Exception as exc:
                logger.warning("[%s] Extraction skipped or failed: %s", self.agent_name, exc)

        # 4. Synthesize SourceItem objects with factual summaries
        sources: list[SourceItem] = []
        for idx, item in enumerate(target_slice, start=1):
            src_id = f"src_{idx:02d}"
            extracted_text = extracted_content_map.get(
                item.url,
                item.content or "[Extraction unavailable for this source]",
            )

            # Generate concise summary with LLM if substantial content is present
            summary_text = item.snippet or item.content[:300]
            if len(extracted_text) > 200 and extracted_text != "[Extraction unavailable for this source]":
                try:
                    summary_prompt = f"Source Title: {item.title}\nURL: {item.url}\n\nContent:\n{extracted_text[:3000]}"
                    llm_req = LLMRequest(
                        system_prompt=SUMMARIZER_PROMPT,
                        messages=[ChatMessage(role="user", content=summary_prompt)],
                        max_tokens=400,
                        temperature=0.2,
                    )
                    llm_resp = await self.provider.generate_text(llm_req, model=self.model)
                    summary_text = llm_resp.content.strip()
                    total_usage = total_usage + llm_resp.usage
                except Exception as exc:
                    logger.debug("[%s] LLM summarization failed for %s, falling back to snippet: %s", self.agent_name, src_id, exc)
                    summary_text = item.snippet or item.content[:300]

            domain = item.domain or urlparse(item.url).netloc.lower()
            source_item = SourceItem(
                source_id=src_id,
                title=item.title or f"Source from {domain}",
                url=item.url,
                domain=domain,
                published_date=item.published_date,
                search_score=item.score,
                extracted_content=extracted_text,
                summary=summary_text,
                related_questions=input_data.research_questions[:2],
            )
            sources.append(source_item)

        output = ResearcherOutput(
            sources=sources,
            total_sources_found=len(raw_results),
            deduplicated_count=len(deduped_results),
            filtered_count=filtered_count,
            queries_executed=queries_executed,
        )

        meta = {
            "queries_executed": len(queries_executed),
            "sources_collected": len(sources),
            "total_found": len(raw_results),
        }

        return output, total_usage, meta
