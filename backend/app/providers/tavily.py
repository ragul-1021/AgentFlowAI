"""Asynchronous Tavily web search and content extraction integration."""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from app.providers.base import (
    BaseSearchProvider,
    ExtractRequest,
    ExtractResponse,
    ExtractResult,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.providers.exceptions import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from app.providers.guardrails import SearchQuotaTracker
from app.providers.ratelimit import ConcurrencyLimiter
from app.providers.retry import execute_with_retry

logger = logging.getLogger(__name__)

TAVILY_API_BASE = "https://api.tavily.com"


class TavilyProvider(BaseSearchProvider):
    """Asynchronous Tavily web research provider implementation."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        limiter: ConcurrencyLimiter | None = None,
        quota_tracker: SearchQuotaTracker | None = None,
        monthly_budget: int = 1000,
        quota_storage_path: str | None = None,
    ) -> None:
        super().__init__(provider_name="tavily")
        self._api_key = api_key.strip() if api_key else ""
        self._client = client
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._limiter = limiter
        self._tracker = quota_tracker or SearchQuotaTracker.get_instance(
            monthly_budget,
            persistence_path=quota_storage_path,
        )

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise ProviderAuthError(
                "Tavily API key is not configured.", provider=self.provider_name
            )

    def _handle_response_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return

        status = response.status_code
        retry_after: float | None = None
        header_val = response.headers.get("retry-after")
        if header_val:
            try:
                retry_after = float(header_val)
            except ValueError:
                pass

        try:
            err_json = response.json()
            message = err_json.get("detail", {}).get("error", response.text)
        except Exception:
            message = response.text or f"HTTP {status}"

        clean_msg = f"Tavily API returned error: {message[:300]}"

        if status in (401, 403):
            raise ProviderAuthError(clean_msg, provider=self.provider_name, status_code=status)
        if status == 429:
            raise ProviderRateLimitError(
                clean_msg,
                provider=self.provider_name,
                status_code=status,
                retry_after=retry_after,
            )
        if status == 400:
            raise ProviderBadRequestError(clean_msg, provider=self.provider_name, status_code=status)
        if status >= 500:
            raise ProviderUnavailableError(clean_msg, provider=self.provider_name, status_code=status)

        raise ProviderError(clean_msg, provider=self.provider_name, status_code=status)

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        self._ensure_api_key()
        url = f"{TAVILY_API_BASE}/{endpoint}"
        payload["api_key"] = self._api_key

        async def _call() -> tuple[dict[str, Any], int]:
            started = time.perf_counter()
            client = self._client or httpx.AsyncClient(timeout=self._timeout)
            try:
                if self._limiter:
                    async with self._limiter.acquire(self.provider_name):
                        resp = await client.post(url, json=payload, timeout=self._timeout)
                else:
                    resp = await client.post(url, json=payload, timeout=self._timeout)

                self._handle_response_status(resp)
                latency = max(1, int((time.perf_counter() - started) * 1000))
                return resp.json(), latency
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(
                    f"Tavily request timed out after {self._timeout}s",
                    provider=self.provider_name,
                ) from exc
            finally:
                if self._client is None:
                    await client.aclose()

        return await execute_with_retry(
            _call,
            provider=self.provider_name,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        self._tracker.check_and_increment(1)
        payload: dict[str, Any] = {
            "query": request.query,
            "search_depth": request.search_depth,
            "max_results": request.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        if request.include_domains:
            payload["include_domains"] = request.include_domains
        if request.exclude_domains:
            payload["exclude_domains"] = request.exclude_domains

        data, latency = await self._post("search", payload)

        raw_results = data.get("results", [])
        results: list[SearchResult] = []
        for item in raw_results:
            raw_url = item.get("url", "")
            domain = urlparse(raw_url).netloc.lower()
            snippet = item.get("content") or item.get("snippet", "")
            results.append(
                SearchResult(
                    url=raw_url,
                    title=item.get("title", ""),
                    content=snippet,
                    snippet=snippet,
                    published_date=item.get("published_date"),
                    score=item.get("score"),
                    domain=domain,
                    raw_metadata={"raw_score": item.get("score")},
                )
            )

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            latency_ms=latency,
            provider=self.provider_name,
        )

    async def extract(self, request: ExtractRequest) -> ExtractResponse:
        self._tracker.check_and_increment(len(request.urls))
        payload = {"urls": request.urls}
        data, latency = await self._post("extract", payload)

        raw_results = data.get("results", [])
        results: list[ExtractResult] = []
        for item in raw_results:
            results.append(
                ExtractResult(
                    url=item.get("url", ""),
                    raw_content=item.get("raw_content", ""),
                    extracted_text=item.get("content"),
                )
            )

        return ExtractResponse(
            results=results,
            latency_ms=latency,
            provider=self.provider_name,
        )
