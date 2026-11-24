"""Deterministic mock providers for offline testing and zero-credit development."""

from __future__ import annotations

from enum import Enum
import json
from typing import Any, TypeVar, Union, get_args, get_origin
from urllib.parse import urlparse

from pydantic import BaseModel

from app.core.tokens import TokenUsage, calculate_cost
from app.providers.base import (
    BaseLLMProvider,
    BaseSearchProvider,
    ExtractRequest,
    ExtractResponse,
    ExtractResult,
    LLMRequest,
    LLMResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.providers.guardrails import (
    SearchQuotaTracker,
    validate_model_allowed,
    validate_token_limit,
)

T = TypeVar("T")


def _synthesize_mock_value(field_name: str, annotation: Any, depth: int = 0) -> Any:
    """Recursively synthesize a plausible mock value for an arbitrary type annotation."""
    if annotation is None:
        return None

    origin = get_origin(annotation)
    if origin is list:
        if depth >= 1:
            return []
        args = get_args(annotation)
        elem_type = args[0] if args else str
        return [_synthesize_mock_value(f"{field_name}_item", elem_type, depth=depth + 1)]

    if origin is dict:
        return {"key": "value"}

    if origin is Union or str(origin) == "typing.Union":
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return _synthesize_mock_value(field_name, args[0], depth=depth)
        return None

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            if depth >= 2:
                return None
            return _synthesize_mock_model(annotation, depth=depth + 1)
        if issubclass(annotation, Enum):
            members = list(annotation)
            return members[0]
        if issubclass(annotation, bool) or annotation is bool:
            return True
        if issubclass(annotation, str):
            lower = field_name.lower()
            if "url" in lower:
                return "https://example.com/source/1"
            if "domain" in lower:
                return "example.com"
            if "id" in lower:
                return "src_01"
            return f"Mock {field_name.replace('_', ' ').title()}"
        if issubclass(annotation, int):
            lower = field_name.lower()
            if "score" in lower or "quality" in lower or "completeness" in lower:
                return 88
            if "count" in lower:
                return 250
            return 5
        if issubclass(annotation, float):
            return 0.95

    return f"Mock {field_name}"


def _synthesize_mock_model(model_cls: type[BaseModel], depth: int = 0) -> BaseModel:
    """Instantiate a Pydantic model with synthesized fields satisfying validation constraints."""
    sample_data: dict[str, Any] = {}
    for name, field_info in model_cls.model_fields.items():
        sample_data[name] = _synthesize_mock_value(name, field_info.annotation, depth=depth)
    try:
        return model_cls.model_validate(sample_data)
    except Exception:
        # Fallback to model construct if complex cross-field validation is present
        return model_cls.model_construct(**sample_data)


class MockGeminiProvider(BaseLLMProvider):
    """Deterministic mock for Google Gemini."""

    def __init__(
        self,
        default_model: str = "gemini-2.5-flash",
        *,
        free_only: bool = True,
        allow_paid_models: bool = False,
        custom_allowlist: str = "",
        max_request_tokens: int = 8192,
    ) -> None:
        super().__init__(provider_name="gemini", default_model=default_model)
        self._free_only = free_only
        self._allow_paid_models = allow_paid_models
        self._custom_allowlist = custom_allowlist
        self._max_request_tokens = max_request_tokens

    async def generate_text(self, request: LLMRequest, model: str | None = None) -> LLMResponse:
        target_model = model or self.default_model
        validate_token_limit(request.max_tokens, self._max_request_tokens)
        validate_model_allowed(
            target_model,
            self.provider_name,
            free_only=self._free_only,
            allow_paid_models=self._allow_paid_models,
            custom_allowlist=self._custom_allowlist,
        )
        prompt_preview = request.messages[-1].content[:60] if request.messages else "query"
        content = (
            f"[Mock Gemini] Comprehensive analysis for '{prompt_preview}'. "
            "Key considerations include architectural modularity, source verification, "
            "and structured multi-agent coordination."
        )
        usage = TokenUsage(
            prompt_tokens=120,
            completion_tokens=85,
            total_tokens=205,
            cost_estimate=calculate_cost(target_model, 120, 85),
        )
        return LLMResponse(
            content=content,
            usage=usage,
            latency_ms=15,
            provider=self.provider_name,
            model=target_model,
            raw_metadata={"mock": True},
        )

    async def generate_structured(
        self,
        request: LLMRequest,
        response_schema: type[T],
        model: str | None = None,
    ) -> tuple[T, LLMResponse]:
        target_model = model or self.default_model
        validate_token_limit(request.max_tokens, self._max_request_tokens)
        validate_model_allowed(
            target_model,
            self.provider_name,
            free_only=self._free_only,
            allow_paid_models=self._allow_paid_models,
            custom_allowlist=self._custom_allowlist,
        )

        # Attempt to synthesize a valid mock object from the schema
        structured: Any = None
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            structured = _synthesize_mock_model(response_schema)
            content = json.dumps(structured.model_dump())
        elif isinstance(response_schema, dict):
            structured = {"status": "ok", "mock_plan": ["Section 1", "Section 2"]}
            content = json.dumps(structured)
        else:
            structured = {"result": "mock_data"}
            content = json.dumps(structured)

        usage = TokenUsage(
            prompt_tokens=150,
            completion_tokens=110,
            total_tokens=260,
            cost_estimate=calculate_cost(target_model, 150, 110),
        )
        response = LLMResponse(
            content=content,
            structured_data=structured,
            usage=usage,
            latency_ms=18,
            provider=self.provider_name,
            model=target_model,
            raw_metadata={"mock": True},
        )
        return structured, response


class MockGroqProvider(BaseLLMProvider):
    """Deterministic mock for Groq."""

    def __init__(
        self,
        default_model: str = "mock-groq-model",
        *,
        free_only: bool = True,
        allow_paid_models: bool = False,
        custom_allowlist: str = "mock-groq-model",
        max_request_tokens: int = 8192,
    ) -> None:
        super().__init__(provider_name="groq", default_model=default_model)
        self._free_only = free_only
        self._allow_paid_models = allow_paid_models
        self._custom_allowlist = custom_allowlist or "mock-groq-model"
        self._max_request_tokens = max_request_tokens

    async def generate_text(self, request: LLMRequest, model: str | None = None) -> LLMResponse:
        target_model = model or self.default_model
        validate_token_limit(request.max_tokens, self._max_request_tokens)
        validate_model_allowed(
            target_model,
            self.provider_name,
            free_only=self._free_only,
            allow_paid_models=self._allow_paid_models,
            custom_allowlist=self._custom_allowlist,
        )
        content = (
            "[Mock Groq] Synthesized draft content. The findings demonstrate robust "
            "performance with high factual grounding across cited external literature."
        )
        usage = TokenUsage(
            prompt_tokens=95,
            completion_tokens=180,
            total_tokens=275,
            cost_estimate=calculate_cost(target_model, 95, 180),
        )
        return LLMResponse(
            content=content,
            usage=usage,
            latency_ms=8,
            provider=self.provider_name,
            model=target_model,
            raw_metadata={"mock": True},
        )

    async def generate_structured(
        self,
        request: LLMRequest,
        response_schema: type[T],
        model: str | None = None,
    ) -> tuple[T, LLMResponse]:
        target_model = model or self.default_model
        validate_token_limit(request.max_tokens, self._max_request_tokens)
        validate_model_allowed(
            target_model,
            self.provider_name,
            free_only=self._free_only,
            allow_paid_models=self._allow_paid_models,
            custom_allowlist=self._custom_allowlist,
        )
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            structured = _synthesize_mock_model(response_schema)
            content = json.dumps(structured.model_dump())
        else:
            structured = {"summary": "Mock summary", "confidence": 0.9}
            content = json.dumps(structured)

        usage = TokenUsage(
            prompt_tokens=110,
            completion_tokens=140,
            total_tokens=250,
            cost_estimate=calculate_cost(target_model, 110, 140),
        )
        response = LLMResponse(
            content=content,
            structured_data=structured,
            usage=usage,
            latency_ms=9,
            provider=self.provider_name,
            model=target_model,
            raw_metadata={"mock": True},
        )
        return structured, response


class MockTavilyProvider(BaseSearchProvider):
    """Deterministic mock for Tavily web search."""

    def __init__(
        self,
        quota_tracker: SearchQuotaTracker | None = None,
        monthly_budget: int = 1000,
    ) -> None:
        super().__init__(provider_name="tavily")
        self._tracker = quota_tracker or SearchQuotaTracker.get_instance(monthly_budget)

    async def search(self, request: SearchRequest) -> SearchResponse:
        self._tracker.check_and_increment(1)
        domains = request.include_domains or ["example.com", "research.org", "techcrunch.com"]
        count = min(request.max_results, 5)

        results: list[SearchResult] = []
        for i in range(1, count + 1):
            domain = domains[(i - 1) % len(domains)]
            url = f"https://{domain}/article/{i}?query={request.query.replace(' ', '+')}"
            snippet = (
                f"Detailed research publication #{i} covering '{request.query}'. "
                f"Key verified insights and analysis regarding domain developments."
            )
            results.append(
                SearchResult(
                    url=url,
                    title=f"Research on {request.query} - Source {i}",
                    content=snippet,
                    snippet=snippet,
                    published_date="2026-01-15T08:00:00Z",
                    score=round(0.98 - (i * 0.05), 2),
                    domain=domain,
                    raw_metadata={"mock": True},
                )
            )

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            latency_ms=12,
            provider=self.provider_name,
        )

    async def extract(self, request: ExtractRequest) -> ExtractResponse:
        self._tracker.check_and_increment(len(request.urls))
        results: list[ExtractResult] = []
        for url in request.urls:
            domain = urlparse(url).netloc
            results.append(
                ExtractResult(
                    url=url,
                    raw_content=f"<html><body><h1>Extracted content from {domain}</h1><p>Full article text.</p></body></html>",
                    extracted_text=f"Full article text extracted from {url}. Contains complete source context.",
                )
            )

        return ExtractResponse(
            results=results,
            latency_ms=10,
            provider=self.provider_name,
        )
