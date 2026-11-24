"""Data transfer models and base provider interfaces.

All future AI agents communicate through these normalized types, decoupling
agent business logic from provider SDK specifics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeVar
from pydantic import BaseModel, ConfigDict, Field

from app.core.tokens import TokenUsage

T = TypeVar("T")


class ChatMessage(BaseModel):
    """Normalized chat message."""
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """Standardized input request for text or structured generation."""
    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage] = Field(min_length=1)
    system_prompt: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    response_schema: Any | None = None  # Pydantic model class or JSON schema dict
    extra_params: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Normalized output from an LLM invocation."""
    content: str
    structured_data: Any | None = None
    usage: TokenUsage
    latency_ms: int
    provider: str
    model: str
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Normalized query parameters for search providers."""
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    max_results: int = Field(default=10, ge=1, le=50)
    search_depth: Literal["basic", "advanced"] = "basic"
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Normalized individual search hit."""
    url: str
    title: str
    content: str
    snippet: str | None = None
    published_date: str | None = None
    score: float | None = None
    domain: str = ""
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Standardized search response."""
    query: str
    results: list[SearchResult]
    total_results: int
    latency_ms: int
    provider: str = "tavily"


class ExtractRequest(BaseModel):
    """Normalized URL extraction request."""
    model_config = ConfigDict(extra="forbid")
    urls: list[str] = Field(min_length=1, max_length=20)


class ExtractResult(BaseModel):
    """Normalized extracted webpage content."""
    url: str
    raw_content: str
    extracted_text: str | None = None


class ExtractResponse(BaseModel):
    """Standardized extraction response."""
    results: list[ExtractResult]
    latency_ms: int
    provider: str = "tavily"


class BaseLLMProvider(ABC):
    """Abstract interface for LLM text and structured generation."""

    def __init__(self, provider_name: str, default_model: str) -> None:
        self.provider_name = provider_name
        self.default_model = default_model

    @abstractmethod
    async def generate_text(
        self, request: LLMRequest, model: str | None = None
    ) -> LLMResponse:
        """Generate unstructured text from messages."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        request: LLMRequest,
        response_schema: type[T],
        model: str | None = None,
    ) -> tuple[T, LLMResponse]:
        """Generate text and parse/validate against a typed Pydantic schema."""
        ...


class BaseSearchProvider(ABC):
    """Abstract interface for web research providers."""

    def __init__(self, provider_name: str = "tavily") -> None:
        self.provider_name = provider_name

    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a web search and return normalized results."""
        ...

    @abstractmethod
    async def extract(self, request: ExtractRequest) -> ExtractResponse:
        """Extract main content from specific URLs."""
        ...
