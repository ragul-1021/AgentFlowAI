"""Unit tests for Gemini, Groq, Tavily, and Mock AI providers."""

from __future__ import annotations

import json
import pytest
import httpx
from pydantic import BaseModel

from app.config import Settings
from app.providers import (
    ChatMessage,
    ExtractRequest,
    GeminiProvider,
    GroqProvider,
    LLMRequest,
    MockGeminiProvider,
    MockGroqProvider,
    MockTavilyProvider,
    ProviderAuthError,
    ProviderBadRequestError,
    SearchRequest,
    StructuredOutputError,
    TavilyProvider,
    create_gemini_provider,
    create_groq_provider,
    create_tavily_provider,
)


class ResearchOutline(BaseModel):
    title: str
    sections: list[str]
    confidence: float


@pytest.mark.asyncio
async def test_gemini_text_generation_success() -> None:
    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Gemini generated research response."}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 25,
            "totalTokenCount": 75,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-goog-api-key") == "fake-gemini-key"
        body = json.loads(request.content)
        assert "contents" in body
        return httpx.Response(200, json=mock_response)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider("fake-gemini-key", client=client)

    req = LLMRequest(
        messages=[ChatMessage(role="user", content="Explain quantum computing")],
        system_prompt="You are an expert researcher.",
    )
    res = await provider.generate_text(req)

    assert res.content == "Gemini generated research response."
    assert res.provider == "gemini"
    assert res.usage.prompt_tokens == 50
    assert res.usage.completion_tokens == 25
    assert res.usage.total_tokens == 75
    assert res.usage.cost_estimate > 0


@pytest.mark.asyncio
async def test_gemini_structured_generation_success() -> None:
    expected_data = {
        "title": "Quantum Computing",
        "sections": ["Intro", "Qubits", "Applications"],
        "confidence": 0.95,
    }
    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(expected_data)}],
                    "role": "model",
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 60, "candidatesTokenCount": 40},
    }

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=mock_response))
    )
    provider = GeminiProvider("fake-gemini-key", client=client)

    req = LLMRequest(messages=[ChatMessage(role="user", content="Plan outline")])
    data, res = await provider.generate_structured(req, response_schema=ResearchOutline)

    assert isinstance(data, ResearchOutline)
    assert data.title == "Quantum Computing"
    assert len(data.sections) == 3
    assert res.structured_data == data


@pytest.mark.asyncio
async def test_gemini_structured_generation_malformed_json_raises() -> None:
    mock_response = {
        "candidates": [{"content": {"parts": [{"text": "Not valid json"}]}}]
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=mock_response))
    )
    provider = GeminiProvider("fake-gemini-key", client=client)

    req = LLMRequest(messages=[ChatMessage(role="user", content="test")])
    with pytest.raises(StructuredOutputError):
        await provider.generate_structured(req, response_schema=ResearchOutline)


@pytest.mark.asyncio
async def test_gemini_missing_api_key_raises_auth_error() -> None:
    provider = GeminiProvider(api_key="")
    with pytest.raises(ProviderAuthError, match="not configured"):
        await provider.generate_text(LLMRequest(messages=[ChatMessage(role="user", content="hi")]))


@pytest.mark.asyncio
async def test_gemini_auth_failure_status_401() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "API key not valid"}}))
    )
    provider = GeminiProvider("bad-key", client=client)
    with pytest.raises(ProviderAuthError):
        await provider.generate_text(LLMRequest(messages=[ChatMessage(role="user", content="hi")]))


@pytest.mark.asyncio
async def test_groq_text_generation_success() -> None:
    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Groq draft output."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 120, "total_tokens": 200},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer fake-groq-key"
        body = json.loads(request.content)
        assert body["model"] == "custom-free-model"
        return httpx.Response(200, json=mock_response)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider("fake-groq-key", default_model="custom-free-model", custom_allowlist="custom-free-model", client=client)

    req = LLMRequest(messages=[ChatMessage(role="user", content="Draft report")])
    res = await provider.generate_text(req)

    assert res.content == "Groq draft output."
    assert res.provider == "groq"
    assert res.usage.prompt_tokens == 80
    assert res.usage.completion_tokens == 120
    assert res.usage.total_tokens == 200


@pytest.mark.asyncio
async def test_groq_structured_generation_success() -> None:
    outline = {
        "title": "AI Trends",
        "sections": ["Overview", "Analysis"],
        "confidence": 0.9,
    }
    mock_response = {
        "choices": [{"message": {"content": json.dumps(outline)}}],
        "usage": {"prompt_tokens": 90, "completion_tokens": 40},
    }

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=mock_response))
    )
    provider = GroqProvider("fake-groq-key", default_model="custom-free-model", custom_allowlist="custom-free-model", client=client)

    req = LLMRequest(messages=[ChatMessage(role="user", content="Structured outline")])
    data, res = await provider.generate_structured(req, response_schema=ResearchOutline)

    assert isinstance(data, ResearchOutline)
    assert data.title == "AI Trends"
    assert res.structured_data == data


@pytest.mark.asyncio
async def test_groq_structured_generation_malformed_json_raises() -> None:
    mock_response = {"choices": [{"message": {"content": "{invalid_json"}}]}
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=mock_response))
    )
    provider = GroqProvider("fake-groq-key", default_model="custom-free-model", custom_allowlist="custom-free-model", client=client)

    with pytest.raises(StructuredOutputError):
        await provider.generate_structured(
            LLMRequest(messages=[ChatMessage(role="user", content="hi")]),
            response_schema=ResearchOutline,
        )


@pytest.mark.asyncio
async def test_groq_bad_request_status_400() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(400, json={"error": {"message": "Invalid model parameter"}}))
    )
    provider = GroqProvider("fake-key", default_model="custom-free-model", custom_allowlist="custom-free-model", client=client)
    with pytest.raises(ProviderBadRequestError):
        await provider.generate_text(LLMRequest(messages=[ChatMessage(role="user", content="hi")]))


@pytest.mark.asyncio
async def test_tavily_search_success() -> None:
    mock_response = {
        "query": "superconductors",
        "results": [
            {
                "url": "https://nature.com/articles/123",
                "title": "Superconductor discovery",
                "content": "Ambient pressure superconductivity analysis.",
                "published_date": "2026-02-01",
                "score": 0.97,
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_key"] == "fake-tavily-key"
        assert body["query"] == "superconductors"
        return httpx.Response(200, json=mock_response)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = TavilyProvider("fake-tavily-key", client=client)

    res = await provider.search(SearchRequest(query="superconductors", max_results=5))

    assert res.total_results == 1
    assert res.results[0].title == "Superconductor discovery"
    assert res.results[0].domain == "nature.com"
    assert res.results[0].score == 0.97


@pytest.mark.asyncio
async def test_tavily_extract_success() -> None:
    mock_response = {
        "results": [
            {
                "url": "https://example.com/doc",
                "raw_content": "<html><body>Hello</body></html>",
                "content": "Extracted text content",
            }
        ]
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=mock_response))
    )
    provider = TavilyProvider("fake-tavily-key", client=client)

    res = await provider.extract(ExtractRequest(urls=["https://example.com/doc"]))
    assert len(res.results) == 1
    assert res.results[0].extracted_text == "Extracted text content"


@pytest.mark.asyncio
async def test_mock_gemini_text_and_structured() -> None:
    mock_gemini = MockGeminiProvider()
    res = await mock_gemini.generate_text(LLMRequest(messages=[ChatMessage(role="user", content="test")]))
    assert "[Mock Gemini]" in res.content
    assert res.usage.total_tokens > 0

    data, s_res = await mock_gemini.generate_structured(
        LLMRequest(messages=[ChatMessage(role="user", content="outline")]),
        response_schema=ResearchOutline,
    )
    assert isinstance(data, ResearchOutline)
    assert s_res.structured_data == data


@pytest.mark.asyncio
async def test_mock_groq_text_and_structured() -> None:
    mock_groq = MockGroqProvider()
    res = await mock_groq.generate_text(LLMRequest(messages=[ChatMessage(role="user", content="test")]))
    assert "[Mock Groq]" in res.content

    data, _ = await mock_groq.generate_structured(
        LLMRequest(messages=[ChatMessage(role="user", content="outline")]),
        response_schema=ResearchOutline,
    )
    assert isinstance(data, ResearchOutline)


@pytest.mark.asyncio
async def test_mock_tavily_search_and_extract() -> None:
    mock_tavily = MockTavilyProvider()
    search_res = await mock_tavily.search(SearchRequest(query="AI safety", max_results=3))
    assert search_res.total_results == 3
    assert all(r.score is not None for r in search_res.results)

    extract_res = await mock_tavily.extract(ExtractRequest(urls=["https://example.com"]))
    assert len(extract_res.results) == 1
    assert "Full article text" in (extract_res.results[0].extracted_text or "")


def test_provider_factory_mock_mode() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://agentflow:pass@localhost:5432/db",
        ai_mock_mode=True,
        gemini_api_key="real-gemini-key",
    )
    gemini = create_gemini_provider(settings)
    assert isinstance(gemini, MockGeminiProvider)

    groq = create_groq_provider(settings)
    assert isinstance(groq, MockGroqProvider)

    tavily = create_tavily_provider(settings)
    assert isinstance(tavily, MockTavilyProvider)


def test_provider_factory_live_mode_with_keys() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://agentflow:pass@localhost:5432/db",
        ai_mock_mode=False,
        gemini_api_key="key1",
        groq_api_key="key2",
        tavily_api_key="key3",
        groq_model="custom-free-model",
        free_model_allowlist="custom-free-model",
    )
    gemini = create_gemini_provider(settings)
    assert isinstance(gemini, GeminiProvider)

    groq = create_groq_provider(settings)
    assert isinstance(groq, GroqProvider)

    tavily = create_tavily_provider(settings)
    assert isinstance(tavily, TavilyProvider)
