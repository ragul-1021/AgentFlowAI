"""Asynchronous Google Gemini integration using REST API and httpx."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from app.core.tokens import TokenUsage, calculate_cost
from app.providers.base import BaseLLMProvider, LLMRequest, LLMResponse
from app.providers.exceptions import (
    PaidModelBlockedError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StructuredOutputError,
)
from app.providers.guardrails import validate_model_allowed, validate_token_limit
from app.providers.ratelimit import ConcurrencyLimiter
from app.providers.retry import execute_with_retry

logger = logging.getLogger(__name__)

T = TypeVar("T")

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(BaseLLMProvider):
    """Asynchronous Google Gemini provider implementation."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-2.5-flash",
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        limiter: ConcurrencyLimiter | None = None,
        free_only: bool = True,
        allow_paid_models: bool = False,
        custom_allowlist: str = "",
        max_request_tokens: int = 8192,
    ) -> None:
        super().__init__(provider_name="gemini", default_model=default_model)
        self._api_key = api_key.strip() if api_key else ""
        self._client = client
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._limiter = limiter
        self._free_only = free_only
        self._allow_paid_models = allow_paid_models
        self._custom_allowlist = custom_allowlist
        self._max_request_tokens = max_request_tokens

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise ProviderAuthError(
                "Gemini API key is not configured.", provider=self.provider_name
            )

    def _format_contents(self, request: LLMRequest) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        contents: list[dict[str, Any]] = []
        system_instruction: dict[str, Any] | None = None

        if request.system_prompt:
            system_instruction = {"parts": [{"text": request.system_prompt}]}

        for msg in request.messages:
            if msg.role == "system":
                # If explicit system message present, set as system_instruction
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        return contents, system_instruction

    def _build_generation_config(
        self, request: LLMRequest, response_schema: type[BaseModel] | dict[str, Any] | None = None
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": request.temperature,
        }
        if request.max_tokens:
            config["maxOutputTokens"] = request.max_tokens

        if response_schema is not None:
            config["responseMimeType"] = "application/json"
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                config["responseSchema"] = response_schema.model_json_schema()
            elif isinstance(response_schema, dict):
                config["responseSchema"] = response_schema

        return config

    def _handle_response_status(self, response: httpx.Response, model: str) -> None:
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
            message = err_json.get("error", {}).get("message", response.text)
        except Exception:
            message = response.text or f"HTTP {status}"

        # Clean any potential secrets from the message
        clean_msg = f"Gemini API returned error: {message[:300]}"

        if status in (401, 403):
            raise ProviderAuthError(clean_msg, provider=self.provider_name, model=model, status_code=status)
        if status == 429:
            raise ProviderRateLimitError(
                clean_msg,
                provider=self.provider_name,
                model=model,
                status_code=status,
                retry_after=retry_after,
            )
        if status == 400:
            raise ProviderBadRequestError(clean_msg, provider=self.provider_name, model=model, status_code=status)
        if status >= 500:
            raise ProviderUnavailableError(clean_msg, provider=self.provider_name, model=model, status_code=status)

        raise ProviderError(clean_msg, provider=self.provider_name, model=model, status_code=status)

    async def _execute_http(self, payload: dict[str, Any], model: str) -> tuple[dict[str, Any], int]:
        self._ensure_api_key()
        url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

        async def _call() -> tuple[dict[str, Any], int]:
            started = time.perf_counter()
            client = self._client or httpx.AsyncClient(timeout=self._timeout)
            try:
                if self._limiter:
                    async with self._limiter.acquire(self.provider_name):
                        resp = await client.post(url, json=payload, headers=headers, timeout=self._timeout)
                else:
                    resp = await client.post(url, json=payload, headers=headers, timeout=self._timeout)

                self._handle_response_status(resp, model)
                latency = max(1, int((time.perf_counter() - started) * 1000))
                return resp.json(), latency
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(
                    f"Gemini request timed out after {self._timeout}s",
                    provider=self.provider_name,
                    model=model,
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

    def _parse_llm_response(self, data: dict[str, Any], latency_ms: int, model: str) -> LLMResponse:
        candidates = data.get("candidates", [])
        if not candidates:
            content = ""
        else:
            first = candidates[0]
            parts = first.get("content", {}).get("parts", [])
            content = parts[0].get("text", "") if parts else ""

        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
        total_tokens = usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens)

        cost = calculate_cost(model, prompt_tokens, completion_tokens)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_estimate=cost,
        )

        return LLMResponse(
            content=content,
            usage=usage,
            latency_ms=latency_ms,
            provider=self.provider_name,
            model=model,
            raw_metadata={"finish_reason": candidates[0].get("finishReason") if candidates else None},
        )

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
        contents, system_instruction = self._format_contents(request)
        gen_config = self._build_generation_config(request, request.response_schema)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_config,
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction

        data, latency = await self._execute_http(payload, target_model)
        return self._parse_llm_response(data, latency, target_model)

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
        contents, system_instruction = self._format_contents(request)
        gen_config = self._build_generation_config(request, response_schema)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_config,
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction

        data, latency = await self._execute_http(payload, target_model)
        response = self._parse_llm_response(data, latency, target_model)

        try:
            parsed_json = json.loads(response.content)
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                structured = response_schema.model_validate(parsed_json)
            else:
                structured = parsed_json
            response.structured_data = structured
            return structured, response
        except (json.JSONDecodeError, ValueError) as exc:
            raise StructuredOutputError(
                f"Failed to parse structured output from Gemini response: {exc}",
                provider=self.provider_name,
                model=target_model,
                raw_error=response.content,
            ) from exc
