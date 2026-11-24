"""Asynchronous Groq integration using OpenAI-compatible REST API and httpx."""

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

GROQ_API_BASE = "https://api.groq.com/openai/v1"


class GroqProvider(BaseLLMProvider):
    """Asynchronous Groq provider implementation."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "",
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
        super().__init__(provider_name="groq", default_model=default_model)
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
                "Groq API key is not configured.", provider=self.provider_name
            )

    def _format_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

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

        clean_msg = f"Groq API returned error: {message[:300]}"

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
        url = f"{GROQ_API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
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
                    f"Groq request timed out after {self._timeout}s",
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
        choices = data.get("choices", [])
        if not choices:
            content = ""
        else:
            first = choices[0]
            content = first.get("message", {}).get("content", "")

        usage_dict = data.get("usage", {})
        prompt_tokens = usage_dict.get("prompt_tokens", 0)
        completion_tokens = usage_dict.get("completion_tokens", 0)
        total_tokens = usage_dict.get("total_tokens", prompt_tokens + completion_tokens)

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
            raw_metadata={"finish_reason": choices[0].get("finish_reason") if choices else None},
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
        messages = self._format_messages(request)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

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
        messages = self._format_messages(request)

        # Enforce json instruction in prompt if needed
        schema_json = (
            response_schema.model_json_schema()
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel)
            else response_schema
        )
        instruction = f"\nReturn ONLY valid JSON conforming to this schema:\n{json.dumps(schema_json)}"
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += instruction
        else:
            messages.append({"role": "user", "content": instruction})

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": request.temperature,
            "response_format": {"type": "json_object"},
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

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
                f"Failed to parse structured output from Groq response: {exc}",
                provider=self.provider_name,
                model=target_model,
                raw_error=response.content,
            ) from exc
