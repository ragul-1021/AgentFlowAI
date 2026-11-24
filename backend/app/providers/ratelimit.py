"""Concurrency semaphores for rate-limiting outbound provider requests."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class ConcurrencyLimiter:
    """Manages asyncio.Semaphore per provider to enforce concurrency bounds."""

    def __init__(self, limits: dict[str, int]) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {
            provider: asyncio.Semaphore(max_concurrency)
            for provider, max_concurrency in limits.items()
        }
        self._fallback_semaphore = asyncio.Semaphore(10)

    @asynccontextmanager
    async def acquire(self, provider: str) -> AsyncIterator[None]:
        semaphore = self._semaphores.get(provider, self._fallback_semaphore)
        async with semaphore:
            yield
