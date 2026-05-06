"""Async throttling primitives."""

from __future__ import annotations

import asyncio
import time
from contextlib import AbstractAsyncContextManager


class AsyncTokenBucket(AbstractAsyncContextManager["AsyncTokenBucket"]):
    """Process-local async token bucket.

    Acts as an async context manager: each ``async with`` consumes one token.
    Tokens refill at ``rate_per_sec``; capacity defaults to ``rate_per_sec`` so
    a fresh bucket can absorb a one-second burst before throttling kicks in.

    Safe to share across coroutines on a single event loop. Not safe across
    multiple event loops: the internal ``asyncio.Lock`` binds to the first
    loop that uses it.

    Example:
        gate = AsyncTokenBucket(rate_per_sec=5.0)
        async with gate:
            await some_rate_limited_call()
    """

    def __init__(self, rate_per_sec: float, capacity: float | None = None) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive")
        self._rate = rate_per_sec
        self._capacity = capacity if capacity is not None else rate_per_sec
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> AsyncTokenBucket:
        # Holding _lock across the await is intentional: it queues waiters
        # fairly so each one sleeps just long enough to claim the next slot,
        # instead of all waking together and racing for tokens.
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last_refill) * self._rate,
            )
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 1.0
                self._last_refill = time.monotonic()
            self._tokens -= 1.0
        return self

    async def __aexit__(self, *_: object) -> None:
        return None
