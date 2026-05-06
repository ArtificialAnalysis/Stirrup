"""Tests for AsyncTokenBucket."""

from __future__ import annotations

import asyncio
import time

import pytest

from stirrup.utils.throttle import AsyncTokenBucket


async def test_single_acquire_no_wait() -> None:
    """A fresh bucket has full capacity; one acquire returns immediately."""
    bucket = AsyncTokenBucket(rate_per_sec=10.0)
    start = time.monotonic()
    async with bucket:
        pass
    assert time.monotonic() - start < 0.05


async def test_burst_within_capacity() -> None:
    """Up to ``capacity`` concurrent acquires fit inside the initial burst."""
    bucket = AsyncTokenBucket(rate_per_sec=10.0)

    async def acquire() -> None:
        async with bucket:
            pass

    start = time.monotonic()
    await asyncio.gather(*(acquire() for _ in range(10)))
    elapsed = time.monotonic() - start
    assert elapsed < 0.2, f"burst should complete fast, took {elapsed:.3f}s"


async def test_throttles_beyond_capacity() -> None:
    """Acquires beyond initial capacity wait at ``rate_per_sec``."""
    rate = 10.0
    bucket = AsyncTokenBucket(rate_per_sec=rate)

    async def acquire() -> None:
        async with bucket:
            pass

    # 20 acquires with capacity=10, rate=10/s. First 10 burst; remaining 10
    # spread out at 10/s, so total ~1.0s. Allow generous bounds for CI jitter.
    start = time.monotonic()
    await asyncio.gather(*(acquire() for _ in range(20)))
    elapsed = time.monotonic() - start
    assert 0.85 < elapsed < 2.0, f"expected ~1.0s throttle, got {elapsed:.3f}s"


async def test_explicit_capacity_separate_from_rate() -> None:
    """``capacity`` controls burst size independently of ``rate_per_sec``."""
    bucket = AsyncTokenBucket(rate_per_sec=10.0, capacity=2.0)

    async def acquire() -> None:
        async with bucket:
            pass

    # capacity=2, rate=10/s. 4 acquires: 2 burst, then ~0.2s for the rest.
    start = time.monotonic()
    await asyncio.gather(*(acquire() for _ in range(4)))
    elapsed = time.monotonic() - start
    assert 0.15 < elapsed < 0.6, f"expected ~0.2s, got {elapsed:.3f}s"


def test_invalid_rate_raises() -> None:
    """Non-positive rate is rejected at construction time."""
    with pytest.raises(ValueError, match="rate_per_sec must be positive"):
        AsyncTokenBucket(rate_per_sec=0)
    with pytest.raises(ValueError, match="rate_per_sec must be positive"):
        AsyncTokenBucket(rate_per_sec=-1.0)


def test_invalid_capacity_raises() -> None:
    """Non-positive capacity is rejected at construction time."""
    with pytest.raises(ValueError, match="capacity must be positive"):
        AsyncTokenBucket(rate_per_sec=1.0, capacity=0)
    with pytest.raises(ValueError, match="capacity must be positive"):
        AsyncTokenBucket(rate_per_sec=1.0, capacity=-5.0)
