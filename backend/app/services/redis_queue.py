"""Async Redis-backed task queue with fallback."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import redis.asyncio as redis

from app.core.config import Settings

logger = logging.getLogger(__name__)


class RedisQueue:
    """Simple async queue backed by Redis lists."""

    def __init__(self, settings: Settings, client: Optional[redis.Redis]) -> None:
        self._settings = settings
        self._client = client
        self._fallback_queue: Optional[asyncio.Queue] = None

    @classmethod
    async def create(cls, settings: Settings) -> "RedisQueue":
        """Create RedisQueue with connectivity check."""
        client: Optional[redis.Redis] = None
        try:
            client = redis.from_url(settings.redis_url, decode_responses=True)
            await client.ping()
            logger.info("Connected to Redis at %s", settings.redis_url)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Redis unavailable, using in-memory queue: %s", exc)
            client = None
        queue = cls(settings=settings, client=client)
        if client is None:
            queue._fallback_queue = asyncio.Queue()
        return queue

    async def enqueue(self, payload: Any) -> None:
        """Enqueue a payload for async processing."""
        data = json.dumps(payload)
        if self._client is not None:
            await self._client.rpush(self._settings.redis_queue_name, data)
        elif self._fallback_queue is not None:
            await self._fallback_queue.put(data)

    async def dequeue(self, timeout_seconds: int = 0) -> Optional[Any]:
        """Dequeue a payload, optionally blocking."""
        if self._client is not None:
            result = await self._client.blpop(
                self._settings.redis_queue_name,
                timeout=timeout_seconds,
            )
            if result is None:
                return None
            _, payload = result
            return json.loads(payload)
        if self._fallback_queue is not None:
            try:
                payload = await asyncio.wait_for(self._fallback_queue.get(), timeout=timeout_seconds)
                return json.loads(payload)
            except asyncio.TimeoutError:
                return None
        return None

    async def close(self) -> None:
        """Close Redis connection if present."""
        if self._client is not None:
            await self._client.close()
