"""
Redis-backed audio cache for TTS responses.

Cache key: SHA-256(text + state + trend + voice)[:24]
Cache value: raw audio bytes (WAV or PCM depending on endpoint)

Caching is skipped for:
  - Text longer than settings.cache_max_text_chars (default 2000, covers
    all chunk sizes from both stream_and_play sentence-level and
    _synthesize_and_play's 1800-char merged chunks)

The cache is a best-effort layer — misses are fine, the TTS engine runs anyway.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class AudioCache:
    """Redis-backed audio cache (format-agnostic: WAV or PCM bytes)."""

    def __init__(self, redis_url: str, ttl_seconds: int, max_text_chars: int, enabled: bool) -> None:
        self._enabled = enabled and _REDIS_AVAILABLE
        self._ttl = ttl_seconds
        self._max_chars = max_text_chars
        self._client: Optional[object] = None

        if self._enabled:
            try:
                self._client = redis_lib.from_url(redis_url, decode_responses=False)
                self._client.ping()
                logger.info("AudioCache connected | redis=%s | ttl=%ds | max_chars=%d", redis_url, ttl_seconds, max_text_chars)
            except Exception as exc:
                logger.warning("AudioCache Redis connection failed | error=%s | caching disabled", exc)
                self._enabled = False

    def get(self, key: str) -> Optional[bytes]:
        if not self._enabled or self._client is None:
            return None
        try:
            data = self._client.get(key)
            if data:
                logger.debug("AudioCache HIT | key=%s | bytes=%d", key, len(data))
            return data
        except Exception as exc:
            logger.debug("AudioCache get error | key=%s | error=%s", key, exc)
            return None

    def set(self, key: str, audio_bytes: bytes, text: str) -> None:
        if not self._enabled or self._client is None:
            return
        if len(text) > self._max_chars:
            return
        try:
            self._client.setex(key, self._ttl, audio_bytes)
            logger.debug("AudioCache SET | key=%s | bytes=%d | ttl=%ds", key, len(audio_bytes), self._ttl)
        except Exception as exc:
            logger.debug("AudioCache set error | key=%s | error=%s", key, exc)

    def invalidate(self, key: str) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            self._client.delete(key)
        except Exception:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled