"""Centralized cache management."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.caches import BaseCache, InMemoryCache

from agents.utils.structured_safe_sqlite_cache import StructuredSafeSQLiteCache

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages model caches."""

    _caches: dict[str, BaseCache] = {}

    @classmethod
    def get_cache(cls, cache_dir: Path | None, model_id: str) -> BaseCache:
        """
        Get or create cache for model.

        Args:
            cache_dir: Base directory for cache storage. If None, returns in-memory cache.
            model_id: Model identifier for cache key.

        Returns:
            BaseCache instance (StructuredSafeSQLiteCache or InMemoryCache).
        """
        # Sanitize model_id for filesystem
        safe_model_id = model_id.replace("/", "_").replace(":", "_")

        # Create cache key
        if cache_dir:
            cache_key = f"{cache_dir}|{safe_model_id}"
        else:
            cache_key = f"memory|{safe_model_id}"

        # Return existing cache if available
        if cache_key in cls._caches:
            logger.debug(f"Reusing cache for {safe_model_id}")
            return cls._caches[cache_key]

        # Create new cache
        if cache_dir:
            model_cache_dir = cache_dir / f"litellm_cache_{safe_model_id}"
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_db_path = model_cache_dir / "text_cache.db"
            cache_instance = StructuredSafeSQLiteCache(database_path=str(cache_db_path))
            logger.info(
                f"LLM cache configured at {model_cache_dir} (database: {cache_db_path})"
            )
        else:
            cache_instance = InMemoryCache()
            logger.info(f"LLM in-memory cache configured for {safe_model_id}")

        cls._caches[cache_key] = cache_instance
        return cache_instance

    @classmethod
    def clear(cls) -> None:
        """Clear all caches (useful for testing)."""
        cls._caches.clear()
        logger.debug("Cleared all model caches")
