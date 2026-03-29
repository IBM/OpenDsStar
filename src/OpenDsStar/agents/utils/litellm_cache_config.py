"""
LiteLLM Cache Configuration Utility.

Provides shared cache configuration for smolagents that use LiteLLM.
"""

from __future__ import annotations

import logging
from pathlib import Path

import litellm

logger = logging.getLogger(__name__)


def configure_litellm_cache(cache_dir: Path) -> None:
    """
    Configure LiteLLM's disk cache globally for all agents.

    Sets up a single shared cache for all LiteLLM models. Only configures
    the cache if it hasn't been set yet (litellm.cache is None).

    Args:
        cache_dir: Base directory for cache storage. The actual cache will be
                   stored in {cache_dir}/litellm_cache_smolagents/

    Note:
        - LiteLLM cache is configured globally via litellm.cache
        - Only sets cache if litellm.cache is None (doesn't override existing cache)
        - Cache uses SQLite database (cache.db) for storage
        - Single cache shared by all models and agents
        - Cache keys automatically include model name, so different models
          have separate cache entries (no collision between models)
    """
    # Only set cache if it doesn't exist or is None
    if litellm.cache is None:
        from litellm.caching.caching import Cache
        from litellm.types.caching import LiteLLMCacheType

        # Use litellm_cache_smolagents subdirectory for smolagents cache
        inference_cache_dir = cache_dir / "litellm_cache_smolagents"
        inference_cache_dir.mkdir(parents=True, exist_ok=True)

        litellm.cache = Cache(
            type=LiteLLMCacheType.DISK,
            disk_cache_dir=str(inference_cache_dir),
        )

        logger.info(f"Enabled LiteLLM disk cache at: {inference_cache_dir}")
    else:
        logger.debug("LiteLLM cache already configured, skipping")
