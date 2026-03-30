"""Tests for litellm disk cache configuration in ModelBuilder."""

import tempfile
from pathlib import Path

import pytest

from OpenDsStar.agents.utils.model_builder import ModelBuilder


class TestLiteLLMCache:
    """Test litellm disk cache configuration."""

    def setup_method(self):
        """Reset cache state before each test."""
        from OpenDsStar.agents.utils.cache_manager import CacheManager

        # Reset global state
        CacheManager.clear()

    def test_configure_cache_creates_directory(self):
        """Test that configure_cache creates the cache directory."""
        from OpenDsStar.agents.utils.cache_manager import CacheManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            model_name = "test-model"

            _ = CacheManager.get_cache(cache_dir, model_name)

            # Check that model-specific subdirectory was created
            model_cache_dir = cache_dir / "litellm_cache_test-model"
            assert model_cache_dir.exists()
            assert model_cache_dir.is_dir()

            # Check that cache database was created
            cache_db = model_cache_dir / "text_cache.db"
            assert cache_db.exists()

    def test_configure_cache_returns_text_only_cache(self):
        """Test that configure_cache returns StructuredSafeSQLiteCache instance."""
        from OpenDsStar.agents.utils.cache_manager import CacheManager
        from OpenDsStar.agents.utils.structured_safe_sqlite_cache import (
            StructuredSafeSQLiteCache,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            model_name = "test-model"

            cache = CacheManager.get_cache(cache_dir, model_name)

            assert cache is not None
            assert isinstance(cache, StructuredSafeSQLiteCache)

    def test_configure_cache_reuses_same_model_cache(self):
        """Test that configure_cache reuses cache for same model+cache_dir."""
        from OpenDsStar.agents.utils.cache_manager import CacheManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            model_name = "test-model"

            # First configuration
            first_cache = CacheManager.get_cache(cache_dir, model_name)

            # Second configuration with same cache_dir and model should reuse
            second_cache = CacheManager.get_cache(cache_dir, model_name)

            # Should be the same cache instance (reused)
            assert first_cache is second_cache

    def test_configure_cache_different_models(self):
        """Test that different models get different caches."""
        from OpenDsStar.agents.utils.cache_manager import CacheManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"

            # Configure cache for first model
            cache1 = CacheManager.get_cache(cache_dir, "model1")

            # Configure cache for second model
            cache2 = CacheManager.get_cache(cache_dir, "model2")

            # Should be different cache instances (different models)
            assert cache1 is not cache2

            # Verify both cache directories exist
            assert (cache_dir / "litellm_cache_model1").exists()
            assert (cache_dir / "litellm_cache_model2").exists()

            # Verify both cache databases exist
            assert (cache_dir / "litellm_cache_model1" / "text_cache.db").exists()
            assert (cache_dir / "litellm_cache_model2" / "text_cache.db").exists()

    def test_build_with_cache_dir_configures_cache(self):
        """Test that build() with cache_dir configures the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"

            # Build model with cache_dir
            model, model_id = ModelBuilder.build(
                model="watsonx/mistralai/mistral-medium-2505",
                temperature=0.0,
                cache_dir=cache_dir,
            )

            # Cache directory and database should be created
            assert cache_dir.exists()
            model_cache_dir = (
                cache_dir / "litellm_cache_watsonx_mistralai_mistral-medium-2505"
            )
            assert model_cache_dir.exists()
            cache_db = model_cache_dir / "text_cache.db"
            assert cache_db.exists()

            # Model should have cache attribute
            assert hasattr(model, "cache")
            assert model.cache is not None

    def test_build_without_cache_dir_uses_memory_cache(self):
        """Test that build() without cache_dir uses in-memory cache."""
        from langchain_core.caches import InMemoryCache

        # Build model without cache_dir - should use in-memory cache
        model, model_id = ModelBuilder.build(
            model="watsonx/mistralai/mistral-medium-2505",
            temperature=0.0,
            cache_dir=None,
        )

        # Model should have cache attribute
        assert hasattr(model, "cache")
        assert model.cache is not None
        assert isinstance(model.cache, InMemoryCache)

    def test_cache_dir_must_be_explicit(self):
        """Test that cache_dir can be None (uses in-memory cache) or explicit Path."""
        from langchain_core.caches import InMemoryCache

        # Build without cache_dir - should use in-memory cache
        model, model_id = ModelBuilder.build(
            model="watsonx/mistralai/mistral-medium-2505",
            temperature=0.0,
        )

        # Should use in-memory cache when cache_dir is not provided
        assert hasattr(model, "cache")
        assert isinstance(model.cache, InMemoryCache)

    @pytest.mark.e2e
    def test_cache_actually_caches_llm_responses(self):
        """Test that the cache actually caches LLM responses (requires API keys)."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"

            # Build model with cache
            model, model_id = ModelBuilder.build(
                model="watsonx/mistralai/mistral-medium-2505",
                temperature=0.0,
                cache_dir=cache_dir,
            )

            # Make first call (should hit API) — retry on transient errors
            prompt = "What is 2+2? Answer with just the number."
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    start1 = time.time()
                    response1 = model.invoke(prompt)
                    time1 = time.time() - start1
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(10 * (attempt + 1))
                        continue
                    raise

            # Make second identical call (should hit cache)
            start2 = time.time()
            response2 = model.invoke(prompt)
            time2 = time.time() - start2

            # Verify responses are the same
            assert response1.content == response2.content

            # Second call should be significantly faster (cached)
            # Use 5x threshold to avoid flakiness from fast API responses
            assert (
                time2 < time1 / 5
            ), f"Cache hit ({time2:.3f}s) should be much faster than API call ({time1:.3f}s)"

            # Verify cache database was created
            cache_db_files = list(cache_dir.rglob("*.db"))
            assert len(cache_db_files) > 0, "Cache database should be created"
