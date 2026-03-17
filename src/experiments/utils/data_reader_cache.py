"""Cache utilities for data readers."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from .cache import FileCache

logger = logging.getLogger(__name__)


class DataReaderCache:
    """
    Cache for data reader results (corpus and benchmark data).

    Provides persistent caching of loaded data to avoid re-loading from source
    on subsequent runs. Each data reader gets a unique cache based on its
    configuration parameters.

    The cache is stored in a directory structure:
    <cache_base_dir>/data_reader_cache_<cache_key>/

    This aligns with the structure used by AgentCache and EvaluationCache.

    Args:
        cache_base_dir: Base directory for cache storage (e.g., benchmark cache dir)
        enabled: Whether caching is enabled (default: True)
    """

    def __init__(
        self,
        cache_base_dir: Path | str | None = None,
        enabled: bool = True,
    ):
        self.cache_base_dir = Path(cache_base_dir) if cache_base_dir else None
        self.enabled = enabled

        if self.enabled and self.cache_base_dir:
            self.cache_base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Data reader cache initialized at: {self.cache_base_dir}")

    def _generate_cache_key(
        self,
        reader_name: str,
        dataset_name: str,
        split: str,
        question_limit: int | None,
        document_factor: int | None,
        seed: int | None,
    ) -> str:
        """
        Generate a unique cache key based on data reader configuration.

        Args:
            reader_name: Name of the data reader class
            dataset_name: Name of the dataset
            split: Dataset split (train/test)
            question_limit: Question limit parameter
            document_factor: Document factor parameter
            seed: Random seed

        Returns:
            Unique cache key string
        """
        # Create a deterministic string from all parameters
        params_str = (
            f"{reader_name}_{dataset_name}_{split}_"
            f"q{question_limit}_d{document_factor}_s{seed}"
        )

        # Hash to create a shorter, filesystem-safe key
        hash_obj = hashlib.md5(params_str.encode())
        hash_str = hash_obj.hexdigest()[:16]

        # Create readable cache key
        cache_key = f"{dataset_name}_{split}_{hash_str}"
        return cache_key

    def get_cache_path(self, cache_key: str) -> Path:
        """
        Get the cache directory path for a specific cache key.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache directory
        """
        if self.cache_base_dir is None:
            raise ValueError("Cache base directory not set")
        return self.cache_base_dir / f"data_reader_cache_{cache_key}"

    def get(
        self,
        reader_name: str,
        dataset_name: str,
        split: str,
        question_limit: int | None = None,
        document_factor: int | None = None,
        seed: int | None = None,
    ) -> tuple[Any, Any] | None:
        """
        Retrieve cached corpus and benchmark data.

        Args:
            reader_name: Name of the data reader class
            dataset_name: Name of the dataset
            split: Dataset split (train/test)
            question_limit: Question limit parameter
            document_factor: Document factor parameter
            seed: Random seed

        Returns:
            Tuple of (corpus, benchmark_data) if cached, None otherwise
        """
        if not self.enabled:
            return None

        cache_key = self._generate_cache_key(
            reader_name, dataset_name, split, question_limit, document_factor, seed
        )

        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss for {cache_key}")
            return None

        try:
            # Use FileCache to load the cached data
            with FileCache(cache_path) as cache:
                corpus = cache.get("corpus")
                benchmark_data = cache.get("benchmark_data")

                if corpus is None or benchmark_data is None:
                    logger.warning(f"Incomplete cache for {cache_key}")
                    return None

                logger.info(f"Cache hit for {cache_key} at {cache_path}")
                return corpus, benchmark_data

        except Exception as e:
            logger.warning(f"Failed to load cache for {cache_key}: {e}")
            return None

    def put(
        self,
        corpus: Any,
        benchmark_data: Any,
        reader_name: str,
        dataset_name: str,
        split: str,
        question_limit: int | None = None,
        document_factor: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Store corpus and benchmark data in cache.

        Args:
            corpus: Corpus data to cache
            benchmark_data: Benchmark data to cache
            reader_name: Name of the data reader class
            dataset_name: Name of the dataset
            split: Dataset split (train/test)
            question_limit: Question limit parameter
            document_factor: Document factor parameter
            seed: Random seed
        """
        if not self.enabled:
            return

        cache_key = self._generate_cache_key(
            reader_name, dataset_name, split, question_limit, document_factor, seed
        )

        cache_path = self.get_cache_path(cache_key)

        try:
            # Use FileCache to store the data
            with FileCache(cache_path) as cache:
                cache.put("corpus", corpus)
                cache.put("benchmark_data", benchmark_data)

            logger.info(f"Cached data for {cache_key} at {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to cache data for {cache_key}: {e}")

    def clear(self, cache_key: str | None = None) -> None:
        """
        Clear cached data.

        Args:
            cache_key: Specific cache key to clear, or None to clear all
        """
        if not self.enabled or self.cache_base_dir is None:
            return

        if cache_key:
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists():
                import shutil

                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache for {cache_key}")
        else:
            # Clear all data_reader_cache_* directories
            if self.cache_base_dir.exists():
                import shutil

                for cache_path in self.cache_base_dir.glob("data_reader_cache_*"):
                    if cache_path.is_dir():
                        shutil.rmtree(cache_path)
                logger.info("Cleared all data reader caches")
