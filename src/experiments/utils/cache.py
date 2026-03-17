"""Cache utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class NullCache:
    """No-op cache implementation."""

    def get(self, key: str) -> Any | None:
        return None

    def put(self, key: str, value: Any) -> None:
        return


class FileCache:
    """
    File-based cache implementation using diskcache.

    Provides persistent caching of Python objects to disk with automatic
    serialization/deserialization. Uses diskcache for efficient storage.

    Args:
        cache_dir: Directory to store cache files
        size_limit: Maximum cache size in bytes (default: 1GB)
    """

    def __init__(self, cache_dir: Path, size_limit: int = 1024**3) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize diskcache with size limit
        # Type checker: diskcache is guaranteed to be available here due to check above
        import diskcache as dc  # Import locally to satisfy type checker

        self._cache = dc.Cache(
            str(self.cache_dir),
            size_limit=size_limit,
        )

    def get(self, key: str) -> Any | None:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        return self._cache.get(key)

    def put(self, key: str, value: Any) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be picklable)
        """
        self._cache.set(key, value)

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()

    def close(self) -> None:
        """Close the cache (cleanup resources)."""
        self._cache.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
