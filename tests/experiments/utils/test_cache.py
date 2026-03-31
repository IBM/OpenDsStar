"""Tests for cache utilities."""

from OpenDsStar.experiments.utils.cache import FileCache, NullCache


class TestNullCache:
    """Test NullCache implementation."""

    def test_null_cache_behavior(self):
        """Test that NullCache always returns None (no-op cache)."""
        cache = NullCache()

        # Put should do nothing
        cache.put("key1", "value1")
        cache.put("key2", {"complex": "data"})

        # Get should always return None
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("nonexistent") is None


class TestFileCache:
    """Test FileCache implementation."""

    def test_initialization_creates_directory(self, tmp_path):
        """Test that initialization creates cache directory if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        cache = FileCache(cache_dir)
        assert cache_dir.exists()
        cache.close()

    def test_overwrite_existing_key(self, temp_cache_dir):
        """Test that putting same key overwrites previous value."""
        cache = FileCache(temp_cache_dir)
        cache.put("key", "value1")
        assert cache.get("key") == "value1"

        cache.put("key", "value2")
        assert cache.get("key") == "value2"
        cache.close()

    def test_clear_removes_all_entries(self, temp_cache_dir):
        """Test that clear removes all cached entries."""
        cache = FileCache(temp_cache_dir)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", {"nested": "data"})

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None
        cache.close()

    def test_persistence_across_instances(self, temp_cache_dir):
        """Test that cache persists data across different instances (file-based)."""
        # First instance writes data
        cache1 = FileCache(temp_cache_dir)
        cache1.put("persistent_key", "persistent_value")
        cache1.put("complex_data", {"nested": {"data": [1, 2, 3]}})
        cache1.close()

        # Second instance should read the same data
        cache2 = FileCache(temp_cache_dir)
        assert cache2.get("persistent_key") == "persistent_value"
        assert cache2.get("complex_data") == {"nested": {"data": [1, 2, 3]}}
        cache2.close()

    def test_special_characters_in_key(self, temp_cache_dir):
        """Test that keys with special characters are handled correctly."""
        cache = FileCache(temp_cache_dir)
        special_keys = [
            "key:with:colons",
            "key/with/slashes",
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
        ]

        for key in special_keys:
            cache.put(key, f"value_for_{key}")

        for key in special_keys:
            assert cache.get(key) == f"value_for_{key}"

        cache.close()

    def test_none_value_ambiguity(self, temp_cache_dir):
        """Test that None value is indistinguishable from missing key (edge case)."""
        cache = FileCache(temp_cache_dir)

        # Store None as value
        cache.put("none_key", None)

        # Both None value and missing key return None
        # This is expected behavior but important to document
        assert cache.get("none_key") is None
        assert cache.get("missing_key") is None

        cache.close()

    def test_context_manager_cleanup(self, temp_cache_dir):
        """Test that context manager properly closes cache."""
        with FileCache(temp_cache_dir) as cache:
            cache.put("key", "value")
            assert cache.get("key") == "value"

        # Cache should be closed after context
        # Verify by opening new instance and checking persistence
        cache2 = FileCache(temp_cache_dir)
        assert cache2.get("key") == "value"
        cache2.close()

    def test_context_manager_handles_exceptions(self, temp_cache_dir):
        """Test that context manager closes cache even when exception occurs."""
        try:
            with FileCache(temp_cache_dir) as cache:
                cache.put("key", "value")
                raise ValueError("Test error")
        except ValueError:
            pass

        # Cache should still be properly closed and data persisted
        cache2 = FileCache(temp_cache_dir)
        assert cache2.get("key") == "value"
        cache2.close()
