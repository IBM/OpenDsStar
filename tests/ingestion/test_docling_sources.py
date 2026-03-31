"""
Unit tests for docling_based_ingestion sources module.

Tests TempMaterializer and SourceFile functionality.
"""

from pathlib import Path

import pytest

from OpenDsStar.ingestion.docling_based_ingestion.sources import (
    SourceFile,
    TempMaterializer,
)


class TestSourceFile:
    """Test suite for SourceFile dataclass."""

    def test_init_basic(self):
        """Test basic SourceFile initialization."""
        source = SourceFile(display_name="test.txt", path_hint="/path/to/test.txt")
        assert source.display_name == "test.txt"
        assert source.path_hint == "/path/to/test.txt"
        assert source.stream_factory is None
        assert source.temp_path is None

    def test_init_with_stream_factory(self):
        """Test SourceFile with stream factory."""

        def factory():
            return b"test content"

        source = SourceFile(
            display_name="test.txt",
            path_hint="/path/to/test.txt",
            stream_factory=factory,
        )
        assert source.stream_factory is not None
        assert source.stream_factory() == b"test content"

    def test_init_with_temp_path(self):
        """Test SourceFile with temp path."""
        source = SourceFile(
            display_name="test.txt",
            path_hint="/path/to/test.txt",
            temp_path="/tmp/temp_file.txt",
        )
        assert source.temp_path == "/tmp/temp_file.txt"


class TestTempMaterializer:
    """Test suite for TempMaterializer class."""

    def test_init(self):
        """Test TempMaterializer initialization."""
        materializer = TempMaterializer()
        assert materializer._temp_files == []

    def test_materialize_creates_temp_file(self):
        """Test that materialize creates a temporary file."""
        materializer = TempMaterializer()
        content = b"Hello, World!"

        def factory():
            return content

        try:
            temp_path = materializer.materialize("test.txt", factory)

            # Verify file was created
            assert Path(temp_path).exists()

            # Verify content
            with open(temp_path, "rb") as f:
                assert f.read() == content

            # Verify it's tracked
            assert len(materializer._temp_files) == 1
            assert materializer._temp_files[0] == Path(temp_path)
        finally:
            materializer.cleanup()

    def test_materialize_preserves_suffix(self):
        """Test that materialize preserves file suffix."""
        materializer = TempMaterializer()

        def factory():
            return b"content"

        try:
            temp_path = materializer.materialize("test.json", factory)
            assert temp_path.endswith(".json")
        finally:
            materializer.cleanup()

    def test_materialize_uses_tmp_for_no_suffix(self):
        """Test that materialize uses .tmp when no suffix provided."""
        materializer = TempMaterializer()

        def factory():
            return b"content"

        try:
            temp_path = materializer.materialize("test", factory)
            assert temp_path.endswith(".tmp")
        finally:
            materializer.cleanup()

    def test_materialize_multiple_files(self):
        """Test materializing multiple files."""
        materializer = TempMaterializer()

        try:
            path1 = materializer.materialize("test1.txt", lambda: b"content1")
            path2 = materializer.materialize("test2.txt", lambda: b"content2")
            path3 = materializer.materialize("test3.txt", lambda: b"content3")

            # All files should exist
            assert Path(path1).exists()
            assert Path(path2).exists()
            assert Path(path3).exists()

            # All should be tracked
            assert len(materializer._temp_files) == 3

            # Verify content
            with open(path1, "rb") as f:
                assert f.read() == b"content1"
            with open(path2, "rb") as f:
                assert f.read() == b"content2"
            with open(path3, "rb") as f:
                assert f.read() == b"content3"
        finally:
            materializer.cleanup()

    def test_cleanup_removes_files(self):
        """Test that cleanup removes all temporary files."""
        materializer = TempMaterializer()

        path1 = materializer.materialize("test1.txt", lambda: b"content1")
        path2 = materializer.materialize("test2.txt", lambda: b"content2")

        # Files should exist before cleanup
        assert Path(path1).exists()
        assert Path(path2).exists()

        # Cleanup
        materializer.cleanup()

        # Files should be removed
        assert not Path(path1).exists()
        assert not Path(path2).exists()

        # Tracking list should be cleared
        assert materializer._temp_files == []

    def test_cleanup_handles_missing_files(self):
        """Test that cleanup handles already-deleted files gracefully."""
        materializer = TempMaterializer()

        path1 = materializer.materialize("test1.txt", lambda: b"content1")

        # Manually delete the file
        Path(path1).unlink()

        # Cleanup should not raise an exception
        materializer.cleanup()
        assert materializer._temp_files == []

    def test_cleanup_can_be_called_multiple_times(self):
        """Test that cleanup can be called multiple times safely."""
        materializer = TempMaterializer()

        materializer.materialize("test.txt", lambda: b"content")

        # Multiple cleanups should not raise
        materializer.cleanup()
        materializer.cleanup()
        materializer.cleanup()

        assert materializer._temp_files == []

    def test_materialize_with_large_content(self):
        """Test materializing large content."""
        materializer = TempMaterializer()

        # Create 1MB of content
        large_content = b"A" * (1024 * 1024)

        try:
            temp_path = materializer.materialize("large.bin", lambda: large_content)

            # Verify file size
            assert Path(temp_path).stat().st_size == len(large_content)

            # Verify content
            with open(temp_path, "rb") as f:
                assert f.read() == large_content
        finally:
            materializer.cleanup()

    def test_context_manager_pattern(self):
        """Test using TempMaterializer in a try-finally pattern."""
        materializer = TempMaterializer()
        temp_paths = []

        try:
            path1 = materializer.materialize("test1.txt", lambda: b"content1")
            path2 = materializer.materialize("test2.txt", lambda: b"content2")
            temp_paths = [path1, path2]

            # Files should exist
            assert Path(path1).exists()
            assert Path(path2).exists()
        finally:
            materializer.cleanup()

        # Files should be cleaned up
        for path in temp_paths:
            assert not Path(path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
