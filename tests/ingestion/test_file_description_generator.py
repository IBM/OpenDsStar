"""Tests for FileDescriptionGenerator._normalize_llm_output method."""

from unittest.mock import MagicMock

import pytest

from OpenDsStar.ingestion.docling_based_ingestion.file_description_generator import (
    FileDescriptionGenerator,
)


@pytest.fixture
def mock_generator():
    """Create a FileDescriptionGenerator with mocked dependencies."""
    mock_llm = MagicMock()
    mock_cache = MagicMock()
    return FileDescriptionGenerator(
        llm=mock_llm,
        description_cache=mock_cache,
        model="test-model",
    )


class TestNormalizeLLMOutput:
    """Tests for _normalize_llm_output method."""

    def test_normalize_with_file_path(self, mock_generator):
        """Test that file_path is used instead of doc_id when provided."""
        result = mock_generator._normalize_llm_output(
            "Test content about the file",
            doc_id="018_Staff::all.parquet::1c7093c5e20f64ab",
            display_name="018_Staff::all.parquet",
            file_path="018_Staff/all.parquet",
        )

        # Should contain the relative path
        assert "018_Staff/all.parquet" in result
        # Should NOT contain the hash from doc_id
        assert "1c7093c5e20f64ab" not in result
        # Should contain the display name
        assert "018_Staff::all.parquet" in result
        # Should contain the content
        assert "Test content about the file" in result

    def test_normalize_without_file_path_falls_back_to_doc_id(self, mock_generator):
        """Test that doc_id is used when file_path is not provided."""
        result = mock_generator._normalize_llm_output(
            "Test content",
            doc_id="018_Staff::all.parquet::1c7093c5e20f64ab",
            display_name="018_Staff::all.parquet",
            file_path="",  # Empty file_path
        )

        # Should fall back to doc_id (with hash)
        assert "018_Staff::all.parquet::1c7093c5e20f64ab" in result
        assert "Test content" in result

    def test_normalize_with_markdown_fences(self, mock_generator):
        """Test that markdown code fences are stripped."""
        result = mock_generator._normalize_llm_output(
            "```\nTest content\n```",
            doc_id="test::123",
            display_name="test.csv",
            file_path="data/test.csv",
        )

        # Fences should be stripped
        assert "```" not in result
        assert "Test content" in result
        assert "data/test.csv" in result

    def test_normalize_with_object_having_content_attribute(self, mock_generator):
        """Test normalization with an object that has a content attribute."""
        mock_output = MagicMock()
        mock_output.content = "Content from object"

        result = mock_generator._normalize_llm_output(
            mock_output,
            doc_id="test::456",
            display_name="test.parquet",
            file_path="dataset/test.parquet",
        )

        assert "Content from object" in result
        assert "dataset/test.parquet" in result

    def test_normalize_without_doc_id_and_display_name(self, mock_generator):
        """Test that no prefix is added when doc_id or display_name is missing."""
        result = mock_generator._normalize_llm_output(
            "Just content",
            doc_id="",
            display_name="",
            file_path="some/path.csv",
        )

        # Should not have the File Name/File Path headers
        assert "## File Name" not in result
        assert "## File Path" not in result
        # Should just have the content
        assert result == "Just content"

    def test_normalize_preserves_content_structure(self, mock_generator):
        """Test that the output structure is correct."""
        result = mock_generator._normalize_llm_output(
            "Description of the file",
            doc_id="test::789",
            display_name="test.csv",
            file_path="data/test.csv",
        )

        lines = result.split("\n")
        # Check structure
        assert lines[0] == "## File Name"
        assert lines[1] == "test.csv"
        assert lines[2] == ""
        assert lines[3] == "## File Path"
        assert lines[4] == "data/test.csv"
        assert lines[5] == ""
        assert "Description of the file" in "\n".join(lines[6:])

    def test_normalize_with_complex_file_path(self, mock_generator):
        """Test with a more complex file path structure."""
        result = mock_generator._normalize_llm_output(
            "Content",
            doc_id="complex::abc123",
            display_name="complex_file.parquet",
            file_path="datasets/2024/Q1/complex_file.parquet",
        )

        assert "datasets/2024/Q1/complex_file.parquet" in result
        assert "abc123" not in result
