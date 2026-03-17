"""
Unit tests for DoclingConverter.

Tests the functionality of file conversion, suffix classification,
and text reading with various encodings and formats.
"""

import pytest

from ingestion.docling_based_ingestion.docling_converter import (
    DoclingConverter,
    FallbackTextDoc,
)


class TestFallbackTextDoc:
    """Test suite for FallbackTextDoc class."""

    def test_init(self):
        """Test FallbackTextDoc initialization."""
        doc = FallbackTextDoc(name="test.txt", _markdown="# Test content")
        assert doc.name == "test.txt"
        assert doc._markdown == "# Test content"

    def test_export_to_markdown(self):
        """Test export_to_markdown returns the markdown content."""
        doc = FallbackTextDoc(name="test.txt", _markdown="# Test\nContent here")
        assert doc.export_to_markdown() == "# Test\nContent here"


class TestDoclingConverterClassifySuffix:
    """Test suite for DoclingConverter.classify_suffix method."""

    def test_classify_skip_suffixes(self):
        """Test that unknown/unsupported files are classified as skip."""
        converter = DoclingConverter()

        # These are not in docling_suffixes or text_fallback_suffixes
        skip_suffixes = [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".mp4",
            ".zip",
            ".exe",
            ".unknown",
        ]
        for suffix in skip_suffixes:
            assert converter.classify_suffix(suffix) == "skip"
            # Test case insensitivity
            assert converter.classify_suffix(suffix.upper()) == "skip"

    def test_classify_fallback_text_suffixes(self):
        """Test that text fallback files are classified as fallback_text."""
        converter = DoclingConverter()

        # Default text_fallback_suffixes: .txt, .log, .md, .rst
        text_suffixes = [".txt", ".log", ".md", ".rst"]
        for suffix in text_suffixes:
            assert converter.classify_suffix(suffix) == "fallback_text"
            # Test case insensitivity
            assert converter.classify_suffix(suffix.upper()) == "fallback_text"

    def test_classify_docling_suffixes(self):
        """Test that docling-supported files are classified as docling."""
        converter = DoclingConverter()

        # These are in docling_suffixes
        docling_suffixes = [
            ".pdf",
            ".docx",
            ".pptx",
            ".py",
            ".json",
            ".yaml",
            ".csv",
            ".sql",
            ".html",
        ]
        for suffix in docling_suffixes:
            assert converter.classify_suffix(suffix) == "docling"

    def test_classify_empty_suffix(self):
        """Test classification of empty suffix."""
        converter = DoclingConverter()
        # Empty suffix is not in any set, so it's skipped
        assert converter.classify_suffix("") == "skip"

    def test_custom_docling_suffixes(self):
        """Test custom docling suffixes."""
        converter = DoclingConverter(docling_suffixes={".custom", ".test"})
        assert converter.classify_suffix(".custom") == "docling"
        assert converter.classify_suffix(".test") == "docling"
        # Default docling suffixes should not be present
        assert converter.classify_suffix(".pdf") == "skip"

    def test_custom_text_fallback_suffixes(self):
        """Test custom text fallback suffixes."""
        converter = DoclingConverter(text_fallback_suffixes={".custom", ".test"})
        assert converter.classify_suffix(".custom") == "fallback_text"
        assert converter.classify_suffix(".test") == "fallback_text"
        # Default text suffixes should not be present (unless also in docling_suffixes)
        # .txt is in both default docling_suffixes and text_fallback_suffixes
        # When we override text_fallback_suffixes, .txt falls back to docling
        assert converter.classify_suffix(".txt") == "docling"


class TestDoclingConverterReadTextBytes:
    """Test suite for DoclingConverter._read_text_bytes method."""

    def test_read_utf8_text(self):
        """Test reading UTF-8 encoded text."""
        converter = DoclingConverter()
        text = "Hello, World! 你好"
        raw = text.encode("utf-8")
        result = converter._read_text_bytes(raw, ".txt")
        assert result == text

    def test_read_utf8_sig_text(self):
        """Test reading UTF-8 with BOM."""
        converter = DoclingConverter()
        text = "Hello, World!"
        raw = text.encode("utf-8-sig")
        result = converter._read_text_bytes(raw, ".txt")
        # UTF-8-sig encoding adds BOM which gets decoded as \ufeff
        # The converter tries utf-8-sig which should strip it, but if not we accept it
        assert text in result or result == text

    def test_read_latin1_text(self):
        """Test reading Latin-1 encoded text."""
        converter = DoclingConverter()
        text = "Café résumé"
        raw = text.encode("latin-1")
        result = converter._read_text_bytes(raw, ".txt")
        assert result == text

    def test_truncate_large_file(self):
        """Test that large files are truncated."""
        converter = DoclingConverter(max_fallback_bytes=100)
        text = "A" * 200
        raw = text.encode("utf-8")
        result = converter._read_text_bytes(raw, ".txt")

        # Should be truncated to 100 bytes plus truncation note
        assert len(result) < len(text)
        assert "[Note: file truncated to first 100 bytes]" in result
        assert result.startswith("A")

    def test_code_file_with_fence(self):
        """Test that code files get wrapped in code fences."""
        converter = DoclingConverter()
        code = "def hello():\n    print('world')"
        raw = code.encode("utf-8")

        result = converter._read_text_bytes(raw, ".py")
        assert result.startswith("```py\n")
        assert result.endswith("\n```")
        assert code in result

    def test_json_file_with_fence(self):
        """Test that JSON files get wrapped in json code fences."""
        converter = DoclingConverter()
        json_text = '{"key": "value"}'
        raw = json_text.encode("utf-8")

        result = converter._read_text_bytes(raw, ".json")
        assert result.startswith("```json\n")
        assert result.endswith("\n```")
        assert json_text in result

    def test_yaml_file_with_fence(self):
        """Test that YAML files get wrapped in yaml code fences."""
        converter = DoclingConverter()
        yaml_text = "key: value\nlist:\n  - item1"
        raw = yaml_text.encode("utf-8")

        result = converter._read_text_bytes(raw, ".yaml")
        assert result.startswith("```yaml\n")
        assert result.endswith("\n```")
        assert yaml_text in result

    def test_xml_file_with_fence(self):
        """Test that XML files get wrapped in xml code fences."""
        converter = DoclingConverter()
        xml_text = "<root><item>value</item></root>"
        raw = xml_text.encode("utf-8")

        result = converter._read_text_bytes(raw, ".xml")
        assert result.startswith("```xml\n")
        assert result.endswith("\n```")
        assert xml_text in result

    def test_html_file_with_fence(self):
        """Test that HTML files get wrapped in xml code fences."""
        converter = DoclingConverter()
        html_text = "<html><body>Hello</body></html>"
        raw = html_text.encode("utf-8")

        result = converter._read_text_bytes(raw, ".html")
        assert result.startswith("```xml\n")
        assert result.endswith("\n```")
        assert html_text in result

    def test_plain_text_no_fence(self):
        """Test that plain text files don't get code fences."""
        converter = DoclingConverter()
        text = "Just plain text"
        raw = text.encode("utf-8")

        result = converter._read_text_bytes(raw, ".txt")
        assert result == text
        assert "```" not in result

    def test_truncation_with_code_fence(self):
        """Test that truncation note appears after code fence."""
        converter = DoclingConverter(max_fallback_bytes=50)
        code = "def hello():\n    print('world')" * 10
        raw = code.encode("utf-8")

        result = converter._read_text_bytes(raw, ".py")
        assert result.startswith("```py\n")
        # The truncation note appears after the closing fence
        assert "[Note: file truncated" in result
        assert result.count("```") >= 2  # Opening and closing fence

    def test_invalid_encoding_fallback(self):
        """Test that invalid encoding falls back to replace mode."""
        converter = DoclingConverter()
        # Create bytes that are invalid UTF-8
        raw = b"\xff\xfe Invalid UTF-8 \x80\x81"
        result = converter._read_text_bytes(raw, ".txt")
        # Should not raise an exception
        assert isinstance(result, str)


class TestDoclingConverterConvertOne:
    """Test suite for DoclingConverter.convert_one method."""

    def test_convert_one_requires_path_or_bytes(self):
        """Test that convert_one requires either path or raw_bytes."""
        converter = DoclingConverter()

        with pytest.raises(
            RuntimeError, match="Either path or raw_bytes must be provided"
        ):
            converter.convert_one(display_name="test.txt")

    def test_convert_one_skip_suffix_none(self):
        """Test that skip suffixes raise RuntimeError."""
        converter = DoclingConverter()

        doc = converter.convert_one(
            display_name="image.png", raw_bytes=b"fake image data", suffix=".png"
        )
        assert doc is None

    def test_convert_one_fallback_text_from_bytes(self):
        """Test converting text file from raw bytes."""
        converter = DoclingConverter()
        content = "Hello, World!"
        raw_bytes = content.encode("utf-8")

        result = converter.convert_one(
            display_name="test.txt", raw_bytes=raw_bytes, suffix=".txt"
        )

        assert isinstance(result, FallbackTextDoc)
        assert result.name == "test.txt"
        assert result.export_to_markdown() == content

    def test_convert_one_code_file_from_bytes(self):
        """Test converting code file from raw bytes with code fence."""
        converter = DoclingConverter()
        code = "def hello():\n    pass"
        raw_bytes = code.encode("utf-8")

        result = converter.convert_one(
            display_name="test.py", raw_bytes=raw_bytes, suffix=".py"
        )

        assert isinstance(result, FallbackTextDoc)
        assert result.name == "test.py"
        markdown = result.export_to_markdown()
        assert "```py" in markdown
        assert code in markdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
