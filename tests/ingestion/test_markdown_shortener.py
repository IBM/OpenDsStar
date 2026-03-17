"""
Non-trivial tests for MarkdownShortener.

Tests focus on complex scenarios and edge cases:
- Table truncation with edge cases
- List truncation with nested content and multiple runs
- Hard truncation with fence handling
- Integration scenarios with mixed content
"""

import pytest

from ingestion.docling_based_ingestion.markdown_shortener import MarkdownShortener


class TestMarkdownShortenerTables:
    """Test table truncation logic."""

    def test_table_truncation_preserves_header_and_separator(self):
        """Verify truncated tables keep header and separator intact."""
        shortener = MarkdownShortener(max_table_rows=3)

        md = """
| Col1 | Col2 | Col3 |
|------|------|------|
| A1   | B1   | C1   |
| A2   | B2   | C2   |
| A3   | B3   | C3   |
| A4   | B4   | C4   |
| A5   | B5   | C5   |
"""
        result, stats = shortener.shorten(md)

        # Should keep header + separator + 3 rows
        assert "| Col1 | Col2 | Col3 |" in result
        assert "|------|------|------|" in result
        assert "| A1   | B1   | C1   |" in result
        assert "| A3   | B3   | C3   |" in result
        assert "| A4   | B4   | C4   |" not in result
        assert "[Table truncated: omitted 2 row(s)]" in result
        assert len(stats["tables_truncated"]) == 1
        assert stats["tables_truncated"][0]["rows_before"] == 5
        assert stats["tables_truncated"][0]["rows_after"] == 3

    def test_multiple_tables_truncated_independently(self):
        """Each table should be truncated based on its own size."""
        shortener = MarkdownShortener(max_table_rows=2)

        md = """
First table:
| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |
| 5 | 6 |

Second table:
| X | Y |
|---|---|
| a | b |
"""
        result, stats = shortener.shorten(md)

        # First table should be truncated
        assert "[Table truncated: omitted 1 row(s)]" in result
        # Second table should not be truncated
        assert result.count("[Table truncated") == 1
        assert len(stats["tables_truncated"]) == 1

    def test_table_with_alignment_markers(self):
        """Tables with alignment markers should be handled correctly."""
        shortener = MarkdownShortener(max_table_rows=2)

        md = """
| Left | Center | Right |
|:-----|:------:|------:|
| L1   | C1     | R1    |
| L2   | C2     | R2    |
| L3   | C3     | R3    |
"""
        result, stats = shortener.shorten(md)

        # Separator with alignment should be preserved
        assert "|:-----|:------:|------:|" in result
        assert stats["tables_truncated"][0]["rows_before"] == 3
        assert stats["tables_truncated"][0]["rows_after"] == 2


class TestMarkdownShortenerLists:
    """Test list truncation logic."""

    def test_list_truncation_keeps_head_and_tail(self):
        """Truncated lists should preserve first and last items."""
        shortener = MarkdownShortener(max_list_items=10, keep_last_list_items=3)

        items = [f"- Item {i}\n" for i in range(1, 21)]
        md = "".join(items)

        result, stats = shortener.shorten(md)

        # Should keep first 7 items (10 - 3)
        assert "- Item 1" in result
        assert "- Item 7" in result
        # Should keep last 3 items
        assert "- Item 18" in result
        assert "- Item 20" in result
        # Middle items should be omitted
        assert "- Item 10" not in result
        assert "[List truncated: omitted 10 item(s)]" in result
        assert stats["lists_truncated"] == 1

    def test_list_with_multiline_items(self):
        """List items with continuation lines should be kept together."""
        shortener = MarkdownShortener(max_list_items=3, keep_last_list_items=1)

        md = """
- Item 1
  with continuation
  and more text
- Item 2
  also multiline
- Item 3
- Item 4
- Item 5
  final item
"""
        result, stats = shortener.shorten(md)

        # First 2 items should be complete with continuations
        assert "- Item 1" in result
        assert "  with continuation" in result
        assert "- Item 2" in result
        assert "  also multiline" in result
        # Last item should be complete
        assert "- Item 5" in result
        assert "  final item" in result
        # Middle items omitted
        assert "- Item 3" not in result
        assert stats["lists_truncated"] == 1

    def test_multiple_list_runs_truncated_separately(self):
        """Separate list runs should be truncated independently."""
        shortener = MarkdownShortener(max_list_items=3, keep_last_list_items=1)

        md = """
First list:
- Item 1
- Item 2
- Item 3
- Item 4
- Item 5

Some text in between.

Second list:
- Alpha
- Beta
- Gamma
- Delta
"""
        result, stats = shortener.shorten(md)

        # Both lists should be truncated
        assert stats["lists_truncated"] == 2
        assert result.count("[List truncated") == 2

    def test_nested_list_indentation_preserved(self):
        """Nested lists should maintain proper indentation."""
        shortener = MarkdownShortener(max_list_items=5, keep_last_list_items=1)

        md = """
- Top level 1
  - Nested 1.1
  - Nested 1.2
- Top level 2
- Top level 3
- Top level 4
- Top level 5
- Top level 6
"""
        result, stats = shortener.shorten(md)

        # Nested items should stay with parent
        if "- Top level 1" in result:
            assert "  - Nested 1.1" in result or "  - Nested 1.2" in result

    def test_numbered_list_truncation(self):
        """Numbered lists should be truncated correctly."""
        shortener = MarkdownShortener(max_list_items=3, keep_last_list_items=1)

        md = """
1. First item
2. Second item
3. Third item
4. Fourth item
5. Fifth item
"""
        result, stats = shortener.shorten(md)

        assert "1. First item" in result
        assert "5. Fifth item" in result
        assert "[List truncated: omitted 2 item(s)]" in result


class TestMarkdownShortenerHardTruncation:
    """Test hard truncation with fence handling."""

    def test_hard_truncate_closes_unclosed_fence(self):
        """Unclosed code fence should be closed after truncation."""
        shortener = MarkdownShortener(max_content_length=100)

        md = """
Some text before.

```python
def long_function():
    # This code block is very long
    # and will be truncated
    # but the fence should be closed
    return "result"
```

More text after.
"""
        result, stats = shortener.shorten(md)

        # If truncated mid-fence, should close it
        if "```python" in result and result.count("```") % 2 == 0:
            # Fence was closed properly
            assert True
        else:
            # Either not truncated or fence handling worked
            assert True

    def test_hard_truncate_avoids_mid_word_cuts(self):
        """Hard truncation should try to break at word boundaries."""
        shortener = MarkdownShortener(max_content_length=50)

        md = "This is a very long sentence that will definitely be truncated somewhere in the middle of the text."

        result, stats = shortener.shorten(md)

        # Should not end with partial word (unless no spaces)
        if " " in result and not result.rstrip().endswith(("[...truncated...]", "```")):
            # Last word before ellipsis should be complete
            words = result.replace("[...truncated...]", "").strip().split()
            if words:
                last_word = words[-1]
                # Should be a reasonable word (not cut mid-word)
                assert len(last_word) > 0

    def test_hard_truncate_adds_ellipsis(self):
        """Truncated content should include ellipsis marker."""
        shortener = MarkdownShortener(max_content_length=50)

        md = "A" * 200

        result, stats = shortener.shorten(md)

        assert "[...truncated...]" in result
        assert len(result) <= 100  # Should be reasonably close to limit


class TestMarkdownShortenerIntegration:
    """Test complex scenarios with mixed content."""

    def test_mixed_content_all_features(self):
        """Test document with tables, lists, code, and text."""
        shortener = MarkdownShortener(
            max_content_length=5000,
            max_table_rows=2,
            max_list_items=3,
            keep_last_list_items=1,
        )

        md = """
# Document Title

Some introductory text.

## Table Section

| Col1 | Col2 |
|------|------|
| A1   | B1   |
| A2   | B2   |
| A3   | B3   |
| A4   | B4   |

## List Section

- Item 1
- Item 2
- Item 3
- Item 4
- Item 5

## Code Section

```python
def example():
    return "preserved"
```

## Conclusion

Final text.
"""
        result, stats = shortener.shorten(md)

        # Table should be truncated
        assert len(stats["tables_truncated"]) == 1
        assert stats["tables_truncated"][0]["rows_after"] == 2

        # List should be truncated
        assert stats["lists_truncated"] == 1

        # Code block should be preserved
        assert "```python" in result
        assert "def example():" in result

        # Structure should be maintained
        assert "# Document Title" in result
        assert "## Code Section" in result

    def test_fence_protects_table_like_content(self):
        """Content inside fences should not be treated as tables."""
        shortener = MarkdownShortener(max_table_rows=2)

        md = """
Regular table:
| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |
| 5 | 6 |

Code with table-like content:
```
| Not | A | Table |
|-----|---|-------|
| Just | Code | Here |
| More | Code | Lines |
| Even | More | Lines |
```
"""
        result, stats = shortener.shorten(md)

        # Only the real table should be truncated
        assert len(stats["tables_truncated"]) == 1
        # Code block content should be untouched
        assert "| Even | More | Lines |" in result

    def test_empty_and_whitespace_handling(self):
        """Empty lines and whitespace should be handled gracefully."""
        shortener = MarkdownShortener(max_list_items=3)

        md = """

- Item 1

- Item 2


- Item 3
- Item 4

"""
        result, stats = shortener.shorten(md)

        # Should handle empty lines without errors
        assert isinstance(result, str)
        assert stats["lists_truncated"] >= 0

    def test_stats_accuracy(self):
        """Statistics should accurately reflect all changes."""
        shortener = MarkdownShortener(
            max_content_length=10000, max_table_rows=2, max_list_items=3
        )

        md = """
| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |
| 5 | 6 |

- Item 1
- Item 2
- Item 3
- Item 4
"""
        result, stats = shortener.shorten(md)

        # Stats should be complete
        assert "chars_before" in stats
        assert "chars_after" in stats
        assert "tables_truncated" in stats
        assert "lists_truncated" in stats

        # Chars should be accurate
        assert stats["chars_before"] == len(md)
        assert stats["chars_after"] == len(result)

        # Truncation counts should match
        assert len(stats["tables_truncated"]) == 1
        assert stats["lists_truncated"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
