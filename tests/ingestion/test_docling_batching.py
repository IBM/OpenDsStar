"""
Unit tests for batching utilities.

Tests the iter_batches function for processing items in chunks.
"""

import pytest

from OpenDsStar.ingestion.docling_based_ingestion.batching import iter_batches


class TestIterBatches:
    """Test suite for iter_batches function."""

    def test_empty_list(self):
        """Test batching an empty list."""
        result = list(iter_batches([], 5))
        assert result == []

    def test_single_batch(self):
        """Test when all items fit in a single batch."""
        items = [1, 2, 3, 4, 5]
        result = list(iter_batches(items, 10))
        assert len(result) == 1
        assert result[0] == [1, 2, 3, 4, 5]

    def test_exact_batches(self):
        """Test when items divide evenly into batches."""
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = list(iter_batches(items, 5))
        assert len(result) == 2
        assert result[0] == [1, 2, 3, 4, 5]
        assert result[1] == [6, 7, 8, 9, 10]

    def test_uneven_batches(self):
        """Test when items don't divide evenly into batches."""
        items = [1, 2, 3, 4, 5, 6, 7, 8]
        result = list(iter_batches(items, 3))
        assert len(result) == 3
        assert result[0] == [1, 2, 3]
        assert result[1] == [4, 5, 6]
        assert result[2] == [7, 8]  # Last batch is smaller

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        items = [1, 2, 3]
        result = list(iter_batches(items, 1))
        assert len(result) == 3
        assert result[0] == [1]
        assert result[1] == [2]
        assert result[2] == [3]

    def test_batch_size_larger_than_items(self):
        """Test when batch size is larger than number of items."""
        items = [1, 2, 3]
        result = list(iter_batches(items, 100))
        assert len(result) == 1
        assert result[0] == [1, 2, 3]

    def test_with_strings(self):
        """Test batching strings."""
        items = ["a", "b", "c", "d", "e", "f", "g"]
        result = list(iter_batches(items, 3))
        assert len(result) == 3
        assert result[0] == ["a", "b", "c"]
        assert result[1] == ["d", "e", "f"]
        assert result[2] == ["g"]

    def test_with_tuples(self):
        """Test batching tuples."""
        items = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]
        result = list(iter_batches(items, 2))
        assert len(result) == 3
        assert result[0] == [(1, "a"), (2, "b")]
        assert result[1] == [(3, "c"), (4, "d")]
        assert result[2] == [(5, "e")]

    def test_with_dicts(self):
        """Test batching dictionaries."""
        items = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
        result = list(iter_batches(items, 2))
        assert len(result) == 2
        assert result[0] == [{"id": 1}, {"id": 2}]
        assert result[1] == [{"id": 3}, {"id": 4}]

    def test_returns_lists(self):
        """Test that batches are returned as lists, not views."""
        items = [1, 2, 3, 4, 5]
        result = list(iter_batches(items, 2))

        # Verify each batch is a list
        for batch in result:
            assert isinstance(batch, list)

        # Verify modifying a batch doesn't affect original
        result[0][0] = 999
        assert items[0] == 1  # Original unchanged

    def test_iterator_behavior(self):
        """Test that iter_batches returns an iterator."""
        items = [1, 2, 3, 4, 5, 6]
        batches = iter_batches(items, 2)

        # Should be an iterator
        assert hasattr(batches, "__iter__")
        assert hasattr(batches, "__next__")

        # Can iterate manually
        batch1 = next(batches)
        assert batch1 == [1, 2]

        batch2 = next(batches)
        assert batch2 == [3, 4]

        batch3 = next(batches)
        assert batch3 == [5, 6]

        # Should raise StopIteration when exhausted
        with pytest.raises(StopIteration):
            next(batches)

    def test_large_batch(self):
        """Test with a large number of items."""
        items = list(range(1000))
        result = list(iter_batches(items, 100))

        assert len(result) == 10
        assert result[0] == list(range(0, 100))
        assert result[9] == list(range(900, 1000))

    def test_preserves_order(self):
        """Test that batching preserves item order."""
        items = list(range(20))
        result = list(iter_batches(items, 7))

        # Flatten and verify order
        flattened = []
        for batch in result:
            flattened.extend(batch)

        assert flattened == items

    def test_with_tuple_input(self):
        """Test that it works with tuple input (any sequence)."""
        items = (1, 2, 3, 4, 5)
        result = list(iter_batches(items, 2))

        assert len(result) == 3
        assert result[0] == [1, 2]
        assert result[1] == [3, 4]
        assert result[2] == [5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
