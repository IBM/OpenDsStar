"""Tests for validation utilities."""

import pytest

from experiments.core.types import AgentOutput, GroundTruth, ProcessedBenchmark
from experiments.utils.validation import (
    ensure_unique_question_ids,
    index_by_question_id,
)


class TestEnsureUniqueQuestionIds:
    """Test ensure_unique_question_ids function."""

    def test_duplicate_question_ids_raises_error(self):
        """Test that duplicate question IDs raise ValueError with proper message."""
        gt = GroundTruth(answers=["answer"])
        benchmarks = [
            ProcessedBenchmark(question_id="q1", question="Q1?", ground_truth=gt),
            ProcessedBenchmark(question_id="q2", question="Q2?", ground_truth=gt),
            ProcessedBenchmark(question_id="q1", question="Q1 again?", ground_truth=gt),
        ]

        with pytest.raises(ValueError, match="Duplicate question_id"):
            ensure_unique_question_ids(benchmarks)

    def test_multiple_duplicates_in_error_message(self):
        """Test that error message includes all duplicate IDs."""
        gt = GroundTruth(answers=["answer"])
        benchmarks = [
            ProcessedBenchmark(question_id="q1", question="Q1?", ground_truth=gt),
            ProcessedBenchmark(question_id="q2", question="Q2?", ground_truth=gt),
            ProcessedBenchmark(question_id="q1", question="Q1 dup1?", ground_truth=gt),
            ProcessedBenchmark(question_id="q3", question="Q3?", ground_truth=gt),
            ProcessedBenchmark(question_id="q2", question="Q2 dup1?", ground_truth=gt),
            ProcessedBenchmark(question_id="q1", question="Q1 dup2?", ground_truth=gt),
        ]

        with pytest.raises(ValueError, match="Duplicate question_id") as exc_info:
            ensure_unique_question_ids(benchmarks)

        # Check that error message includes all duplicate IDs
        error_msg = str(exc_info.value)
        assert "q1" in error_msg
        assert "q2" in error_msg


class TestIndexByQuestionId:
    """Test index_by_question_id function."""

    def test_duplicate_raises_error(self):
        """Test that duplicate IDs raise ValueError during indexing."""
        gt = GroundTruth(answers=["answer"])
        benchmarks = [
            ProcessedBenchmark(question_id="q1", question="Q1?", ground_truth=gt),
            ProcessedBenchmark(question_id="q1", question="Q1 dup?", ground_truth=gt),
        ]

        with pytest.raises(ValueError, match="Duplicate question_id=q1"):
            index_by_question_id(benchmarks, "question_id")

    def test_custom_id_attribute(self):
        """Test that function works with custom attribute names (generic behavior)."""

        # Create simple objects with custom ID attribute
        class Item:
            def __init__(self, custom_id, value):
                self.custom_id = custom_id
                self.value = value

        items = [
            Item("id1", "value1"),
            Item("id2", "value2"),
            Item("id3", "value3"),
        ]

        result = index_by_question_id(items, "custom_id")

        assert len(result) == 3
        assert "id1" in result
        assert "id2" in result
        assert "id3" in result
        assert result["id1"].value == "value1"
        assert result["id2"].value == "value2"
        assert result["id3"].value == "value3"

    def test_preserves_insertion_order(self):
        """Test that dict preserves insertion order (Python 3.7+ guarantee)."""
        gt = GroundTruth(answers=["answer"])
        benchmarks = [
            ProcessedBenchmark(question_id="q3", question="Q3?", ground_truth=gt),
            ProcessedBenchmark(question_id="q1", question="Q1?", ground_truth=gt),
            ProcessedBenchmark(question_id="q2", question="Q2?", ground_truth=gt),
        ]

        result = index_by_question_id(benchmarks, "question_id")

        # Dict should preserve insertion order
        keys = list(result.keys())
        assert keys == ["q3", "q1", "q2"]

    def test_works_with_agent_outputs(self):
        """Test indexing different types (AgentOutput vs ProcessedBenchmark)."""
        outputs = [
            AgentOutput(question_id="q1", answer="a1"),
            AgentOutput(question_id="q2", answer="a2"),
            AgentOutput(question_id="q3", answer="a3"),
        ]

        result = index_by_question_id(outputs, "question_id")

        assert len(result) == 3
        assert result["q1"].answer == "a1"
        assert result["q2"].answer == "a2"
        assert result["q3"].answer == "a3"

    def test_missing_attribute_raises_error(self):
        """Test that missing attribute raises AttributeError (edge case)."""

        class Item:
            def __init__(self, value):
                self.value = value

        items = [Item("value1")]

        with pytest.raises(AttributeError):
            index_by_question_id(items, "nonexistent_attr")


class TestValidationIntegration:
    """Test integration between validation functions."""

    def test_workflow_catches_duplicates_at_both_stages(self):
        """Test that both validation functions catch duplicates (defense in depth)."""
        gt = GroundTruth(answers=["answer"])
        benchmarks = [
            ProcessedBenchmark(question_id="q1", question="Q1?", ground_truth=gt),
            ProcessedBenchmark(question_id="q1", question="Q1 dup?", ground_truth=gt),
        ]

        # ensure_unique should catch this
        with pytest.raises(ValueError, match="Duplicate"):
            ensure_unique_question_ids(benchmarks)

        # index_by_question_id would also catch it
        with pytest.raises(ValueError, match="Duplicate"):
            index_by_question_id(benchmarks, "question_id")
