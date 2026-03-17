"""Tests for core types - immutability validation only."""

from dataclasses import FrozenInstanceError
from io import BytesIO

import pytest

from experiments.core.types import (
    AgentOutput,
    BenchmarkEntry,
    Document,
    EvalResult,
    GroundTruth,
    ProcessedBenchmark,
)


class TestImmutability:
    """Test that all core types are immutable (frozen dataclasses)."""

    def test_document_is_frozen(self):
        """Test Document immutability (critical for data integrity)."""
        doc = Document(
            document_id="doc1",
            path="test.txt",
            mime_type="text/plain",
            extra_metadata={},
            stream_factory=lambda: BytesIO(b"test"),
        )
        with pytest.raises(FrozenInstanceError):
            doc.document_id = "doc2"  # type: ignore

    def test_ground_truth_is_frozen(self):
        """Test GroundTruth immutability (critical for evaluation integrity)."""
        gt = GroundTruth(answers=["answer1"])
        with pytest.raises(FrozenInstanceError):
            gt.answers = ["answer2"]  # type: ignore

    def test_benchmark_entry_is_frozen(self):
        """Test BenchmarkEntry immutability (critical for benchmark integrity)."""
        entry = BenchmarkEntry(
            question_id="q1",
            question="What is 2+2?",
            ground_truth=GroundTruth(answers=["4"]),
        )
        with pytest.raises(FrozenInstanceError):
            entry.question_id = "q2"  # type: ignore

    def test_processed_benchmark_is_frozen(self):
        """Test ProcessedBenchmark immutability (critical for benchmark integrity)."""
        benchmark = ProcessedBenchmark(
            question_id="q1",
            question="Test?",
            ground_truth=GroundTruth(answers=["Test"]),
        )
        with pytest.raises(FrozenInstanceError):
            benchmark.question = "New question"  # type: ignore

    def test_agent_output_is_frozen(self):
        """Test AgentOutput immutability (critical for result integrity)."""
        output = AgentOutput(
            question_id="q1",
            answer="Test answer",
        )
        with pytest.raises(FrozenInstanceError):
            output.answer = "New answer"  # type: ignore

    def test_eval_result_is_frozen(self):
        """Test EvalResult immutability (critical for evaluation integrity)."""
        result = EvalResult(
            question_id="q1",
            score=0.95,
            passed=True,
        )
        with pytest.raises(FrozenInstanceError):
            result.score = 0.5  # type: ignore
