"""Simple data reader implementation."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ...core.types import BenchmarkEntry, GroundTruth
from ...interfaces.data_reader import DataReader

logger = logging.getLogger(__name__)


class SimpleDataReader(DataReader):
    """
    Simple data reader that reads from a provided list.

    This is a placeholder implementation. Replace with your actual
    data loading logic (e.g., from files, databases, APIs, etc.).
    """

    def __init__(self, benchmarks: Sequence[BenchmarkEntry]) -> None:
        """
        Initialize with a list of benchmarks.

        Args:
            benchmarks: Pre-loaded benchmark entries
        """
        self._benchmarks = list(benchmarks)
        self._data = None  # No corpus data for simple demo

    def read_data(self) -> None:
        """
        Read data (no-op for demo, data is pre-loaded).
        """
        logger.info(f"reading_benchmarks count={len(self._benchmarks)}")
        # Data is already loaded in __init__

    def get_data(self) -> Any:
        """
        Get the corpus data (None for demo).

        Returns:
            None (no corpus for demo)
        """
        return self._data

    def get_benchmark(self) -> Sequence[BenchmarkEntry]:
        """
        Get the benchmark entries.

        Returns:
            Sequence of raw benchmark entries
        """
        return self._benchmarks


# Backward compatibility alias
SimpleBenchmarkReader = SimpleDataReader


def create_sample_benchmarks() -> list[BenchmarkEntry]:
    """Create sample benchmarks for testing."""
    return [
        BenchmarkEntry(
            question_id="q1",
            question="What is 2 + 2?",
            ground_truth=GroundTruth(
                answers=[4, "4"],
            ),
        ),
        BenchmarkEntry(
            question_id="q2",
            question="What is the capital of France?",
            ground_truth=GroundTruth(
                answers=["Paris"],
            ),
        ),
    ]
