"""DataBench data reader implementation."""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Sequence

import pandas as pd

from experiments.core.context import PipelineContext
from experiments.core.types import BenchmarkEntry, Document, GroundTruth
from experiments.interfaces.data_reader import DataReader

logger = logging.getLogger(__name__)


class DataBenchDataReader(DataReader[Sequence[Document]]):
    """
    DataBench data reader that loads 10 CSV datasets as corpus.

    Each CSV is converted to a Document for use with retrieval tools.
    """

    def __init__(self, base_path: str = "databench_subset"):
        """
        Initialize DataBench data reader.

        Args:
            base_path: Base directory containing DataBench datasets
        """
        self.base_path = base_path
        self._corpus: Sequence[Document] | None = None
        self._benchmarks: Sequence[BenchmarkEntry] | None = None
        self.available_datasets = [
            "001_Forbes",
            "006_London",
            "007_Fifa",
            "013_Roller",
            "014_Airbnb",
            "020_Real",
            "030_Professionals",
            "040_Speed",
            "058_US",
            "070_OpenFoodFacts",
        ]

    def read_data(self) -> None:
        """Read and load all data (corpus and benchmarks) from source."""
        # We don't need a full context for simple reading
        self._corpus = self._read_corpus_internal()
        self._benchmarks = self._read_benchmarks_internal()

    def get_data(self) -> Sequence[Document]:
        """Get the loaded corpus/data."""
        if self._corpus is None:
            raise ValueError("read_data() must be called before get_data()")
        return self._corpus

    def get_benchmark(self) -> Sequence[BenchmarkEntry]:
        """Get the loaded benchmark entries."""
        if self._benchmarks is None:
            raise ValueError("read_data() must be called before get_benchmark()")
        return self._benchmarks

    @property
    def name(self) -> str:
        """Name of this data reader."""
        return "databench_reader"

    def _read_benchmarks_internal(self) -> Sequence[BenchmarkEntry]:
        """
        Read DataBench benchmark entries.

        Args:
            ctx: Pipeline context
            **kwargs: Additional arguments

        Returns:
            Sequence of BenchmarkEntry objects with questions
        """
        logger.info(f"Reading DataBench benchmarks from {self.base_path}")

        benchmarks = []
        for dataset_id in self.available_datasets:
            questions_file = os.path.join(
                self.base_path, dataset_id, f"{dataset_id}_questions.csv"
            )

            if not os.path.exists(questions_file):
                logger.warning(f"Questions file not found: {questions_file}")
                continue

            try:
                questions_df = pd.read_csv(questions_file)

                for idx, row in questions_df.iterrows():
                    benchmark = BenchmarkEntry(
                        question_id=f"{dataset_id}_{idx}",
                        question=str(row.get("question", "")),
                        ground_truth=GroundTruth(
                            answers=[str(row.get("answer", ""))],
                            extra={"dataset_id": dataset_id, "row_index": idx},
                        ),
                        additional_information={"dataset_id": dataset_id},
                    )
                    benchmarks.append(benchmark)

                logger.info(f"Loaded {len(questions_df)} questions from {dataset_id}")

            except Exception as e:
                logger.error(f"Error loading questions from {dataset_id}: {e}")
                continue

        logger.info(f"Total benchmarks loaded: {len(benchmarks)}")
        return benchmarks

    def read_benchmarks(
        self, ctx: PipelineContext, **kwargs: Any
    ) -> Sequence[BenchmarkEntry]:
        """Read DataBench benchmark entries (for compatibility)."""
        return self._read_benchmarks_internal()

    def _read_corpus_internal(self) -> Sequence[Document]:
        """
        Read DataBench corpus (10 CSV datasets as documents).

        Each CSV is converted to a Document with its content as text.

        Args:
            ctx: Pipeline context
            **kwargs: Additional arguments

        Returns:
            Sequence of Document objects
        """
        logger.info(f"Reading DataBench corpus from {self.base_path}")

        corpus = []
        for dataset_id in self.available_datasets:
            csv_file = os.path.join(self.base_path, dataset_id, f"{dataset_id}.csv")

            if not os.path.exists(csv_file):
                logger.warning(f"CSV file not found: {csv_file}")
                continue

            try:
                df = pd.read_csv(csv_file)

                # Convert DataFrame to text representation
                # Include column names and data
                text_parts = [
                    f"Dataset: {dataset_id}",
                    f"Columns: {', '.join(df.columns.tolist())}",
                    f"Number of rows: {len(df)}",
                    f"Number of columns: {len(df.columns)}",
                    "\nData:",
                ]

                # Add sample rows (first 100 to keep it manageable)
                sample_size = min(100, len(df))
                for idx, row in df.head(sample_size).iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    text_parts.append(f"Row {idx}: {row_text}")

                if len(df) > sample_size:
                    text_parts.append(f"\n... and {len(df) - sample_size} more rows")

                # Add summary statistics for numeric columns
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if numeric_cols:
                    text_parts.append("\nNumeric column statistics:")
                    for col in numeric_cols:
                        stats = df[col].describe()
                        text_parts.append(
                            f"{col}: mean={stats['mean']:.2f}, "
                            f"min={stats['min']:.2f}, max={stats['max']:.2f}"
                        )

                content = "\n".join(text_parts)

                # Create stream factory for the content
                def make_stream_factory(text_content: str):
                    def stream_factory():
                        return io.BytesIO(text_content.encode("utf-8"))

                    return stream_factory

                doc = Document(
                    document_id=dataset_id,
                    path=csv_file,
                    mime_type="text/csv",
                    extra_metadata={
                        "dataset_id": dataset_id,
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "columns": df.columns.tolist(),
                        "content": content,  # Store the text content in metadata
                    },
                    stream_factory=make_stream_factory(content),
                )
                corpus.append(doc)

                logger.info(
                    f"Loaded {dataset_id}: {len(df)} rows, {len(df.columns)} columns"
                )

            except Exception as e:
                logger.error(f"Error loading corpus from {dataset_id}: {e}")
                continue

        logger.info(f"Total corpus documents: {len(corpus)}")
        return corpus

    def read_corpus(self, ctx: PipelineContext, **kwargs: Any) -> Sequence[Document]:
        """Read DataBench corpus (for compatibility)."""
        return self._read_corpus_internal()


class SimpleBenchmarkReader(DataReader[None]):
    """Simple benchmark reader for demo/testing purposes."""

    def __init__(self, benchmarks: Sequence[BenchmarkEntry]):
        """Initialize with pre-created benchmarks."""
        self._benchmarks = benchmarks

    def read_data(self) -> None:
        """No-op for simple reader."""
        pass

    def get_data(self) -> None:
        """Simple reader has no corpus data."""
        return None

    def get_benchmark(self) -> Sequence[BenchmarkEntry]:
        """Return the pre-created benchmarks."""
        return self._benchmarks

    @property
    def name(self) -> str:
        """Name of this data reader."""
        return "simple_benchmark_reader"


def create_sample_benchmarks() -> Sequence[BenchmarkEntry]:
    """Create sample benchmark entries for testing."""
    return [
        BenchmarkEntry(
            question_id="demo_1",
            question="What is 2 + 2?",
            ground_truth=GroundTruth(answers=["4"]),
            additional_information={},
        ),
        BenchmarkEntry(
            question_id="demo_2",
            question="What is the capital of France?",
            ground_truth=GroundTruth(answers=["Paris"]),
            additional_information={},
        ),
    ]
