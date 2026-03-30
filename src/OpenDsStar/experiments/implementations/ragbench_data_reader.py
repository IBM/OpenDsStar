"""Base class for ragbench-based data readers."""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal, Sequence

from ragworkbench import DataLoaderFactory, DatasetName
from ragworkbench.datasets_loader.data_models import DataSamplingParams

from ..core.types import BenchmarkEntry, Document
from ..interfaces.data_reader import DataReader
from .ragbench_converters import (
    convert_ragbench_benchmark_to_entries,
    convert_ragbench_corpus_to_documents,
)

logger = logging.getLogger(__name__)


class RagbenchDataReader(DataReader):
    """
    Base class for reading datasets using RagDataLoaderFactory from ragbench package.

    This reader provides common functionality for loading both corpus and benchmark data
    from ragbench-based datasets. Subclasses need to specify the dataset name and answer type.
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "test",
        question_limit: int | None = None,
        document_factor: int | None = None,
        seed: int | None = None,
        use_cache: bool = True,
        cache_base_dir: Path | str | None = None,
    ):
        """
        Initialize the ragbench data reader.

        Args:
            split: Dataset split to load ("train" or "test")
            question_limit: Optional limit on number of questions (None = all)
            document_factor: Optional factor for non-relevant documents (None = all)
            seed: Random seed for sampling (default: 43)
            use_cache: Whether to use file caching for loaded data (default: True)
            cache_base_dir: Base directory for cache storage (e.g., benchmark cache dir).
                          If None, caching will be disabled.
        """
        self.split = split
        self.question_limit = question_limit
        self.document_factor = document_factor
        self.seed = seed
        self.use_cache = use_cache
        self.cache_base_dir = Path(cache_base_dir) if cache_base_dir else None
        self._corpus = None
        self._corpus_converted = None
        self._benchmark_data = None
        self._benchmark_entries = None

    @abstractmethod
    def get_dataset_name(self) -> DatasetName:
        """
        Get the dataset name for this reader.

        Returns:
            DatasetName enum value for the specific dataset
        """
        raise NotImplementedError

    @staticmethod
    def post_data_reading(corpus: Any, benchmark_data: Any) -> tuple[Any, Any]:
        """
        Optional hook called after data is read but before conversion.

        Subclasses can override this to perform dataset-specific preprocessing
        or data preparation. By default, returns the data as-is.

        Args:
            corpus: The loaded corpus data
            benchmark_data: The loaded benchmark data

        Returns:
            Tuple of (corpus, benchmark_data), potentially modified
        """
        return corpus, benchmark_data

    def read_data(self) -> None:
        """
        Load dataset data (corpus and benchmarks) using RagDataLoaderFactory.
        Uses file cache to avoid re-loading data on subsequent runs.
        """
        dataset_name = self.get_dataset_name()

        logger.info(
            f"Loading {dataset_name.value} data: split={self.split}, "
            f"question_limit={self.question_limit}, "
            f"document_factor={self.document_factor}"
        )

        # Cache miss or caching disabled - load from source
        logger.info(f"Loading {dataset_name.value} data from source...")

        # Create sampling params only if limits are specified
        sampling_params = None
        if self.question_limit is not None or self.document_factor is not None:
            sampling_params = DataSamplingParams(
                question_limit=self.question_limit,
                document_factor=self.document_factor,
                seed=self.seed if self.seed is not None else 43,
            )
            logger.info(f"Using data sampling: {sampling_params.as_id()}")

        # Load data using RagDataLoaderFactory
        data_loader = DataLoaderFactory.create_loader(
            dataset_name=dataset_name,
            split=self.split,  # type: ignore
            sampling_params=sampling_params or DataSamplingParams(),
            cache_dir=self.cache_base_dir,
        )

        self._corpus = data_loader.get_corpus()
        self._benchmark_data = data_loader.get_benchmark()

        # Call post-processing hook
        self._corpus, self._benchmark_data = self.post_data_reading(
            self._corpus, self._benchmark_data
        )

        corpus_size = (
            len(self._corpus) if hasattr(self._corpus, "__len__") else "unknown"
        )
        benchmark_size = (
            len(self._benchmark_data)
            if hasattr(self._benchmark_data, "__len__")
            else "unknown"
        )

        logger.info(
            f"{dataset_name.value} data loaded: corpus_size={corpus_size}, "
            f"benchmark_size={benchmark_size}"
        )

        # Convert to benchmark entries
        self._benchmark_entries = self._convert_to_benchmark_entries(
            self._benchmark_data
        )
        self._corpus_converted = convert_ragbench_corpus_to_documents(self._corpus)

    def get_data(self) -> Sequence[Document]:
        """
        Return the loaded corpus as a sequence of Documents.

        Returns:
            Sequence of Document objects from the corpus

        Raises:
            ValueError: If read_data() hasn't been called yet
        """
        if self._corpus_converted is None:
            raise ValueError("Data not loaded yet. Call read_data() first.")

        return self._corpus_converted

    def get_benchmark(self) -> Sequence[BenchmarkEntry]:
        """
        Return the loaded benchmark entries.

        Returns:
            Sequence of benchmark entries

        Raises:
            ValueError: If read_data() hasn't been called yet
        """
        if self._benchmark_entries is None:
            raise ValueError("Data not loaded yet. Call read_data() first.")
        return self._benchmark_entries

    @staticmethod
    def _convert_to_benchmark_entries(benchmark_data: Any) -> Sequence[BenchmarkEntry]:
        """
        Convert benchmark data to BenchmarkEntry format.

        Args:
            benchmark_data: Raw benchmark data from data loader (RagBenchmark)

        Returns:
            Sequence of BenchmarkEntry objects
        """
        return convert_ragbench_benchmark_to_entries(benchmark_data)
