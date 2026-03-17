"""Interface for reading data (corpus and benchmarks)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

from ..core.types import BenchmarkEntry

# Type variable for corpus data type
CorpusT = TypeVar("CorpusT")


class DataReader(ABC, Generic[CorpusT]):
    """
    Interface for reading data including corpus and benchmarks.

    A DataReader is responsible for loading both:
    1. The data/corpus (e.g., documents, knowledge base)
    2. The benchmark questions/tasks

    Type Parameters:
        CorpusT: The type of the corpus data returned by get_data()

    Example:
        class MyDataReader(DataReader[List[Document]]):
            def get_data(self) -> List[Document]:
                return self.corpus

            def get_config(self) -> Dict[str, Any]:
                return {"split": self.split, "limit": self.limit}
    """

    @abstractmethod
    def read_data(self) -> None:
        """
        Read and load all data (corpus and benchmarks) from source.

        This method should load both the corpus and benchmarks and store them
        internally so they can be retrieved via get_data() and get_benchmark().
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> CorpusT:
        """
        Get the loaded corpus/data.

        Returns:
            The corpus data in the format specified by the type parameter

        Raises:
            ValueError: If read_data() hasn't been called yet
        """
        raise NotImplementedError

    @abstractmethod
    def get_benchmark(self) -> Sequence[BenchmarkEntry]:
        """
        Get the loaded benchmark entries.

        Returns:
            Sequence of raw benchmark entries

        Raises:
            ValueError: If read_data() hasn't been called yet
        """
        raise NotImplementedError
