"""KramaBench data reader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ragworkbench import DatasetName

from ...implementations.ragbench_data_reader import RagbenchDataReader


class KramaBenchDataReader(RagbenchDataReader):
    """
    Reads KramaBench dataset using RagDataLoaderFactory.

    This reader loads both the corpus and benchmark data from KramaBench.
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
        Initialize the KramaBench data reader.

        Args:
            split: Dataset split to load ("train" or "test")
            question_limit: Optional limit on number of questions (None = all)
            document_factor: Optional factor for non-relevant documents (None = all)
            seed: Random seed for sampling (default: 43)
            use_cache: Whether to use file caching for loaded data (default: True)
            cache_base_dir: Base directory for cache storage (e.g., benchmark cache dir).
                          If None, caching will be disabled.
        """
        super().__init__(
            split=split,
            question_limit=question_limit,
            document_factor=document_factor,
            seed=seed,
            use_cache=use_cache,
            cache_base_dir=cache_base_dir,
        )

    def get_dataset_name(self) -> DatasetName:
        return DatasetName.KRAMABENCH
