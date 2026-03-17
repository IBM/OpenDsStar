"""DataBench data reader implementation (HuggingFace: cardiffnlp/databench)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO, Callable, Literal, Sequence

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import snapshot_download

from ...core.types import BenchmarkEntry, Document, GroundTruth
from ...interfaces.data_reader import DataReader

logger = logging.getLogger(__name__)


class DataBenchDataReader(DataReader[Sequence[Document]]):
    """
    Reads DataBench dataset from HuggingFace (cardiffnlp/databench).

    Loads both QA and SemEval splits, downloads parquet data files,
    and converts to the experiment framework's format.
    """

    def __init__(
        self,
        qa_split: Literal["train"] = "train",
        semeval_split: Literal["train", "validation", "test"] = "train",
        question_limit: int | None = None,
        seed: int = 43,
        use_cache: bool = True,
        cache_base_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize DataBench data reader.

        Args:
            qa_split: HF split for config="qa" (typically "train")
            semeval_split: HF split for config="semeval"
            question_limit: Optional limit on number of questions (None = all)
            seed: Random seed for sampling
            use_cache: If True, cache datasets locally
            cache_base_dir: Base directory for caching
        """
        self.qa_split = qa_split
        self.semeval_split = semeval_split
        self.question_limit = question_limit
        self.seed = seed
        self.use_cache = use_cache
        self.cache_base_dir = (
            Path(cache_base_dir).expanduser().resolve() if cache_base_dir else None
        )

        self._hf_cache_dir = (
            (self.cache_base_dir / "hf_datasets") if self.cache_base_dir else None
        )

        # Internal state
        self._qa_dataset: Dataset | None = None
        self._semeval_dataset: Dataset | None = None
        self._benchmark_entries: Sequence[BenchmarkEntry] | None = None
        self._corpus: Sequence[Document] | None = None

    def _cache_path(self, config_name: str, split: str) -> Path | None:
        """Get cache path for a specific config/split."""
        if not (self.use_cache and self.cache_base_dir is not None):
            return None
        return (
            self.cache_base_dir
            / "processed"
            / "cardiffnlp__databench"
            / config_name
            / split
        )

    def _load_one_split(
        self, config_name: Literal["qa", "semeval"], split: str
    ) -> Dataset:
        """Load a single config/split from HuggingFace or cache."""
        cache_path = self._cache_path(config_name=config_name, split=split)

        # Try loading from cache first
        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading {config_name}:{split} from cache: {cache_path}")
            ds = load_from_disk(str(cache_path))
            if not isinstance(ds, Dataset):
                raise ValueError(f"Expected Dataset, got {type(ds)}")
            return ds

        # Download from HuggingFace
        logger.info(f"Downloading {config_name}:{split} from HuggingFace...")
        ds = load_dataset(
            "cardiffnlp/databench",
            name=config_name,
            split=split,
            cache_dir=str(self._hf_cache_dir) if self._hf_cache_dir else None,
        )

        if not isinstance(ds, Dataset):
            raise ValueError(f"Expected Dataset from load_dataset, got {type(ds)}")

        # Save to cache
        if cache_path is not None:
            cache_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(cache_path))
            logger.info(f"Saved {config_name}:{split} to cache: {cache_path}")

        return ds

    def _download_data_files(self) -> Path:
        """Download DataBench parquet data files from HuggingFace."""
        if self.cache_base_dir is None:
            raise ValueError("cache_base_dir must be set to download data files")

        # Files are stored in cache/data/data due to HF repo structure
        data_dir = self.cache_base_dir / "data" / "data"

        # Check if already downloaded
        if data_dir.exists() and any(data_dir.glob("**/*.parquet")):
            logger.info(f"Data files already exist in {data_dir}")
            return data_dir

        logger.info("Downloading DataBench data files...")
        snapshot_download(
            repo_id="cardiffnlp/databench",
            repo_type="dataset",
            local_dir=str(self.cache_base_dir / "data"),
            local_dir_use_symlinks=False,
            allow_patterns=["data/**/*.parquet", "data/**/*.yml"],
        )
        logger.info(f"Data files downloaded to {data_dir}")
        return data_dir

    def _load_corpus_from_parquet_files(self, data_dir: Path) -> Sequence[Document]:
        """Load only 'all.parquet' files from subdirectories as corpus Documents."""
        corpus = []
        # Only load "all.parquet" files from each dataset subdirectory
        parquet_files = list(data_dir.glob("**/all.parquet"))

        logger.info(f"Found {len(parquet_files)} 'all.parquet' files")

        for parquet_file in parquet_files:

            def make_stream_factory(file_path: Path) -> Callable[[], BinaryIO]:
                def stream_factory() -> BinaryIO:
                    return open(file_path, "rb")

                return stream_factory

            dataset_name = (
                parquet_file.parent.name
                if parquet_file.parent != data_dir
                else parquet_file.stem
            )

            doc = Document(
                document_id=f"{dataset_name}::{parquet_file.name}",
                path=str(parquet_file.relative_to(data_dir)),
                mime_type="application/parquet",
                extra_metadata={"dataset": dataset_name, "filename": parquet_file.name},
                stream_factory=make_stream_factory(parquet_file),
            )
            corpus.append(doc)

        logger.info(f"Created corpus with {len(corpus)} documents")
        return corpus

    def read_data(self) -> None:
        """Load DataBench data from HuggingFace and convert to experiment format."""
        logger.info(
            f"Loading DataBench: qa_split={self.qa_split}, "
            f"semeval_split={self.semeval_split}, "
            f"question_limit={self.question_limit}"
        )

        # Load both splits
        self._qa_dataset = self._load_one_split("qa", self.qa_split)
        self._semeval_dataset = self._load_one_split("semeval", self.semeval_split)

        # Download and load data files as corpus
        data_dir = self._download_data_files()
        self._corpus = self._load_corpus_from_parquet_files(data_dir)

        # Convert to benchmark entries
        self._benchmark_entries = self._convert_to_benchmark_entries()

        logger.info(
            f"DataBench data loaded: benchmark_entries={len(self._benchmark_entries)}, "
            f"corpus_size={len(self._corpus)}"
        )

    def _convert_to_benchmark_entries(self) -> Sequence[BenchmarkEntry]:
        """Convert HuggingFace datasets to BenchmarkEntry format."""
        entries = []

        # Convert QA dataset
        if self._qa_dataset is not None:
            for idx in range(len(self._qa_dataset)):
                row = self._qa_dataset[idx]  # type: ignore
                question_id = f"qa_{idx}"
                question = str(row.get("question", ""))  # type: ignore
                answer = str(row.get("answer", ""))  # type: ignore

                entry = BenchmarkEntry(
                    question_id=question_id,
                    question=question,  # Only the plain question
                    ground_truth=GroundTruth(
                        answers=[answer],
                        extra={"source": "qa", "row": dict(row), "dataset": row.get("dataset", "")},  # type: ignore
                    ),
                    additional_information={"split": self.qa_split, "source": "qa"},
                )
                entries.append(entry)

        # Convert SemEval dataset
        if self._semeval_dataset is not None:
            for idx in range(len(self._semeval_dataset)):
                row = self._semeval_dataset[idx]  # type: ignore
                question_id = f"semeval_{idx}"
                question = str(row.get("question", ""))  # type: ignore
                answer = str(row.get("answer", ""))  # type: ignore

                entry = BenchmarkEntry(
                    question_id=question_id,
                    question=question,  # Only the plain question
                    ground_truth=GroundTruth(
                        answers=[answer],
                        extra={"source": "semeval", "row": dict(row), "dataset": row.get("dataset", "")},  # type: ignore
                    ),
                    additional_information={
                        "split": self.semeval_split,
                        "source": "semeval",
                    },
                )
                entries.append(entry)

        # Apply question limit if specified
        if self.question_limit is not None and self.question_limit < len(entries):
            import random

            random.seed(self.seed)
            entries = random.sample(entries, self.question_limit)
            logger.info(f"Sampled {self.question_limit} questions from total")

        return entries

    def get_data(self) -> Sequence[Document]:
        """Get the corpus data (parquet files)."""
        if self._corpus is None:
            raise ValueError("Data not loaded yet. Call read_data() first.")
        return self._corpus

    def get_benchmark(self) -> Sequence[BenchmarkEntry]:
        """Get the benchmark entries."""
        if self._benchmark_entries is None:
            raise ValueError("Data not loaded yet. Call read_data() first.")
        return self._benchmark_entries
