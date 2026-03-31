"""Tests for DataBench data reader."""

from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from OpenDsStar.experiments.benchmarks.databench.data_reader import DataBenchDataReader


@pytest.mark.unit
def test_databench_loads_only_all_parquet_files(tmp_path):
    """Test that DataBench data reader only loads all.parquet files, not qa.parquet or sample.parquet."""
    # Create mock directory structure
    data_dir = tmp_path / "data" / "data"
    data_dir.mkdir(parents=True)

    # Create subdirectories with different parquet files
    for dataset_name in ["001_Test", "002_Sample"]:
        dataset_dir = data_dir / dataset_name
        dataset_dir.mkdir()

        # Create all three types of parquet files
        (dataset_dir / "all.parquet").write_bytes(b"all_data")
        (dataset_dir / "qa.parquet").write_bytes(b"qa_data")
        (dataset_dir / "sample.parquet").write_bytes(b"sample_data")

    # Create a real Dataset object for mocking
    mock_dataset = Dataset.from_dict(
        {
            "question": ["test question"],
            "answer": ["test answer"],
            "dataset": ["test_dataset"],
        }
    )

    with patch(
        "OpenDsStar.experiments.benchmarks.databench.data_reader.load_from_disk",
        return_value=mock_dataset,
    ):
        with patch(
            "OpenDsStar.experiments.benchmarks.databench.data_reader.load_dataset",
            return_value=mock_dataset,
        ):
            # Create reader
            reader = DataBenchDataReader(
                qa_split="train",
                semeval_split="train",
                question_limit=1,
                use_cache=True,
                cache_base_dir=tmp_path,
            )

            # Mock _download_data_files to return our test directory
            reader._download_data_files = Mock(return_value=data_dir)

            # Load data
            reader.read_data()

            # Get corpus
            corpus = reader.get_data()

    # Verify corpus contains only all.parquet files
    assert (
        len(corpus) == 2
    ), f"Expected 2 documents (one per dataset), got {len(corpus)}"

    # Verify all documents are all.parquet files
    for doc in corpus:
        assert (
            "all.parquet" in doc.path
        ), f"Document path should contain 'all.parquet', got: {doc.path}"
        assert (
            "qa.parquet" not in doc.path
        ), f"Document should not be qa.parquet: {doc.path}"
        assert (
            "sample.parquet" not in doc.path
        ), f"Document should not be sample.parquet: {doc.path}"

    # Verify document IDs follow expected format
    for doc in corpus:
        assert (
            "::" in doc.document_id
        ), f"Document ID should contain '::', got: {doc.document_id}"
        assert doc.document_id.endswith(
            "::all.parquet"
        ), f"Document ID should end with '::all.parquet', got: {doc.document_id}"


@pytest.mark.unit
def test_databench_corpus_structure(tmp_path):
    """Test that DataBench corpus has expected structure."""
    # Create mock directory structure
    data_dir = tmp_path / "data" / "data"
    data_dir.mkdir(parents=True)

    dataset_dir = data_dir / "001_Test"
    dataset_dir.mkdir()
    (dataset_dir / "all.parquet").write_bytes(b"test_parquet_data")

    # Create a real Dataset object for mocking
    mock_dataset = Dataset.from_dict(
        {
            "question": ["test question"],
            "answer": ["test answer"],
            "dataset": ["test_dataset"],
        }
    )

    with patch(
        "OpenDsStar.experiments.benchmarks.databench.data_reader.load_from_disk",
        return_value=mock_dataset,
    ):
        with patch(
            "OpenDsStar.experiments.benchmarks.databench.data_reader.load_dataset",
            return_value=mock_dataset,
        ):
            reader = DataBenchDataReader(
                qa_split="train",
                semeval_split="train",
                question_limit=1,
                use_cache=True,
                cache_base_dir=tmp_path,
            )

            reader._download_data_files = Mock(return_value=data_dir)
            reader.read_data()
            corpus = reader.get_data()

    # Check document structure
    assert len(corpus) == 1
    doc = corpus[0]

    assert doc.document_id is not None
    assert doc.path is not None
    assert doc.mime_type == "application/parquet"
    assert doc.stream_factory is not None
    assert callable(doc.stream_factory)

    # Verify stream factory works
    stream = doc.stream_factory()
    assert stream is not None
    content = stream.read()
    assert content == b"test_parquet_data", "Stream should return the correct content"
