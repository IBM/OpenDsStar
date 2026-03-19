"""
Test caching behavior for DoclingDescriptionBuilder.

Covers:
- process_directory path-based caching (should cache on second run)
- process_corpus stream_factory caching (should cache on second run)
- caching disabled (should never cache)
- stream content change should miss cache (content-hash-based key)
"""

import logging
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from experiments.core.types import Document
from ingestion.docling_based_ingestion.docling_description_builder import (
    DoclingDescriptionBuilder,
)


@pytest.fixture
def test_file(tmp_path: Path) -> Path:
    test_files_dir = tmp_path / "test_files"
    test_files_dir.mkdir(exist_ok=True)

    p = test_files_dir / "test_doc.md"
    p.write_text(
        """# Test Document

This is a test document for caching.

## Section 1
Some content here with enough text to make it substantial.

## Section 2
More content here to ensure the document is long enough for testing purposes.
""",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def mock_llm() -> Mock:
    mock = Mock(spec=BaseChatModel)

    long_description = (
        "This is a comprehensive test document that contains various sections "
        "about testing methodologies, best practices, and implementation details. "
        "The document provides valuable information for developers."
    )

    def batch_side_effect(batch_inputs):
        return [MagicMock(content=long_description) for _ in batch_inputs]

    mock.batch.side_effect = batch_side_effect
    mock.invoke.return_value = MagicMock(content=long_description)
    return mock


@pytest.fixture
def cache_setup(tmp_path: Path):
    """Provide cache directory setup for tests."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir(exist_ok=True)

    cache_dir = storage_root / "cache"
    db_storage = storage_root / "db"
    db_storage.mkdir(parents=True, exist_ok=True)
    db_uri = str(db_storage / "test_milvus.db")
    
    return {
        "cache_dir": cache_dir,
        "db_uri": db_uri,
    }


def _make_stream_doc(doc_id: str, path: str, data: bytes) -> Document:
    def sf():
        return BytesIO(data)

    return Document(
        document_id=doc_id,
        path=path,
        mime_type="text/markdown",
        extra_metadata={},
        stream_factory=sf,
    )


def test_process_directory_uses_cache_on_second_run(
    cache_setup: dict, mock_llm: Mock, test_file: Path, caplog
):
    caplog.set_level(logging.INFO)
    test_dir = test_file.parent

    # Patch Milvus and ModelBuilder
    with patch("ingestion.docling_based_ingestion.milvus_manager.Milvus") as milvus_cls:
        milvus_instance = MagicMock()
        milvus_cls.return_value = milvus_instance
        milvus_instance.add_documents.return_value = None

        with patch(
            "ingestion.docling_based_ingestion.docling_description_builder.ModelBuilder.build"
        ) as mock_build:
            mock_build.return_value = (mock_llm, "mock-model")

            builder = DoclingDescriptionBuilder(
                cache_dir=str(cache_setup["cache_dir"]),
                model="mock-model",
                embedding_model="ibm-granite/granite-embedding-english-r2",
                db_uri=cache_setup["db_uri"],
                enable_caching=True,
                max_content_length=50_000,
            )

            builder.analysis_cache.clear()
            builder.description_cache.clear()

            caplog.clear()
            _, results_1, _ = builder.process_directory(test_dir)

            assert len(results_1) == 1
            doc_id_1 = next(iter(results_1))
            assert results_1[doc_id_1]["success"]
            # First run should not have cached_md
            assert "cached_md=0" in caplog.text or "missing_md=1" in caplog.text

            caplog.clear()
            _, results_2, _ = builder.process_directory(test_dir)

            assert len(results_2) == 1
            doc_id_2 = next(iter(results_2))
            assert results_2[doc_id_2]["success"]
            # Second run should use cache
            assert "cached_md=1" in caplog.text and "missing_md=0" in caplog.text

            assert results_1[doc_id_1]["answer"] == results_2[doc_id_2]["answer"]
            assert (
                results_1[doc_id_1]["md_fingerprint"] == results_2[doc_id_2]["md_fingerprint"]
            )


def test_process_corpus_stream_uses_cache_on_second_run(
    cache_setup: dict, mock_llm: Mock, caplog
):
    caplog.set_level(logging.INFO)

    doc = _make_stream_doc(
        "stream_doc_1",
        "stream_doc_1.md",
        b"# Stream Doc\n\nhello world\n",
    )

    # Patch Milvus and ModelBuilder
    with patch("ingestion.docling_based_ingestion.milvus_manager.Milvus") as milvus_cls:
        milvus_instance = MagicMock()
        milvus_cls.return_value = milvus_instance
        milvus_instance.add_documents.return_value = None

        with patch(
            "ingestion.docling_based_ingestion.docling_description_builder.ModelBuilder.build"
        ) as mock_build:
            mock_build.return_value = (mock_llm, "mock-model")

            builder = DoclingDescriptionBuilder(
                cache_dir=str(cache_setup["cache_dir"]),
                model="mock-model",
                embedding_model="ibm-granite/granite-embedding-english-r2",
                db_uri=cache_setup["db_uri"],
                enable_caching=True,
                max_content_length=50_000,
            )

            builder.analysis_cache.clear()
            builder.description_cache.clear()

            caplog.clear()
            builder.process_corpus([doc])
            # First run should not have cached_md
            assert "cached_md=0" in caplog.text or "missing_md=1" in caplog.text

            caplog.clear()
            builder.process_corpus([doc])
            # Second run should use cache
            assert "cached_md=1" in caplog.text and "missing_md=0" in caplog.text


def test_process_corpus_stream_cache_miss_when_content_changes(
    cache_setup: dict, mock_llm: Mock, caplog
):
    caplog.set_level(logging.INFO)

    state = {"data": b"# Stream\nv1\n"}

    def sf():
        return BytesIO(state["data"])

    doc = Document(
        document_id="stream_doc_2",
        path="stream_doc_2.md",
        mime_type="text/markdown",
        extra_metadata={},
        stream_factory=sf,
    )

    # Patch Milvus and ModelBuilder
    with patch("ingestion.docling_based_ingestion.milvus_manager.Milvus") as milvus_cls:
        milvus_instance = MagicMock()
        milvus_cls.return_value = milvus_instance
        milvus_instance.add_documents.return_value = None

        with patch(
            "ingestion.docling_based_ingestion.docling_description_builder.ModelBuilder.build"
        ) as mock_build:
            mock_build.return_value = (mock_llm, "mock-model")

            builder = DoclingDescriptionBuilder(
                cache_dir=str(cache_setup["cache_dir"]),
                model="mock-model",
                embedding_model="ibm-granite/granite-embedding-english-r2",
                db_uri=cache_setup["db_uri"],
                enable_caching=True,
                max_content_length=50_000,
            )

            builder.analysis_cache.clear()
            builder.description_cache.clear()

            caplog.clear()
            builder.process_corpus([doc])
            # First run should not have cached_md
            assert "cached_md=0" in caplog.text or "missing_md=1" in caplog.text

            caplog.clear()
            builder.process_corpus([doc])
            # Second run should use cache
            assert "cached_md=1" in caplog.text and "missing_md=0" in caplog.text

            state["data"] = b"# Stream\nv2 changed\n"

            caplog.clear()
            builder.process_corpus([doc])
            # Content changed, should be cache miss
            assert "cached_md=0" in caplog.text or "missing_md=1" in caplog.text


def test_caching_disabled(test_file: Path, tmp_path: Path, mock_llm: Mock, caplog):
    """
    When enable_caching=False, should not use cache even on second run.

    Also: mock Milvus so this test doesn't depend on local Milvus config.
    """
    caplog.set_level(logging.INFO)

    storage_root = tmp_path / "storage"
    storage_root.mkdir(exist_ok=True)

    cache_dir = storage_root / "cache"
    db_storage = storage_root / "db"
    db_storage.mkdir(parents=True, exist_ok=True)
    db_uri = str(db_storage / "test_milvus.db")

    # Patch Milvus in the milvus_manager module where it's actually used
    with patch("ingestion.docling_based_ingestion.milvus_manager.Milvus") as milvus_cls:
        milvus_instance = MagicMock()
        milvus_cls.return_value = milvus_instance
        milvus_instance.add_documents.return_value = None

        # Also patch ModelBuilder to return our mock LLM
        with patch(
            "ingestion.docling_based_ingestion.docling_description_builder.ModelBuilder.build"
        ) as mock_build:
            mock_build.return_value = (mock_llm, "mock-model")

            builder = DoclingDescriptionBuilder(
                embedding_model="ibm-granite/granite-embedding-english-r2",
                db_uri=db_uri,
                cache_dir=str(cache_dir),
                enable_caching=False,
            )

            test_dir = test_file.parent

            caplog.clear()
            builder.process_directory(test_dir)
            # With caching disabled, should always be cache miss
            assert "cached_md=0" in caplog.text or "missing_md=1" in caplog.text

            caplog.clear()
            builder.process_directory(test_dir)
            # Second run should also be cache miss (caching disabled)
            assert "cached_md=0" in caplog.text or "missing_md=1" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
