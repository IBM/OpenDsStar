"""
Tests for the tabular fast path in DoclingDescriptionBuilder.

Covers:
- CSV/TSV/Excel/Parquet files route through the fast path
- PDF/DOCX/image files still go through docling
- Mixed input (some tabular, some not) routes correctly
- Output format matches the standard analysis_results structure
- Large CSV: only first N rows are read, not the whole file
- Caching works for tabular fast path
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel

from OpenDsStar.ingestion.docling_based_ingestion.docling_description_builder import (
    TABULAR_EXTENSIONS,
    DoclingDescriptionBuilder,
)

LONG_DESCRIPTION = (
    "This dataset contains information about property listings. "
    "Key columns include listing_id, name, host_id, and price. "
    "It can be used for rental market analysis and pricing studies."
)


@pytest.fixture
def mock_llm() -> Mock:
    mock = Mock(spec=BaseChatModel)

    def batch_side_effect(batch_inputs):
        return [MagicMock(content=LONG_DESCRIPTION) for _ in batch_inputs]

    def invoke_side_effect(prompt):
        result = MagicMock()
        result.content = LONG_DESCRIPTION
        return result

    mock.batch.side_effect = batch_side_effect
    mock.invoke.side_effect = invoke_side_effect
    return mock


@pytest.fixture
def builder(tmp_path: Path, mock_llm: Mock) -> DoclingDescriptionBuilder:
    cache_dir = tmp_path / "cache"
    with patch(
        "OpenDsStar.ingestion.docling_based_ingestion.docling_description_builder.ModelBuilder.build"
    ) as mock_build:
        mock_build.return_value = (mock_llm, "mock-model")
        return DoclingDescriptionBuilder(
            cache_dir=str(cache_dir),
            model="mock-model",
            embedding_model="ibm-granite/granite-embedding-english-r2",
            enable_caching=True,
            max_content_length=50_000,
        )


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": [f"Place {i}" for i in range(10)],
            "price": [100.0 + i * 10 for i in range(10)],
        }
    )
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def tsv_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.tsv"
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    df.to_csv(p, sep="\t", index=False)
    return p


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.parquet"
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def excel_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.xlsx"
    df = pd.DataFrame({"alpha": [10, 20], "beta": ["a", "b"]})
    df.to_excel(p, index=False)
    return p


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    p = tmp_path / "readme.md"
    p.write_text(
        "# Readme\n\nThis is a markdown document with enough text to be substantial.\n"
        "It has several sections and meaningful content for testing purposes.\n",
        encoding="utf-8",
    )
    return p


# -------------------------------------------------------
# Test: tabular extensions constant
# -------------------------------------------------------


def test_tabular_extensions_includes_expected():
    assert ".csv" in TABULAR_EXTENSIONS
    assert ".tsv" in TABULAR_EXTENSIONS
    assert ".xlsx" in TABULAR_EXTENSIONS
    assert ".xls" in TABULAR_EXTENSIONS
    assert ".parquet" in TABULAR_EXTENSIONS
    # JSON is NOT tabular (not inherently tabular)
    assert ".json" not in TABULAR_EXTENSIONS
    # Non-tabular formats should not be present
    assert ".pdf" not in TABULAR_EXTENSIONS
    assert ".docx" not in TABULAR_EXTENSIONS


# -------------------------------------------------------
# Test: _build_tabular_summary
# -------------------------------------------------------


def test_build_tabular_summary_csv(csv_file: Path):
    summary = DoclingDescriptionBuilder._build_tabular_summary(str(csv_file))
    assert "## File Name" in summary
    assert "data.csv" in summary
    assert "10 rows" in summary
    assert "3 columns" in summary
    assert "listing_id" in summary
    assert "price" in summary
    assert "## Sample Data" in summary


def test_build_tabular_summary_tsv(tsv_file: Path):
    summary = DoclingDescriptionBuilder._build_tabular_summary(str(tsv_file))
    assert "3 rows" in summary
    assert "col_a" in summary
    assert "col_b" in summary


def test_build_tabular_summary_parquet(parquet_file: Path):
    summary = DoclingDescriptionBuilder._build_tabular_summary(str(parquet_file))
    assert "3 rows" in summary
    assert "2 columns" in summary


def test_build_tabular_summary_excel(excel_file: Path):
    summary = DoclingDescriptionBuilder._build_tabular_summary(str(excel_file))
    assert "2 rows" in summary
    assert "alpha" in summary
    assert "beta" in summary


# -------------------------------------------------------
# Test: _read_tabular_metadata only reads first N rows for CSV
# -------------------------------------------------------


def test_csv_nrows_limit(tmp_path: Path):
    """Verify that for CSV, only sample_rows are read for the sample, not the full file."""
    p = tmp_path / "large.csv"
    df = pd.DataFrame({"a": range(10000), "b": range(10000)})
    df.to_csv(p, index=False)

    row_count, col_count, col_dtypes, sample_df = (
        DoclingDescriptionBuilder._read_tabular_metadata(str(p), sample_rows=5)
    )

    assert row_count == 10000  # Full row count via usecols=[0]
    assert col_count == 2
    assert len(sample_df) == 5  # Only 5 rows in sample


# -------------------------------------------------------
# Test: CSV files go through fast path (not docling)
# -------------------------------------------------------


def test_csv_uses_fast_path(builder, csv_file: Path, caplog):
    caplog.set_level(logging.INFO)

    with patch("OpenDsStar.ingestion.docling_based_ingestion.milvus_manager.Milvus"):
        results, _ = builder.describe_files([csv_file], progress_label="TestCSV")

    assert "Fast path: analyzing 1 tabular file(s)" in caplog.text
    assert len(results) >= 1
    # At least one result should be successful
    any_success = any(r.get("success") for r in results.values())
    assert any_success


def test_parquet_uses_fast_path(builder, parquet_file: Path, caplog):
    caplog.set_level(logging.INFO)

    with patch("OpenDsStar.ingestion.docling_based_ingestion.milvus_manager.Milvus"):
        results, _ = builder.describe_files(
            [parquet_file], progress_label="TestParquet"
        )

    assert "Fast path: analyzing 1 tabular file(s)" in caplog.text


# -------------------------------------------------------
# Test: non-tabular files go through docling (not fast path)
# -------------------------------------------------------


def test_markdown_uses_docling(builder, md_file: Path, caplog):
    caplog.set_level(logging.INFO)

    with patch("OpenDsStar.ingestion.docling_based_ingestion.milvus_manager.Milvus"):
        results, _ = builder.describe_files([md_file], progress_label="TestMD")

    assert "Fast path" not in caplog.text


# -------------------------------------------------------
# Test: mixed input routes correctly
# -------------------------------------------------------


def test_mixed_input_splits_correctly(builder, csv_file: Path, md_file: Path, caplog):
    caplog.set_level(logging.INFO)

    with patch("OpenDsStar.ingestion.docling_based_ingestion.milvus_manager.Milvus"):
        results, _ = builder.describe_files(
            [csv_file, md_file], progress_label="TestMixed"
        )

    # Should see fast path for the CSV
    assert "Fast path: analyzing 1 tabular file(s)" in caplog.text
    # Should have results for both files
    assert len(results) >= 2


# -------------------------------------------------------
# Test: output format matches standard analysis_results
# -------------------------------------------------------


def test_tabular_result_format(builder, csv_file: Path):
    with patch("OpenDsStar.ingestion.docling_based_ingestion.milvus_manager.Milvus"):
        results, _ = builder.describe_files([csv_file], progress_label="TestFormat")

    assert len(results) >= 1
    for doc_id, result in results.items():
        # Standard keys that must be present
        assert "success" in result
        assert "answer" in result
        assert "file_path" in result
        assert "filename" in result
        assert "doc_id" in result
        assert "fatal_error" in result
        assert "outputs" in result


# -------------------------------------------------------
# Test: caching works for tabular fast path
# -------------------------------------------------------


def test_tabular_caching(builder, csv_file: Path, caplog):
    caplog.set_level(logging.INFO)

    with patch("OpenDsStar.ingestion.docling_based_ingestion.milvus_manager.Milvus"):
        # First call
        results_1, _ = builder.describe_files([csv_file], progress_label="TestCache1")

        # Reset mock call count
        builder.llm.invoke.reset_mock()

        # Second call should use cache
        results_2, _ = builder.describe_files([csv_file], progress_label="TestCache2")

    # LLM should NOT have been called on the second run (cached)
    builder.llm.invoke.assert_not_called()

    # Both runs should produce the same result
    for doc_id in results_1:
        if doc_id in results_2:
            assert results_1[doc_id]["answer"] == results_2[doc_id]["answer"]


def test_tabular_analysis_cache_avoids_rereading_file(builder, csv_file: Path):
    """Verify the tabular summary is stored in analysis_cache so the file isn't re-read."""
    with patch("OpenDsStar.ingestion.docling_based_ingestion.milvus_manager.Milvus"):
        builder.describe_files([csv_file], progress_label="TestAnalysisCache1")

    # The analysis cache should now have an entry for the CSV file
    cached = builder.analysis_cache.get(Path(str(csv_file.resolve())))
    assert cached is not None
    summary, _stats = cached
    assert "data.csv" in summary
    assert "Tabular data file" in summary


# -------------------------------------------------------
# Test: _build_tabular_summary column format
# -------------------------------------------------------


def test_tabular_summary_quotes_column_names(csv_file: Path):
    summary = DoclingDescriptionBuilder._build_tabular_summary(str(csv_file))
    assert "'listing_id' (int64)" in summary
    assert "'name' (object)" in summary
    assert "'price' (float64)" in summary


def test_tabular_summary_columns_one_per_line(csv_file: Path):
    summary = DoclingDescriptionBuilder._build_tabular_summary(str(csv_file))
    columns_section = summary.split("## Columns\n")[1].split("\n## ")[0]
    lines = [ln for ln in columns_section.strip().splitlines() if ln.strip()]
    assert len(lines) == 3  # listing_id, name, price
    for line in lines:
        assert line.startswith("- '")


# -------------------------------------------------------
# Test: _extract_columns_section
# -------------------------------------------------------


def test_extract_columns_section():
    summary = (
        "## Columns\n"
        "- 'col_a' (int64)\n"
        "- 'col_b' (object)\n\n"
        "## Sample Data (first 5 rows)\n| col_a | col_b |\n"
    )
    section = DoclingDescriptionBuilder._extract_columns_section(summary)
    assert "## Structured Data - Exact Column Names" in section
    assert "1. 'col_a' (int64)" in section
    assert "2. 'col_b' (object)" in section


# -------------------------------------------------------
# Test: _extract_sample_section
# -------------------------------------------------------


def test_extract_sample_section():
    summary = (
        "## Overview\nSome overview\n\n"
        "## Sample Data (first 5 rows)\n| a | b |\n|---|---|\n| 1 | 2 |"
    )
    section = DoclingDescriptionBuilder._extract_sample_section(summary)
    assert section.startswith("## Sampled rows/data")
    assert "| a | b |" in section


# -------------------------------------------------------
# Test: programmatic columns appended to description
# -------------------------------------------------------


def test_description_includes_programmatic_columns(builder, csv_file: Path):
    """Verify exact column names are appended when LLM omits them."""
    with patch("OpenDsStar.ingestion.docling_based_ingestion.milvus_manager.Milvus"):
        results, _ = builder.describe_files([csv_file], progress_label="TestCols")

    for result in results.values():
        assert result["success"]
        answer = result["answer"]
        assert "## Structured Data - Exact Column Names" in answer
        assert "'listing_id' (int64)" in answer
        assert "'name' (object)" in answer
        assert "'price' (float64)" in answer


def test_description_includes_programmatic_sample_rows(builder, csv_file: Path):
    """Verify sample rows are appended when LLM omits them."""
    with patch("OpenDsStar.ingestion.docling_based_ingestion.milvus_manager.Milvus"):
        results, _ = builder.describe_files([csv_file], progress_label="TestSample")

    for result in results.values():
        assert result["success"]
        answer = result["answer"]
        assert "## Sampled rows/data" in answer
        # Should contain actual data values from the CSV
        assert "Place 0" in answer or "listing_id" in answer
