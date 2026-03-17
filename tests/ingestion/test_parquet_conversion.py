"""Test parquet file conversion in DoclingConverter."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ingestion.docling_based_ingestion.docling_converter import DoclingConverter


@pytest.fixture
def sample_parquet_file():
    """Create a temporary parquet file for testing."""
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["New York", "London", "Paris"],
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_parquet_bytes():
    """Create parquet data as bytes for testing."""
    df = pd.DataFrame(
        {
            "product": ["Widget", "Gadget", "Doohickey"],
            "price": [10.99, 25.50, 5.00],
            "stock": [100, 50, 200],
        }
    )

    from io import BytesIO

    buffer = BytesIO()
    df.to_parquet(buffer)
    return buffer.getvalue()


def test_parquet_classification():
    """Test that .parquet files are classified correctly."""
    converter = DoclingConverter()
    assert converter.classify_suffix(".parquet") == "parquet_to_csv"
    assert converter.classify_suffix(".PARQUET") == "parquet_to_csv"


def test_parquet_from_path(sample_parquet_file):
    """Test converting parquet file from path."""
    converter = DoclingConverter()
    result = converter.convert_one(
        display_name="test.parquet", path=sample_parquet_file
    )

    assert result is not None
    markdown = result.export_to_markdown()

    # Check that CSV content is present
    assert "name" in markdown
    assert "age" in markdown
    assert "city" in markdown
    assert "Alice" in markdown
    assert "Bob" in markdown
    assert "Charlie" in markdown


def test_parquet_from_bytes(sample_parquet_bytes):
    """Test converting parquet from raw bytes."""
    converter = DoclingConverter()
    result = converter.convert_one(
        display_name="test.parquet", raw_bytes=sample_parquet_bytes, suffix=".parquet"
    )

    assert result is not None
    markdown = result.export_to_markdown()

    # Check that CSV content is present
    assert "product" in markdown
    assert "price" in markdown
    assert "stock" in markdown
    assert "Widget" in markdown
    assert "Gadget" in markdown


def test_parquet_conversion_preserves_data(sample_parquet_file):
    """Test that parquet to CSV conversion preserves all data."""
    # Read original data
    original_df = pd.read_parquet(sample_parquet_file)

    # Convert through DoclingConverter
    converter = DoclingConverter()
    result = converter.convert_one(
        display_name="test.parquet", path=sample_parquet_file
    )

    assert result is not None
    markdown = result.export_to_markdown()

    # Verify all values are present in the markdown
    for col in original_df.columns:
        assert col in markdown

    for value in original_df.values.flatten():
        assert str(value) in markdown
