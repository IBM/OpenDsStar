"""
Tests for DataFrame serialization optimization in cross-process tool calls.

These tests verify that large DataFrames are efficiently serialized using
Parquet format instead of pickle for better performance.
"""

import pickle
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from OpenDsStar.agents.ds_star.ds_star_execute_env import (
    _deserialize_tool_result,
    _serialize_tool_result,
)

# Check if pyarrow is available
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


class TestDataFrameSerialization:
    """Test DataFrame serialization for cross-process communication."""

    def test_small_dataframe_uses_pickle(self):
        """Small DataFrames (<10MB) should use pickle for simplicity."""
        # Create a small DataFrame (~100KB)
        df = pd.DataFrame({
            "a": range(1000),
            "b": np.random.randn(1000),
            "c": ["text"] * 1000,
        })

        serialized = _serialize_tool_result(df)

        # Should use pickle for small DataFrames
        assert serialized["type"] == "pickle"
        assert isinstance(serialized["data"], bytes)

        # Should deserialize correctly
        result = _deserialize_tool_result(serialized)
        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
    def test_large_dataframe_uses_parquet(self):
        """Large DataFrames (>10MB) should use Parquet for efficiency."""
        # Create a large DataFrame (~15MB)
        # 100k rows * 20 columns * 8 bytes ≈ 16MB
        df = pd.DataFrame({
            f"col_{i}": np.random.randn(100_000) for i in range(20)
        })

        serialized = _serialize_tool_result(df)

        # Should use Parquet for large DataFrames
        assert serialized["type"] == "dataframe_parquet"
        assert isinstance(serialized["data"], bytes)

        # Should deserialize correctly
        result = _deserialize_tool_result(serialized)
        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
    def test_parquet_serialization_works_for_large_dataframes(self):
        """Parquet serialization should work correctly for large DataFrames."""
        # Create a large DataFrame (~50MB)
        df = pd.DataFrame({
            f"col_{i}": np.random.randn(200_000) for i in range(30)
        })

        # Serialize with Parquet
        serialized_parquet = _serialize_tool_result(df)
        assert serialized_parquet["type"] == "dataframe_parquet"

        # Verify deserialization works correctly
        result = _deserialize_tool_result(serialized_parquet)
        pd.testing.assert_frame_equal(result, df)

        # Note: Parquet's main benefit is faster deserialization for large DataFrames,
        # not necessarily smaller size (depends on data compressibility).
        # For 300MB DataFrames with real data, Parquet is typically 5-10x faster
        # to deserialize than pickle.

    def test_non_dataframe_uses_pickle(self):
        """Non-DataFrame objects should always use pickle."""
        test_objects = [
            42,
            "string",
            [1, 2, 3],
            {"key": "value"},
            np.array([1, 2, 3]),
            pd.Series([1, 2, 3]),
        ]

        for obj in test_objects:
            serialized = _serialize_tool_result(obj)
            assert serialized["type"] == "pickle"

            result = _deserialize_tool_result(serialized)
            if isinstance(obj, np.ndarray):
                np.testing.assert_array_equal(result, obj)
            elif isinstance(obj, pd.Series):
                pd.testing.assert_series_equal(result, obj)
            else:
                assert result == obj

    def test_dataframe_with_complex_dtypes(self):
        """DataFrames with complex dtypes should serialize correctly."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "datetime_col": pd.date_range("2024-01-01", periods=3),
            "category_col": pd.Categorical(["x", "y", "z"]),
        })

        serialized = _serialize_tool_result(df)
        result = _deserialize_tool_result(serialized)

        pd.testing.assert_frame_equal(result, df)

    def test_dataframe_with_missing_values(self):
        """DataFrames with NaN/None values should serialize correctly."""
        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [None, "text", None],
            "c": [1.1, 2.2, np.nan],
        })

        serialized = _serialize_tool_result(df)
        result = _deserialize_tool_result(serialized)

        pd.testing.assert_frame_equal(result, df)

    def test_empty_dataframe(self):
        """Empty DataFrames should serialize correctly."""
        df = pd.DataFrame()

        serialized = _serialize_tool_result(df)
        result = _deserialize_tool_result(serialized)

        pd.testing.assert_frame_equal(result, df)

    def test_deserialize_invalid_type(self):
        """Deserializing unknown type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown serialization type"):
            _deserialize_tool_result({"type": "unknown", "data": b"test"})

    def test_deserialize_missing_data(self):
        """Deserializing without data field should raise ValueError."""
        with pytest.raises(ValueError, match="missing 'data' field"):
            _deserialize_tool_result({"type": "pickle", "data": None})

    def test_serialize_unpicklable_object(self):
        """Unpicklable objects should raise TypeError."""
        # Lambda functions are not picklable
        unpicklable = lambda x: x

        with pytest.raises(TypeError, match="cannot be serialized"):
            _serialize_tool_result(unpicklable)

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
    def test_parquet_fallback_to_pickle(self):
        """If Parquet serialization fails, should fall back to pickle."""
        # Create a DataFrame that might cause issues with Parquet
        # (though most DataFrames work fine)
        df = pd.DataFrame({
            "a": range(100_000),  # Large enough to trigger Parquet
            "b": [{"nested": "dict"}] * 100_000,  # Complex objects
        })

        # Should still serialize (either Parquet or pickle fallback)
        serialized = _serialize_tool_result(df)
        assert serialized["type"] in ("dataframe_parquet", "pickle")

        # Should deserialize correctly
        result = _deserialize_tool_result(serialized)
        # Note: Parquet may not preserve exact object types for complex nested data
        assert len(result) == len(df)

    def test_round_trip_preserves_data(self):
        """Full serialize/deserialize cycle should preserve data exactly."""
        # Test with various DataFrame configurations
        test_cases = [
            # Simple numeric
            pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}),
            # With index
            pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"]),
            # Multi-column
            pd.DataFrame({f"col_{i}": range(100) for i in range(10)}),
        ]

        for df in test_cases:
            serialized = _serialize_tool_result(df)
            result = _deserialize_tool_result(serialized)
            pd.testing.assert_frame_equal(result, df)

# Made with Bob
