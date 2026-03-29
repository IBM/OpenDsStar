"""
Utility functions for data ingestion and conversion.
"""

import logging
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def parquet_to_csv_bytes(
    display_name: str,
    path: Optional[Path] = None,
    raw_bytes: Optional[bytes] = None,
) -> bytes:
    """
    Convert parquet file to CSV bytes in-memory.

    This is a standalone utility function extracted from DoclingConverter
    to make parquet-to-CSV conversion reusable across the codebase.

    Args:
        display_name: Display name for logging purposes
        path: Path to parquet file (if reading from disk)
        raw_bytes: Raw parquet bytes (if already in memory)

    Returns:
        CSV content as bytes (UTF-8 encoded)

    Raises:
        RuntimeError: If neither path nor raw_bytes is provided
        Exception: If conversion fails

    Example:
        >>> from pathlib import Path
        >>> csv_bytes = parquet_to_csv_bytes("data.parquet", path=Path("data.parquet"))
        >>> csv_str = csv_bytes.decode("utf-8")
    """
    try:
        # Read parquet from either bytes or file path
        if raw_bytes is not None:
            df = pd.read_parquet(BytesIO(raw_bytes))
        elif path is not None:
            df = pd.read_parquet(path)
        else:
            raise RuntimeError("Either path or raw_bytes must be provided")

        # Convert to CSV in-memory
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        return csv_str.encode("utf-8")

    except Exception as e:
        logger.error("Failed to convert parquet to CSV | %s: %s", display_name, e)
        raise
