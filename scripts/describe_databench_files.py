#!/usr/bin/env python3
"""
Generate and print file descriptions for all files in the DataBench dataset.

Uses the tabular fast path for CSV files — skips docling, reads only metadata
and sample rows, then sends a compact summary to the LLM.

Usage:
    .venv/bin/python scripts/describe_databench_files.py
"""

import logging
import sys
from pathlib import Path

# Add src/ to path so imports work without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from OpenDsStar.agents.utils.logging_utils import init_logger
from OpenDsStar.ingestion.docling_based_ingestion.docling_description_builder import (
    DoclingDescriptionBuilder,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(
    "/Users/yoavkantor/Library/CloudStorage/OneDrive-IBM/"
    "AgenticRag/open_ds_star/data/DataBench_export/databench_data"
)
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "databench_descriptions1"


def main() -> None:
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    file_paths = sorted(DATA_DIR.iterdir())
    file_paths = [p for p in file_paths if p.is_file() and p.name != ".DS_Store"]
    logger.info("Found %d files in %s", len(file_paths), DATA_DIR)

    builder = DoclingDescriptionBuilder(
        cache_dir=str(CACHE_DIR),
        enable_caching=True,
    )

    results, _ = builder.describe_files(
        file_paths,
        progress_label="DataBench",
    )

    sorted_results = sorted(results.values(), key=lambda r: r.get("filename", ""))
    total = len(sorted_results)
    success_count = 0

    for i, result in enumerate(sorted_results, start=1):
        filename = result.get("filename", "unknown")
        success = result.get("success", False)

        if success:
            logger.info("[%d/%d] %s — OK\n%s", i, total, filename, result["answer"])
            success_count += 1
        else:
            logger.info(
                "[%d/%d] %s — FAILED: %s",
                i,
                total,
                filename,
                result.get("fatal_error", "unknown error"),
            )

    logger.info("Done: %d/%d files described successfully", success_count, total)


if __name__ == "__main__":
    init_logger()
    main()
