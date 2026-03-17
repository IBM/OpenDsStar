#!/usr/bin/env env python
"""
Convert all parquet files in a directory to CSV files.

Usage:
    python scripts/convert_parquet_to_csv.py <directory>

This script:
1. Finds all .parquet files in the given directory (recursively)
2. Converts each to CSV using the project's parquet_to_csv_bytes utility
3. Saves CSVs in ../csvs subdirectory relative to the input directory
4. Names output files as parent_filename.csv (e.g., data/file.parquet -> csvs/data_file.csv)
"""

import argparse
import logging
import sys
from pathlib import Path

from ingestion.utils import parquet_to_csv_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parquet_to_csv(parquet_path: Path) -> str:
    """
    Convert a parquet file to CSV string using the project's utility function.

    Uses: src/ingestion/utils.py::parquet_to_csv_bytes

    Args:
        parquet_path: Path to the parquet file

    Returns:
        CSV content as string

    Raises:
        Exception: If conversion fails
    """
    try:
        # Use the standalone utility function
        csv_bytes = parquet_to_csv_bytes(
            display_name=parquet_path.name,
            path=parquet_path,
            raw_bytes=None,
        )

        # Convert bytes to string
        return csv_bytes.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to convert {parquet_path.name}: {e}")
        raise


def get_output_filename(parquet_path: Path, base_dir: Path) -> str:
    """
    Generate output filename as parent_filename.csv.

    Args:
        parquet_path: Path to the parquet file
        base_dir: Base directory to calculate relative path from

    Returns:
        Output filename (e.g., "parent_filename.csv")
    """
    # Get relative path from base directory
    try:
        rel_path = parquet_path.relative_to(base_dir)
    except ValueError:
        # If not relative, just use the filename
        rel_path = parquet_path

    # Get parent directory name and filename without extension
    parts = list(rel_path.parts[:-1])  # All parts except filename
    filename_stem = rel_path.stem  # Filename without .parquet extension

    # Combine: parent_filename
    if parts:
        output_name = "_".join(parts) + "_" + filename_stem + ".csv"
    else:
        output_name = filename_stem + ".csv"

    return output_name


def convert_directory(input_dir: Path, name_filter: str | None = None) -> None:
    """
    Convert all parquet files in a directory to CSV.

    Args:
        input_dir: Directory containing parquet files
        name_filter: Optional string filter - only convert files with this string in their name
    """
    # Create output directory: ../csvs relative to input directory
    output_dir = input_dir.parent / "csvs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input directory: {input_dir.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    if name_filter:
        logger.info(f"Name filter: '{name_filter}'")

    # Find all parquet files recursively
    all_parquet_files = list(input_dir.rglob("*.parquet"))

    # Apply name filter if provided
    if name_filter:
        parquet_files = [f for f in all_parquet_files if name_filter in f.name]
        logger.info(
            f"Found {len(all_parquet_files)} total parquet file(s), {len(parquet_files)} matching filter"
        )
    else:
        parquet_files = all_parquet_files
        logger.info(f"Found {len(parquet_files)} parquet file(s)")

    if not parquet_files:
        logger.warning(f"No matching .parquet files found in {input_dir}")
        return

    # Convert each file
    success_count = 0
    error_count = 0

    for parquet_path in parquet_files:
        try:
            # Generate output filename
            output_filename = get_output_filename(parquet_path, input_dir)
            output_path = output_dir / output_filename

            logger.info(f"Converting: {parquet_path.name} -> {output_filename}")

            # Convert parquet to CSV using the project's utility function
            csv_content = parquet_to_csv(parquet_path)

            # Write CSV file
            output_path.write_text(csv_content, encoding="utf-8")

            logger.info(f"✓ Saved: {output_path}")
            success_count += 1

        except Exception as e:
            logger.error(f"✗ Failed to convert {parquet_path.name}: {e}")
            error_count += 1

    # Summary
    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Errors:  {error_count}")
    logger.info(f"  Total:   {len(parquet_files)}")
    logger.info(f"Output directory: {output_dir.absolute()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert all parquet files in a directory to CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_parquet_to_csv.py /path/to/data
  python scripts/convert_parquet_to_csv.py ./my_data_folder
  python scripts/convert_parquet_to_csv.py /path/to/data --filter "train"
  python scripts/convert_parquet_to_csv.py ./data --filter "2024"

Output:
  CSV files will be saved in ../csvs relative to the input directory.
  Filenames will be in format: parent_filename.csv
  Use --filter to only convert files with specific string in their name.
        """,
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing parquet files to convert",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only convert files with this string in their name (optional)",
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.directory)
    if not input_dir.exists():
        logger.error(f"Directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        logger.error(f"Not a directory: {input_dir}")
        sys.exit(1)

    # Convert all parquet files
    try:
        convert_directory(input_dir, name_filter=args.filter)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
