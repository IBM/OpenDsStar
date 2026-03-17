#!/usr/bin/env python
"""
Export DataBench benchmark data to CSV file.

This script loads the DataBench benchmark using the data reader
and exports the questions and answers to a CSV file.

Usage:
    python scripts/export_databench_to_csv.py [--question-limit N]
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.benchmarks.databench.data_reader import DataBenchDataReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def export_databench_to_csv(
    output_path: Path,
    question_limit: int | None = None,
    qa_split: str = "train",
    semeval_split: str = "train",
) -> None:
    """
    Export DataBench benchmark to CSV file.

    Args:
        output_path: Path where CSV file will be saved
        question_limit: Optional limit on number of questions
        qa_split: Split for QA config
        semeval_split: Split for SemEval config
    """
    from typing import Literal, cast

    logger.info("Initializing DataBench data reader...")

    # Initialize data reader with cache in databench directory
    cache_dir = (
        Path(__file__).parent.parent
        / "src"
        / "experiments"
        / "benchmarks"
        / "databench"
        / "cache"
    )

    reader = DataBenchDataReader(
        qa_split=cast(Literal["train"], qa_split),
        semeval_split=cast(Literal["train", "validation", "test"], semeval_split),
        question_limit=question_limit,
        seed=43,
        use_cache=True,
        cache_base_dir=cache_dir,
    )

    logger.info("Loading DataBench data...")
    reader.read_data()

    # Get benchmark entries
    benchmark_entries = reader.get_benchmark()
    logger.info(f"Loaded {len(benchmark_entries)} benchmark entries")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to CSV
    logger.info(f"Exporting to CSV: {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "question_id",
            "question",
            "answer",
            "source",
            "split",
            "dataset",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in benchmark_entries:
            # Get first answer from ground truth
            answer = entry.ground_truth.answers[0] if entry.ground_truth.answers else ""

            # Get metadata
            source = entry.additional_information.get("source", "")
            split = entry.additional_information.get("split", "")
            dataset = entry.ground_truth.extra.get("dataset", "")

            writer.writerow(
                {
                    "question_id": entry.question_id,
                    "question": entry.question,
                    "answer": answer,
                    "source": source,
                    "split": split,
                    "dataset": dataset,
                }
            )

    logger.info(
        f"✓ Successfully exported {len(benchmark_entries)} entries to {output_path}"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export DataBench benchmark data to CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/export_databench_to_csv.py
  python scripts/export_databench_to_csv.py --question-limit 50
  python scripts/export_databench_to_csv.py --question-limit 100

Output:
  CSV file will be saved to:
  src/experiments/benchmarks/databench/cache/csvs/databench_benchmark.csv
        """,
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=50,
        help="Limit number of questions to export (default: 50)",
    )
    parser.add_argument(
        "--qa-split",
        type=str,
        default="train",
        help="Split for QA config (default: train)",
    )
    parser.add_argument(
        "--semeval-split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Split for SemEval config (default: train)",
    )

    args = parser.parse_args()

    # Set output path
    output_path = (
        Path(__file__).parent.parent
        / "src"
        / "experiments"
        / "benchmarks"
        / "databench"
        / "cache"
        / "csvs"
        / "databench_benchmark.csv"
    )

    logger.info(f"Output path: {output_path}")
    logger.info(f"Question limit: {args.question_limit}")

    # Export to CSV
    try:
        export_databench_to_csv(
            output_path=output_path,
            question_limit=args.question_limit,
            qa_split=args.qa_split,
            semeval_split=args.semeval_split,
        )
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
