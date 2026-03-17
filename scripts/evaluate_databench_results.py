#!/usr/bin/env python
"""
Evaluate DataBench results from a directory.

This script loads result JSON files from a directory, runs the DataBench
evaluator on them, and saves the evaluation results.

Usage:
    .venv/bin/python scripts/evaluate_databench_results.py --results-dir /path/to/results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.benchmarks.databench.data_reader import DataBenchDataReader
from experiments.core.context import PipelineConfig, PipelineContext
from experiments.core.types import AgentOutput
from experiments.evaluators.ragbench_llm_judge import RagbenchLLMJudgeEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_result_file(result_path: Path) -> Dict[str, Any]:
    """Load a result JSON file."""
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_agent_output(result: Dict[str, Any]) -> AgentOutput:
    """Create an AgentOutput from a result dictionary."""
    return AgentOutput(
        question_id=result["question_id"],
        answer=result.get("predicted_answer", ""),
        metadata={
            "execution_success": result.get("execution_success", False),
            "stderr": result.get("stderr", ""),
            "stdout": result.get("stdout", ""),
        },
    )


def evaluate_results_directory(
    results_dir: Path,
    output_dir: Path | None = None,
    qa_split: str = "train",
    semeval_split: str = "train",
) -> Dict[str, Any]:
    """
    Evaluate all result files in a directory.

    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save evaluation results (defaults to results_dir)
        qa_split: Split for QA config
        semeval_split: Split for SemEval config

    Returns:
        Dictionary with evaluation summary
    """
    if output_dir is None:
        output_dir = results_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DataBench data
    logger.info("Loading DataBench data...")
    cache_dir = (
        Path(__file__).parent.parent
        / "src"
        / "experiments"
        / "benchmarks"
        / "databench"
        / "cache"
    )

    from typing import Literal, cast

    reader = DataBenchDataReader(
        qa_split=cast(Literal["train"], qa_split),
        semeval_split=cast(Literal["train", "validation", "test"], semeval_split),
        question_limit=None,  # Load all questions
        seed=43,
        use_cache=True,
        cache_base_dir=cache_dir,
    )

    reader.read_data()
    benchmark_entries = reader.get_benchmark()

    # Create a mapping from question_id to benchmark entry
    benchmark_map = {entry.question_id: entry for entry in benchmark_entries}
    logger.info(f"Loaded {len(benchmark_map)} benchmark entries")

    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = RagbenchLLMJudgeEvaluator()

    # Find all result JSON files (excluding summary.json)
    result_files = sorted(
        [f for f in results_dir.glob("*.json") if f.name != "summary.json"]
    )

    logger.info(f"Found {len(result_files)} result files to evaluate")

    # Evaluate each result
    eval_results: List[Dict[str, Any]] = []
    # Create a minimal context for evaluation
    pipeline_config = PipelineConfig(run_id="evaluation", fail_fast=False)
    ctx = PipelineContext(config=pipeline_config)

    for i, result_file in enumerate(result_files, 1):
        logger.info(f"Evaluating {i}/{len(result_files)}: {result_file.name}")

        try:
            # Load result
            result = load_result_file(result_file)
            question_id = result["question_id"]

            # Get corresponding benchmark entry
            if question_id not in benchmark_map:
                logger.warning(f"Question ID {question_id} not found in benchmark data")
                continue

            benchmark_entry = benchmark_map[question_id]

            # Create agent output
            agent_output = create_agent_output(result)

            # Evaluate
            eval_result = evaluator.evaluate_one(ctx, agent_output, benchmark_entry)

            # Store result
            eval_results.append(
                {
                    "question_id": question_id,
                    "question": result.get("question", ""),
                    "expected_answer": result.get("expected_answer", ""),
                    "predicted_answer": result.get("predicted_answer", ""),
                    "execution_success": result.get("execution_success", False),
                    "eval_score": eval_result.score,
                    "eval_passed": eval_result.passed,
                    "eval_details": eval_result.details,
                }
            )

        except Exception as e:
            logger.error(f"Error evaluating {result_file.name}: {e}", exc_info=True)
            continue

    # Calculate summary statistics
    total_evaluated = len(eval_results)
    passed_count = sum(1 for r in eval_results if r["eval_passed"])
    avg_score = (
        sum(r["eval_score"] for r in eval_results) / total_evaluated
        if total_evaluated > 0
        else 0.0
    )

    summary = {
        "total_evaluated": total_evaluated,
        "passed_count": passed_count,
        "pass_rate": passed_count / total_evaluated if total_evaluated > 0 else 0.0,
        "average_score": avg_score,
        "results_directory": str(results_dir),
        "evaluation_timestamp": results_dir.name,
    }

    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total Evaluated: {total_evaluated}")
    logger.info(f"Passed: {passed_count} ({summary['pass_rate']*100:.1f}%)")
    logger.info(f"Average Score: {avg_score:.3f}")
    logger.info(f"{'='*80}\n")

    # Save detailed results
    detailed_output_path = output_dir / "evaluation_detailed.json"
    with open(detailed_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "results": eval_results,
            },
            f,
            indent=2,
        )
    logger.info(f"✓ Saved detailed results to {detailed_output_path}")

    # Save summary
    summary_output_path = output_dir / "evaluation_summary.json"
    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary to {summary_output_path}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate DataBench results from a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  .venv/bin/python scripts/evaluate_databench_results.py --results-dir /path/to/results
  .venv/bin/python scripts/evaluate_databench_results.py --results-dir /path/to/results --output-dir /path/to/output

Output:
  - evaluation_detailed.json: Detailed evaluation results for each question
  - evaluation_summary.json: Summary statistics
        """,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (defaults to results-dir)",
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

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Evaluate results
    try:
        evaluate_results_directory(
            results_dir=results_dir,
            output_dir=output_dir,
            qa_split=args.qa_split,
            semeval_split=args.semeval_split,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
