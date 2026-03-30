#!/usr/bin/env python3
"""
Evaluate all DataBench results using RagbenchLLMJudgeEvaluator.

This script:
1. Finds all *_output.json files in a directory (recursively)
2. Loads the corresponding *_params.json to recreate the experiment
3. Runs RagbenchLLMJudgeEvaluator on each output
4. Saves all evaluation results to a CSV file
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from experiments.benchmarks.databench.databench_main import DataBenchExperiment
from experiments.core.context import PipelineConfig, PipelineContext
from experiments.core.types import AgentOutput
from experiments.evaluators.ragbench_llm_judge import RagbenchLLMJudgeEvaluator
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_output_files(root_dir: Path) -> List[Path]:
    """Find all *_output.json files recursively."""
    output_files = list(root_dir.rglob("*_output.json"))
    logger.info(f"Found {len(output_files)} output files in {root_dir}")
    return output_files


def load_params_file(output_file: Path) -> Dict[str, Any] | None:
    """Load the corresponding params file for an output file."""
    params_file = output_file.parent / output_file.name.replace(
        "_output.json", "_params.json"
    )
    if not params_file.exists():
        logger.warning(f"Params file not found: {params_file}")
        return None

    with open(params_file, "r") as f:
        return json.load(f)


def load_output_file(output_file: Path) -> Dict[str, Any]:
    """Load an output JSON file."""
    with open(output_file, "r") as f:
        return json.load(f)


def recreate_experiment(params: Dict[str, Any]) -> DataBenchExperiment:
    """Recreate the experiment from params."""
    args = params.get("args", {})
    return DataBenchExperiment(**args)


def evaluate_outputs(
    output_file: Path,
    outputs_data: Dict[str, Any],
    experiment: DataBenchExperiment,
    evaluator: RagbenchLLMJudgeEvaluator,
) -> List[Dict[str, Any]]:
    """Evaluate all outputs in a file."""
    results = []

    # Create a minimal context
    config = PipelineConfig(fail_fast=False, continue_on_error=True)
    ctx = PipelineContext(config=config)

    # Get benchmark data
    data_reader = experiment.get_data_reader()
    data_reader.read_data()  # Load the data first
    benchmark_entries = data_reader.get_benchmark()  # Then get benchmark entries
    benchmark_dict = {entry.question_id: entry for entry in benchmark_entries}

    outputs = outputs_data.get("outputs", [])
    logger.info(f"Evaluating {len(outputs)} outputs from {output_file.name}")

    for output_data in tqdm(outputs, desc=f"Evaluating {output_file.name}"):
        question_id = output_data["question_id"]
        answer = output_data["answer"]
        metadata = output_data.get("metadata", {})

        # Get corresponding benchmark entry
        benchmark = benchmark_dict.get(question_id)
        if not benchmark:
            logger.warning(f"No benchmark entry found for question_id: {question_id}")
            continue

        # Create AgentOutput
        agent_output = AgentOutput(
            question_id=question_id, answer=answer, metadata=metadata
        )

        # Evaluate
        try:
            eval_result = evaluator.evaluate_one(ctx, agent_output, benchmark)

            # Collect result
            result = {
                "output_file": str(output_file),
                "question_id": question_id,
                "question": benchmark.question,
                "predicted_answer": answer,
                "ground_truth": str(benchmark.ground_truth.answers),
                "score": eval_result.score,
                "passed": eval_result.passed,
                "metric_id": eval_result.details.get("metric_id", ""),
                "raw_scores": str(eval_result.details.get("raw_instance_scores", {})),
                "error": eval_result.details.get("error", ""),
            }
            results.append(result)

        except Exception as e:
            logger.error(
                f"Error evaluating question {question_id} from {output_file.name}: {e}",
                exc_info=True,
            )
            results.append(
                {
                    "output_file": str(output_file),
                    "question_id": question_id,
                    "question": benchmark.question,
                    "predicted_answer": answer,
                    "ground_truth": str(benchmark.ground_truth.answers),
                    "score": 0.0,
                    "passed": False,
                    "metric_id": "",
                    "raw_scores": "",
                    "error": str(e),
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all DataBench results using RagbenchLLMJudgeEvaluator"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root directory containing result files",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV file path (default: evaluation_results.csv)",
    )
    parser.add_argument(
        "--metric-id",
        type=str,
        default="metrics.rag.external_rag.answer_correctness.llama_3_3_70b_instruct_watsonx_judge",
        help="Metric ID to use for evaluation",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return

    # Find all output files
    output_files = find_output_files(results_dir)
    if not output_files:
        logger.error(f"No output files found in {results_dir}")
        return

    # Create evaluator
    evaluator = RagbenchLLMJudgeEvaluator(metric_id=args.metric_id)
    logger.info(f"Using evaluator with metric: {args.metric_id}")

    # Process each output file
    all_results = []
    for output_file in output_files:
        logger.info(f"\nProcessing: {output_file}")

        # Load params
        params = load_params_file(output_file)
        if not params:
            logger.warning(f"Skipping {output_file} - no params file")
            continue

        # Load outputs
        outputs_data = load_output_file(output_file)

        # Recreate experiment to get benchmark data
        try:
            experiment = recreate_experiment(params)
        except Exception as e:
            logger.error(f"Failed to recreate experiment for {output_file}: {e}")
            continue

        # Evaluate
        results = evaluate_outputs(output_file, outputs_data, experiment, evaluator)
        all_results.extend(results)

    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = Path(args.output_csv)
        df.to_csv(output_csv, index=False)
        logger.info(f"\n✅ Saved {len(all_results)} evaluation results to {output_csv}")

        # Print summary statistics
        logger.info("\n=== Summary Statistics ===")
        logger.info(f"Total evaluations: {len(df)}")
        logger.info(f"Average score: {df['score'].mean():.3f}")
        logger.info(f"Passed rate: {df['passed'].mean():.3f}")
        logger.info(f"Files processed: {df['output_file'].nunique()}")

        # Group by output file
        logger.info("\n=== Results by File ===")
        file_stats = (
            df.groupby("output_file")
            .agg({"score": "mean", "passed": "mean", "question_id": "count"})
            .rename(columns={"question_id": "count"})
        )
        print(file_stats)

        # Save summary to file
        summary_csv = output_csv.parent / f"{output_csv.stem}_summary.csv"
        file_stats.to_csv(summary_csv)
        logger.info(f"\n✅ Saved summary statistics to {summary_csv}")
    else:
        logger.warning("No results to save")


if __name__ == "__main__":
    main()
