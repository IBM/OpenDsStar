#!/usr/bin/env python3
"""
Analyze experiment results from a directory structure.

Given a directory containing subdirectories with *_output.json files,
this script extracts and aggregates:
1. Final score per metric
2. Sum of input tokens
3. Sum of output tokens
4. Sum of number of LLM calls

Creates a summary table for each subdirectory.

Also writes a CSV summary (scores columns on the right).
Rows are sorted by subdir_name "backwords" (i.e., by reversed string).
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_output_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse an output JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return {}


def extract_tokens_from_trajectory(trajectory: Any) -> tuple[int, int]:
    """
    Extract token usage from trajectory as fallback.

    Tries to extract tokens from trajectory steps when metadata doesn't have them.
    Supports multiple formats used by different agent types.

    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    total_input = 0
    total_output = 0

    if not trajectory:
        return 0, 0

    if isinstance(trajectory, list):
        for step in trajectory:
            if isinstance(step, dict):
                usage = step.get("usage") or step.get("token_usage")
                if isinstance(usage, dict):
                    total_input += usage.get("prompt_tokens", 0) or usage.get(
                        "input_tokens", 0
                    )
                    total_output += usage.get("completion_tokens", 0) or usage.get(
                        "output_tokens", 0
                    )

                llm_output = step.get("llm_output")
                if isinstance(llm_output, dict):
                    token_usage = llm_output.get("token_usage", {})
                    total_input += token_usage.get("prompt_tokens", 0)
                    total_output += token_usage.get("completion_tokens", 0)

    return total_input, total_output


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metrics from output data.

    Returns:
        Dictionary with:
        - scores_by_evaluator: Dict[evaluator_name, List[scores]]
        - total_input_tokens: int
        - total_output_tokens: int
        - total_llm_calls: int
        - num_questions: int
    """
    metrics = {
        "scores_by_evaluator": defaultdict(list),
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_llm_calls": 0,
        "num_questions": 0,
    }

    items = data.get("items", [])
    if items:
        for item in items:
            evaluations = item.get("evaluations", [])
            for eval_result in evaluations:
                evaluator_name = eval_result.get("details", {}).get(
                    "evaluator_class", "unknown"
                )
                score = eval_result.get("score", 0.0)
                metrics["scores_by_evaluator"][evaluator_name].append(score)

            output = item.get("output")
            if output:
                metadata = output.get("metadata", {})
                input_tokens = metadata.get("input_tokens", 0)
                output_tokens = metadata.get("output_tokens", 0)

                if input_tokens == 0 and output_tokens == 0:
                    trajectory = metadata.get("trajectory")
                    if trajectory:
                        input_tokens, output_tokens = extract_tokens_from_trajectory(
                            trajectory
                        )

                metrics["total_input_tokens"] += input_tokens
                metrics["total_output_tokens"] += output_tokens
                metrics["total_llm_calls"] += metadata.get("num_llm_calls", 0)
                metrics["num_questions"] += 1

    elif "outputs" in data and "results" in data:
        outputs = data.get("outputs", [])
        results = data.get("results", [])

        for result in results:
            evaluator_name = result.get("details", {}).get("evaluator_class", "unknown")
            score = result.get("score", 0.0)
            metrics["scores_by_evaluator"][evaluator_name].append(score)

        for output in outputs:
            metadata = output.get("metadata", {})
            input_tokens = metadata.get("input_tokens", 0)
            output_tokens = metadata.get("output_tokens", 0)

            if input_tokens == 0 and output_tokens == 0:
                trajectory = metadata.get("trajectory")
                if trajectory:
                    input_tokens, output_tokens = extract_tokens_from_trajectory(
                        trajectory
                    )

            metrics["total_input_tokens"] += input_tokens
            metrics["total_output_tokens"] += output_tokens
            metrics["total_llm_calls"] += metadata.get("num_llm_calls", 0)
            metrics["num_questions"] += 1

    return metrics


def compute_average_scores(
    scores_by_evaluator: Dict[str, List[float]],
) -> Dict[str, float]:
    """Compute average score for each evaluator."""
    avg_scores: Dict[str, float] = {}
    for evaluator, scores in scores_by_evaluator.items():
        avg_scores[evaluator] = (sum(scores) / len(scores)) if scores else 0.0
    return avg_scores


def analyze_subdirectory(subdir: Path) -> Dict[str, Any] | None:
    """
    Analyze all *_output.json files in a subdirectory.

    Returns:
        Dictionary with aggregated metrics, or None if no output files found
    """
    output_files = list(subdir.glob("*_output.json"))
    if not output_files:
        return None

    all_scores_by_evaluator = defaultdict(list)
    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0
    total_questions = 0

    for output_file in output_files:
        data = load_output_file(output_file)
        if not data:
            continue

        metrics = extract_metrics(data)

        for evaluator, scores in metrics["scores_by_evaluator"].items():
            all_scores_by_evaluator[evaluator].extend(scores)

        total_input_tokens += metrics["total_input_tokens"]
        total_output_tokens += metrics["total_output_tokens"]
        total_llm_calls += metrics["total_llm_calls"]
        total_questions += metrics["num_questions"]

    avg_scores = compute_average_scores(all_scores_by_evaluator)

    return {
        "subdir_name": subdir.name,
        "num_files": len(output_files),
        "num_questions": total_questions,
        "avg_scores": avg_scores,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_llm_calls": total_llm_calls,
    }


def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print results in a formatted table."""
    if not results:
        print("No results to display.")
        return

    all_evaluators = set()
    for result in results:
        all_evaluators.update(result["avg_scores"].keys())
    all_evaluators = sorted(all_evaluators)

    print("\n" + "=" * 120)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 120)

    header_parts = [
        "Subdirectory".ljust(30),
        "Files".rjust(6),
        "Questions".rjust(10),
    ]

    for evaluator in all_evaluators:
        short_name = evaluator.split(".")[-1] if "." in evaluator else evaluator
        header_parts.append(short_name[:15].rjust(15))

    header_parts.extend(
        [
            "Input Tokens".rjust(15),
            "Output Tokens".rjust(15),
            "Total Tokens".rjust(15),
            "LLM Calls".rjust(12),
        ]
    )

    print(" | ".join(header_parts))
    print("-" * 120)

    for result in results:
        row_parts = [
            result["subdir_name"][:30].ljust(30),
            str(result["num_files"]).rjust(6),
            str(result["num_questions"]).rjust(10),
        ]

        for evaluator in all_evaluators:
            score = result["avg_scores"].get(evaluator, 0.0)
            row_parts.append(f"{score:.4f}".rjust(15))

        row_parts.extend(
            [
                f"{result['total_input_tokens']:,}".rjust(15),
                f"{result['total_output_tokens']:,}".rjust(15),
                f"{result['total_tokens']:,}".rjust(15),
                str(result["total_llm_calls"]).rjust(12),
            ]
        )

        print(" | ".join(row_parts))

    print("=" * 120)

    print("\nSUMMARY STATISTICS:")
    print("-" * 60)

    total_questions = sum(r["num_questions"] for r in results)
    total_input_tokens = sum(r["total_input_tokens"] for r in results)
    total_output_tokens = sum(r["total_output_tokens"] for r in results)
    total_llm_calls = sum(r["total_llm_calls"] for r in results)

    print(f"Total Questions:     {total_questions:,}")
    print(f"Total Input Tokens:  {total_input_tokens:,}")
    print(f"Total Output Tokens: {total_output_tokens:,}")
    print(f"Total Tokens:        {total_input_tokens + total_output_tokens:,}")
    print(f"Total LLM Calls:     {total_llm_calls:,}")

    if all_evaluators:
        print("\nAverage Scores Across All Subdirectories:")
        for evaluator in all_evaluators:
            scores = [
                r["avg_scores"].get(evaluator, 0.0)
                for r in results
                if evaluator in r["avg_scores"]
            ]
            if scores:
                avg_score = sum(scores) / len(scores)
                short_name = evaluator.split(".")[-1] if "." in evaluator else evaluator
                print(f"  {short_name}: {avg_score:.4f}")

    print("=" * 120)


def write_results_csv(results: List[Dict[str, Any]], output_csv: Path) -> None:
    """Write results to a CSV file. Scores columns are on the right."""
    if not results:
        print("No results to write to CSV.", file=sys.stderr)
        return

    all_evaluators = set()
    for result in results:
        all_evaluators.update(result["avg_scores"].keys())
    all_evaluators = sorted(all_evaluators)

    fieldnames = [
        "subdir_name",
        "num_files",
        "num_questions",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "total_llm_calls",
    ] + all_evaluators

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row: Dict[str, Any] = {
                "subdir_name": r["subdir_name"],
                "num_files": r["num_files"],
                "num_questions": r["num_questions"],
                "total_input_tokens": r["total_input_tokens"],
                "total_output_tokens": r["total_output_tokens"],
                "total_tokens": r["total_tokens"],
                "total_llm_calls": r["total_llm_calls"],
            }
            for evaluator in all_evaluators:
                row[evaluator] = r["avg_scores"].get(evaluator, 0.0)
            writer.writerow(row)

    print(f"\nWrote CSV summary to: {output_csv}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_experiment_results.py <directory_path>")
        print("\nExample:")
        print(
            "  python analyze_experiment_results.py /path/to/results/kramabench/run_v1"
        )
        sys.exit(1)

    root_dir = Path(sys.argv[1])

    if not root_dir.exists():
        print(f"Error: Directory does not exist: {root_dir}", file=sys.stderr)
        sys.exit(1)

    if not root_dir.is_dir():
        print(f"Error: Path is not a directory: {root_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing results in: {root_dir}")

    subdirs = [d for d in root_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"No subdirectories found in {root_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(subdirs)} subdirectories")

    results: List[Dict[str, Any]] = []

    # Sort subdirectories by *reversed* name ("backwords")
    for subdir in sorted(subdirs, key=lambda p: p.name[::-1], reverse=True):
        print(f"  Analyzing {subdir.name}...", end=" ")
        result = analyze_subdirectory(subdir)
        if result:
            results.append(result)
            print(
                f"✓ ({result['num_files']} files, {result['num_questions']} questions)"
            )
        else:
            print("✗ (no output files found)")

    if not results:
        print("\nNo results found in any subdirectory.", file=sys.stderr)
        sys.exit(1)

    # Ensure rows are also sorted by reversed subdir_name (defensive)
    results.sort(key=lambda r: r["subdir_name"][::-1], reverse=True)

    print_results_table(results)

    output_csv = root_dir / "experiment_results_summary.csv"
    write_results_csv(results, output_csv)


if __name__ == "__main__":
    main()
