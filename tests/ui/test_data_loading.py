#!/usr/bin/env python3
"""
Test script to verify data loading works correctly without requiring streamlit.
"""

import json
from typing import Any, Dict, List


def _flatten_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the new _output.json item structure into a flat dict for the viewer.
    """
    flattened = {}

    # Extract question_id
    flattened["q_id"] = item.get("question_id", "unknown")

    # Extract output fields
    output = item.get("output", {})
    if output:
        flattened["answer"] = output.get("answer", "")

        # Extract metadata fields
        metadata = output.get("metadata", {})
        flattened["question"] = metadata.get("question", "")
        flattened["trajectory"] = metadata.get("trajectory", [])
        flattened["input_tokens"] = metadata.get("input_tokens", 0)
        flattened["output_tokens"] = metadata.get("output_tokens", 0)

        # Extract ground truths if available
        ground_truth = metadata.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            flattened["ground_truths"] = ground_truth.get("answers", [])
        else:
            flattened["ground_truths"] = []

        # Extract error analysis if available
        flattened["error_analysis_results"] = metadata.get("error_analysis_results")
    else:
        # No output available
        flattened["answer"] = ""
        flattened["question"] = ""
        flattened["trajectory"] = []
        flattened["input_tokens"] = 0
        flattened["output_tokens"] = 0
        flattened["ground_truths"] = []
        flattened["error_analysis_results"] = None

    # Extract evaluation score (use first evaluator's score)
    evaluations = item.get("evaluations", [])
    if evaluations and len(evaluations) > 0:
        first_eval = evaluations[0]
        flattened["score"] = first_eval.get("score", 0.0)
    else:
        flattened["score"] = 0.0

    return flattened


def load_output_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and flatten a single _output.json file."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)

            # Handle new _output.json format with "items" field
            if isinstance(parsed, dict) and "items" in parsed:
                for item in parsed["items"]:
                    flattened = _flatten_item(item)
                    data.append(flattened)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return data


def main():
    """Test data loading with actual output files."""
    import glob

    # Find output files
    output_files = glob.glob("**/*_output.json", recursive=True)

    if not output_files:
        print("No *_output.json files found in project")
        return

    test_file = output_files[0]
    print(f"Testing with file: {test_file}")
    print("=" * 60)

    # Load the data
    data = load_output_file(test_file)
    print(f"Loaded {len(data)} records")

    if data:
        # Check first record structure
        first = data[0]
        print(f"\nFirst record keys: {list(first.keys())}")
        print(f"Question ID: {first.get('q_id')}")
        print(f"Question: {first.get('question', '')[:100]}...")
        print(f"Answer: {first.get('answer', '')[:100]}...")
        print(f"Score: {first.get('score')}")
        print(f"Trajectory steps: {len(first.get('trajectory', []))}")
        print(f"Input tokens: {first.get('input_tokens')}")
        print(f"Output tokens: {first.get('output_tokens')}")
        print(f"Ground truths: {first.get('ground_truths', [])}")
        print("\n✓ Data loading successful!")

        # Test with all files
        print(f"\n\nTesting all {len(output_files)} output files:")
        print("=" * 60)
        for i, file_path in enumerate(output_files[:5], 1):  # Test first 5
            data = load_output_file(file_path)
            print(f"{i}. {file_path}: {len(data)} records")

        if len(output_files) > 5:
            print(f"... and {len(output_files) - 5} more files")
    else:
        print("ERROR: No data loaded")


if __name__ == "__main__":
    main()
