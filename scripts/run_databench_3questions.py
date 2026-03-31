#!/usr/bin/env python3
"""
Launch script to run DataBench experiment on 3 questions.

This script runs a DataBench experiment with DS-Star agent on 3 sample questions.
Results are saved to experiments/databench/outputs/ directory.

Usage:
    python scripts/run_databench_3questions.py

Requirements:
    - Valid API credentials in .env file
    - Internet connection for model access and HuggingFace downloads
"""

from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()


def main():
    """Run DataBench experiment on 3 questions."""
    from OpenDsStar.experiments.benchmarks.databench.databench_main import (
        DataBenchExperiment,
    )

    print("=" * 80)
    print("DataBench Experiment - 3 Questions")
    print("=" * 80)
    print()

    # Create experiment with 3 samples
    experiment = DataBenchExperiment(
        qa_split="train",
        semeval_split="train",
        model_agent="watsonx/mistralai/mistral-medium-2505",
        model_file_descriptions="watsonx/mistralai/mistral-medium-2505",
        embedding_model="ibm-granite/granite-embedding-english-r2",
        max_steps=10,  # Allow up to 10 reasoning steps
        question_limit=3,  # Run on 3 questions
        seed=43,  # For reproducibility
        code_mode="stepwise",  # Execute steps separately (efficient)
    )

    print(f"Configuration:")
    print(f"  - Agent: DS-Star")
    print(f"  - Model: watsonx/mistralai/mistral-medium-2505")
    print(f"  - Max steps: 10")
    print(f"  - Questions: 3")
    print(f"  - Code mode: stepwise")
    print()

    # Run experiment
    print("Starting experiment...")
    print("-" * 80)
    outputs, results = experiment.experiment_main(
        run_id="databench_3questions",
        fail_fast=False,  # Continue even if some questions fail
    )

    # Print summary
    print()
    print("=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print()
    print(f"Total questions processed: {len(outputs)}")
    print(f"Total evaluations: {len(results)}")
    print()

    # Calculate and display scores
    if results:
        avg_score = sum(r.score for r in results) / len(results)
        passed = sum(1 for r in results if r.passed)

        print(f"Average score: {avg_score:.3f}")
        print(f"Passed: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
        print()

        print("Individual results:")
        for i, result in enumerate(results, 1):
            status = "✓" if result.passed else "✗"
            print(f"  {status} Question {i}: score={result.score:.3f}")
    print()

    # Show where results are saved
    print("Results saved to:")
    print(f"  - Outputs: experiments/databench/outputs/databench_3questions/")
    print(f"  - Cache: experiments/databench/cache/")
    print()


if __name__ == "__main__":
    main()

# Made with Bob
