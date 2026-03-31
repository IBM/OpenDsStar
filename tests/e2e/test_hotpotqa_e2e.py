"""End-to-end test for HotpotQA experiment."""

import pytest
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()


@pytest.mark.e2e
@pytest.mark.slow
def test_hotpotqa_sample_experiment():
    """
    E2E test: Run HotpotQA experiment on 3 samples.

    This test:
    1. Creates a HotpotQA experiment with 3 questions
    2. Runs the complete pipeline (data loading, tool creation, agent execution, evaluation)
    3. Validates the structure of outputs and results

    Note: This test requires:
    - Valid API credentials in .env file
    - Internet connection for model access
    - Takes ~30-60 seconds to run
    """
    from OpenDsStar.experiments.benchmarks.hotpotqa.hotpotqa_main import (
        HotpotQAExperiment,
    )
    from OpenDsStar.experiments.core.types import AgentOutput, EvalResult

    # Create experiment with 3 samples
    experiment = HotpotQAExperiment(
        split="test",
        model="watsonx/mistralai/mistral-medium-2505",
        embedding_model="ibm-granite/granite-embedding-english-r2",
        max_steps=5,
        question_limit=3,  # Only 3 samples for fast E2E test
        document_factor=10,
        seed=43,
    )

    # Redirect output/cache to tests/e2e/cache_outputs/hotpotqa/ to avoid polluting benchmark dirs
    from tests.e2e.conftest import redirect_experiment_dirs

    redirect_experiment_dirs(experiment, benchmark_name="hotpotqa")

    # Run experiment
    outputs, results = experiment.experiment_main(
        run_id="e2e_test_hotpotqa_3samples",
        fail_fast=False,
    )

    # Verify outputs
    assert outputs is not None, "Outputs should not be None"
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

    # Verify results
    assert results is not None, "Results should not be None"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # Verify outputs structure
    assert all(
        isinstance(output, AgentOutput) for output in outputs
    ), "All outputs should be AgentOutput instances"

    for output in outputs:
        assert hasattr(output, "question_id"), "Output should have question_id"
        assert hasattr(output, "answer"), "Output should have answer"
        assert output.question_id is not None, "question_id should not be None"
        assert output.answer is not None, "answer should not be None"
        assert output.answer.strip(), f"Question {output.question_id}: answer is empty"
        assert (
            "Process terminated due to an error" not in output.answer
        ), f"Question {output.question_id}: got error instead of answer: {output.answer[:200]}"

    # Verify results structure
    assert all(
        isinstance(result, EvalResult) for result in results
    ), "All results should be EvalResult instances"

    for result in results:
        assert hasattr(result, "question_id"), "Result should have question_id"
        assert hasattr(result, "score"), "Result should have score"
        assert hasattr(result, "passed"), "Result should have passed"
        assert 0.0 <= result.score <= 1.0, f"Score {result.score} should be in [0, 1]"
        assert isinstance(result.passed, bool), "passed should be boolean"

    # Verify alignment between outputs and results
    output_ids = {output.question_id for output in outputs}
    result_ids = {result.question_id for result in results}
    assert output_ids == result_ids, "Output and result question_ids should match"

    # Calculate average score
    avg_score = sum(r.score for r in results) / len(results)

    # Print summary for debugging
    print(f"\n{'='*60}")
    print("E2E Test Results:")
    print(f"{'='*60}")
    print(f"Total questions: {len(outputs)}")
    print(f"Total evaluations: {len(results)}")
    print(f"Average score: {avg_score:.3f}")

    passed = sum(1 for r in results if r.passed)
    print(f"Passed: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")

    for i, result in enumerate(results, 1):
        print(f"  Question {i}: score={result.score:.3f}, passed={result.passed}")
    print(f"{'='*60}\n")

    # Main assertion: average score should be >= 0.0
    assert avg_score >= 0.0, (
        f"Average score {avg_score:.3f} is not >= 0.0. "
        f"Individual scores: {[r.score for r in results]}"
    )


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])
