"""Integration test for evaluation cache in pipeline."""

import shutil
import tempfile
from pathlib import Path

from src.experiments.core.types import EvalResult
from src.experiments.utils.evaluation_cache import EvaluationCache


class MockEvaluator:
    """Mock evaluator for testing."""

    def __init__(self, metric_id="test-metric"):
        self.metric_id = metric_id
        self.call_count = 0

    def evaluate_one(self, ctx, output, benchmark):
        """Mock evaluate method that tracks calls."""
        self.call_count += 1
        return EvalResult(
            question_id=output.question_id,
            score=0.8,
            passed=True,
            details={"call_count": self.call_count},
        )


def test_evaluation_cache_reuse():
    """Test that evaluation cache properly reuses results."""
    print("Testing Evaluation Cache Reuse in Pipeline Context...")

    # Create a temporary directory for cache
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create evaluation cache
        evaluation_cache = EvaluationCache(cache_base_dir=temp_dir, enabled=True)

        # Create mock evaluator
        evaluator = MockEvaluator(metric_id="test-metric")

        # Simulate evaluation inputs
        question_id = "q1"
        answer = "Paris is the capital"
        ground_truth = ["Paris"]

        print("\n1. First evaluation (should compute)...")
        # First call - should compute
        cached = evaluation_cache.get(evaluator, question_id, answer, ground_truth)
        assert cached is None, "Cache should be empty initially"

        # Simulate evaluation
        result_dict = {
            "question_id": question_id,
            "answer_type": "llm_as_judge",
            "score": 0.85,
            "passed": True,
            "details": {"computed": True},
        }
        evaluation_cache.put(evaluator, question_id, answer, ground_truth, result_dict)
        print("   ✓ Result computed and cached")

        print("\n2. Second evaluation with same inputs (should use cache)...")
        # Second call - should use cache
        cached = evaluation_cache.get(evaluator, question_id, answer, ground_truth)
        assert cached is not None, "Cache should return result"
        assert cached["score"] == 0.85, "Cached score should match"
        assert cached["details"]["computed"] is True, "Cached details should match"
        print("   ✓ Result retrieved from cache")

        print("\n3. Third evaluation with different answer (should compute)...")
        # Different answer - should compute
        different_answer = "London is the capital"
        cached = evaluation_cache.get(
            evaluator, question_id, different_answer, ground_truth
        )
        assert cached is None, "Cache should miss for different answer"
        print("   ✓ Cache miss for different answer")

        print("\n4. Fourth evaluation with different evaluator (should compute)...")
        # Different evaluator - should compute
        evaluator2 = MockEvaluator(metric_id="different-metric")
        cached = evaluation_cache.get(evaluator2, question_id, answer, ground_truth)
        assert cached is None, "Cache should miss for different evaluator"
        print("   ✓ Cache miss for different evaluator")

        print("\n5. Verify cache persistence...")
        # Create new cache instance with same directory
        evaluation_cache2 = EvaluationCache(cache_base_dir=temp_dir, enabled=True)
        cached = evaluation_cache2.get(evaluator, question_id, answer, ground_truth)
        assert cached is not None, "Cache should persist across instances"
        assert cached["score"] == 0.85, "Cached score should persist"
        print("   ✓ Cache persists across instances")

        print("\n✅ All integration tests passed!")
        print("\nSummary:")
        print("- Evaluation cache properly stores and retrieves results")
        print("- Cache correctly distinguishes between different inputs")
        print("- Cache correctly distinguishes between different evaluators")
        print("- Cache persists across cache instances")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_evaluation_cache_reuse()
