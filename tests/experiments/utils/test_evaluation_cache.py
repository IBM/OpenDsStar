"""Test script for evaluation cache functionality."""

import shutil
import tempfile
from pathlib import Path

from OpenDsStar.experiments.utils.evaluation_cache import EvaluationCache


class MockEvaluator:
    """Mock evaluator for testing."""

    def __init__(
        self,
        metric_id="test-metric",
        sub_scores=None,
    ):
        self.metric_id = metric_id
        self.sub_scores = sub_scores or []

    def evaluate_one(self, ctx, output, benchmark):
        """Mock evaluate method."""
        return {
            "question_id": output.question_id,
            "score": 0.8,
            "passed": True,
            "details": {"test": "result"},
        }


def test_evaluation_cache():
    """Test evaluation cache functionality."""
    print("Testing Evaluation Cache...")

    # Create a temporary directory for cache
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create evaluation cache
        evaluation_cache = EvaluationCache(cache_base_dir=temp_dir, enabled=True)

        # Create mock evaluators
        eval1 = MockEvaluator(metric_id="metric-1", sub_scores=["score1"])
        eval2 = MockEvaluator(
            metric_id="metric-1", sub_scores=["score1"]
        )  # Same config
        eval3 = MockEvaluator(
            metric_id="metric-2", sub_scores=["score2"]
        )  # Different metric

        # Test evaluator key generation
        key1 = evaluation_cache._generate_evaluator_key(eval1)
        key2 = evaluation_cache._generate_evaluator_key(eval2)
        key3 = evaluation_cache._generate_evaluator_key(eval3)

        print(f"Evaluator 1 key: {key1}")
        print(f"Evaluator 2 key: {key2}")
        print(f"Evaluator 3 key: {key3}")

        # Verify same config produces same key
        assert key1 == key2, "Same evaluator config should produce same key"
        print("✓ Same evaluator config produces same key")

        # Verify different config produces different key
        assert key1 != key3, "Different evaluator config should produce different key"
        print("✓ Different evaluator config produces different key")

        # Test full cache key generation
        question_id = "q1"
        answer = "Paris"
        ground_truth = ["Paris", "paris"]

        cache_key1 = evaluation_cache._generate_cache_key(
            eval1, question_id, answer, ground_truth
        )
        cache_key2 = evaluation_cache._generate_cache_key(
            eval2, question_id, answer, ground_truth
        )
        cache_key3 = evaluation_cache._generate_cache_key(
            eval1, question_id, "London", ground_truth
        )

        print(f"\nCache key 1: {cache_key1}")
        print(f"Cache key 2: {cache_key2}")
        print(f"Cache key 3: {cache_key3}")

        # Verify same evaluator + input produces same cache key
        assert (
            cache_key1 == cache_key2
        ), "Same evaluator + input should produce same cache key"
        print("✓ Same evaluator + input produces same cache key")

        # Verify different answer produces different cache key
        assert (
            cache_key1 != cache_key3
        ), "Different answer should produce different cache key"
        print("✓ Different answer produces different cache key")

        # Test cache put and get
        result = {
            "question_id": question_id,
            "answer_type": "llm_as_judge",
            "score": 0.9,
            "passed": True,
            "details": {"test": "cached_result"},
        }

        # Put result in cache
        evaluation_cache.put(eval1, question_id, answer, ground_truth, result)
        print("\n✓ Result stored in cache")

        # Get result from cache
        cached_result = evaluation_cache.get(eval1, question_id, answer, ground_truth)
        assert cached_result is not None, "Should retrieve cached result"
        assert cached_result["score"] == 0.9, "Cached result should match stored result"
        print("✓ Result retrieved from cache successfully")

        # Verify cache miss for different input
        cached_result2 = evaluation_cache.get(
            eval1, question_id, "London", ground_truth
        )
        assert (
            cached_result2 is None
        ), "Should not find cached result for different input"
        print("✓ Cache miss for different input")

        # Test with different evaluator (same input)
        cached_result3 = evaluation_cache.get(eval3, question_id, answer, ground_truth)
        assert (
            cached_result3 is None
        ), "Should not find cached result for different evaluator"
        print("✓ Cache miss for different evaluator")

        print("\n✅ All tests passed!")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_evaluation_cache()
