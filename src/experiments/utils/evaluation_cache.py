"""Evaluation cache utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .cache import FileCache


class EvaluationCache:
    """
    Cache for evaluator results.

    Caches evaluator.evaluate_one() results based on:
    - Evaluator type and configuration (metric_id, sub_scores, etc.)
    - Agent output (question_id and answer)
    - Ground truth (answers)

    The cache is stored in a directory structure:
    <cache_base_dir>/evaluation_cache_<evaluator_hash>/

    where evaluator_hash is computed from the evaluator's configuration.
    This allows reusing expensive evaluation calls (especially LLM-as-judge)
    when running experiments with the same evaluator configuration on the same outputs.
    """

    def __init__(
        self, cache_base_dir: Path | None = None, enabled: bool = True
    ) -> None:
        """
        Initialize the evaluation cache.

        Args:
            cache_base_dir: Base directory for cache storage (e.g., experiment output dir)
            enabled: Whether caching is enabled (default: True)
        """
        self._cache_base_dir = Path(cache_base_dir) if cache_base_dir else None
        self._enabled = enabled
        self._evaluator_cache = None  # Will be initialized when evaluator is first seen
        self._current_evaluator_hash = None

    def _generate_evaluator_key(self, evaluator: Any) -> str:
        """
        Generate a cache key for an evaluator based on its configuration.

        Args:
            evaluator: The evaluator instance

        Returns:
            String key representing the evaluator's configuration
        """
        # Extract evaluator configuration attributes
        config_dict = {}

        # Common attributes to include in cache key
        attrs_to_check = [
            "metric_id",  # Metric identifier (for Unitxt evaluators)
            "sub_scores",  # Sub-scores configuration
            "answer_type",  # Answer type handled by evaluator
            "model_id",  # Model identifier (if evaluator uses LLM)
            "temperature",  # Generation temperature (if applicable)
        ]

        for attr in attrs_to_check:
            if hasattr(evaluator, attr):
                value = getattr(evaluator, attr)
                # Convert to string for serialization
                if hasattr(value, "value"):  # Handle enums
                    config_dict[attr] = value.value
                elif isinstance(value, list):
                    config_dict[attr] = sorted([str(v) for v in value])
                else:
                    config_dict[attr] = str(value)

        # Get evaluator class name
        config_dict["evaluator_class"] = evaluator.__class__.__name__

        # Serialize to JSON for consistent hashing
        config_json = json.dumps(config_dict, sort_keys=True)

        # Generate hash
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def _generate_cache_key(
        self,
        evaluator: Any,
        question_id: str,
        answer: str,
        ground_truth_answers: list[str],
    ) -> str:
        """
        Generate a cache key for an evaluation.

        Args:
            evaluator: The evaluator instance
            question_id: The question identifier
            answer: The agent's answer
            ground_truth_answers: List of ground truth answers

        Returns:
            Cache key string
        """
        evaluator_key = self._generate_evaluator_key(evaluator)

        # Create input hash from question_id, answer, and ground truth
        input_dict = {
            "question_id": question_id,
            "answer": answer,
            "ground_truth": sorted(ground_truth_answers),  # Sort for consistency
        }
        input_json = json.dumps(input_dict, sort_keys=True)
        input_hash = hashlib.sha256(input_json.encode()).hexdigest()[:16]

        return f"eval:{evaluator_key}:{input_hash}"

    def _ensure_evaluator_cache(self, evaluator: Any) -> None:
        """
        Ensure the cache is initialized for the given evaluator.
        Creates a cache directory based on evaluator configuration hash.

        Args:
            evaluator: The evaluator instance
        """
        if not self._enabled or self._cache_base_dir is None:
            return

        evaluator_hash = self._generate_evaluator_key(evaluator)

        # If evaluator changed, create new cache
        if self._current_evaluator_hash != evaluator_hash:
            cache_dir = self._cache_base_dir / f"evaluation_cache_{evaluator_hash}"
            self._evaluator_cache = FileCache(cache_dir)
            self._current_evaluator_hash = evaluator_hash

    def get(
        self,
        evaluator: Any,
        question_id: str,
        answer: str,
        ground_truth_answers: list[str],
    ) -> dict[str, Any] | None:
        """
        Get cached result for an evaluation.

        Args:
            evaluator: The evaluator instance
            question_id: The question identifier
            answer: The agent's answer
            ground_truth_answers: List of ground truth answers

        Returns:
            Cached EvalResult dict or None if not found or caching disabled
        """
        if not self._enabled or self._cache_base_dir is None:
            return None

        self._ensure_evaluator_cache(evaluator)

        if self._evaluator_cache is None:
            return None

        cache_key = self._generate_cache_key(
            evaluator, question_id, answer, ground_truth_answers
        )
        return self._evaluator_cache.get(cache_key)

    def put(
        self,
        evaluator: Any,
        question_id: str,
        answer: str,
        ground_truth_answers: list[str],
        result: dict[str, Any],
    ) -> None:
        """
        Cache a result for an evaluation.

        Args:
            evaluator: The evaluator instance
            question_id: The question identifier
            answer: The agent's answer
            ground_truth_answers: List of ground truth answers
            result: The EvalResult dict from evaluator.evaluate_one()
        """
        if not self._enabled or self._cache_base_dir is None:
            return

        self._ensure_evaluator_cache(evaluator)

        if self._evaluator_cache is None:
            return

        cache_key = self._generate_cache_key(
            evaluator, question_id, answer, ground_truth_answers
        )
        self._evaluator_cache.put(cache_key, result)

    def enable(self) -> None:
        """Enable caching."""
        self._enabled = True

    def disable(self) -> None:
        """Disable caching."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled
