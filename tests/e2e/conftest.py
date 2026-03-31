"""Shared helpers for e2e tests."""

from pathlib import Path

import pytest

E2E_DIR = Path(__file__).parent
E2E_CACHE_OUTPUTS_DIR = E2E_DIR / "cache_outputs"

# 60-minute timeout for all e2e tests (they call external APIs and process data)
E2E_TIMEOUT = 3600


@pytest.fixture(autouse=True)
def e2e_timeout(request):
    """Apply a generous timeout to every test in this directory."""
    # Only set if no explicit timeout marker is already present
    for marker in request.node.iter_markers("timeout"):
        return  # already has an explicit timeout
    request.node.add_marker(pytest.mark.timeout(E2E_TIMEOUT))


def redirect_experiment_dirs(experiment, benchmark_name: str):
    """Override experiment output/cache dirs to tests/e2e/cache_outputs/<benchmark>/.

    This ensures e2e test artifacts (results, cache, logs) are written
    under tests/e2e/cache_outputs/<benchmark>/ instead of polluting the benchmark
    source directories. Each benchmark gets its own subdirectory for better organization.
    The directories persist after test runs for inspection.

    Args:
        experiment: The experiment instance to configure
        benchmark_name: Name of the benchmark (e.g., "hotpotqa", "databench", "kramabench")
    """
    benchmark_dir = E2E_CACHE_OUTPUTS_DIR / benchmark_name
    output_dir = benchmark_dir / "output"
    cache_dir = benchmark_dir / "cache"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    experiment.output_dir = output_dir
    experiment.cache_dir = cache_dir
