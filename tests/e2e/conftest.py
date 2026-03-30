"""Shared helpers for e2e tests."""

from pathlib import Path

import pytest

E2E_DIR = Path(__file__).parent
E2E_OUTPUT_DIR = E2E_DIR / "output"
E2E_CACHE_DIR = E2E_DIR / "cache"

# 60-minute timeout for all e2e tests (they call external APIs and process data)
E2E_TIMEOUT = 3600


@pytest.fixture(autouse=True)
def e2e_timeout(request):
    """Apply a generous timeout to every test in this directory."""
    # Only set if no explicit timeout marker is already present
    for marker in request.node.iter_markers("timeout"):
        return  # already has an explicit timeout
    request.node.add_marker(pytest.mark.timeout(E2E_TIMEOUT))


def redirect_experiment_dirs(experiment):
    """Override experiment output/cache dirs to tests/e2e/.

    This ensures e2e test artifacts (results, cache, logs) are written
    under tests/e2e/ instead of polluting the benchmark source directories.
    The directories persist after test runs for inspection.
    """
    E2E_OUTPUT_DIR.mkdir(exist_ok=True)
    E2E_CACHE_DIR.mkdir(exist_ok=True)
    experiment.output_dir = E2E_OUTPUT_DIR
    experiment.cache_dir = E2E_CACHE_DIR
