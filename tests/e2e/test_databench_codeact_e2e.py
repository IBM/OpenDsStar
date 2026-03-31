"""
End-to-end tests for DataBench with CodeAct agent.

Tests that CodeAct agent can run on DataBench with different model providers.
"""

import os

import pytest
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

_custom_api_configured = bool(
    os.environ.get("CUSTOM_API_BASE", "").strip()
    and os.environ.get("CUSTOM_API_KEY", "").strip()
)


@pytest.mark.e2e
@pytest.mark.slow
def test_databench_codeact_watsonx():
    """
    E2E test: Run DataBench experiment with CodeAct agent on 1 sample.

    This test validates that CodeAct agent can run on DataBench with WatsonX model.
    """
    from OpenDsStar.experiments.benchmarks.databench.databench_main import (
        DataBenchExperiment,
    )
    from OpenDsStar.experiments.implementations.agent_factory import AgentType

    # Create experiment with 1 sample
    experiment = DataBenchExperiment(
        qa_split="train",
        semeval_split="train",
        model_agent="watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        model_file_descriptions="watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        embedding_model="ibm-granite/granite-embedding-english-r2",
        max_steps=3,
        agent_type=AgentType.CODEACT_SMOLAGENTS,
        question_limit=1,
        seed=43,
    )

    # Redirect output/cache to tests/e2e/cache_outputs/databench/ to avoid polluting benchmark dirs
    from tests.e2e.conftest import redirect_experiment_dirs

    redirect_experiment_dirs(experiment, benchmark_name="databench")

    # Run experiment
    outputs, results = experiment.experiment_main(
        run_id="e2e_test_databench_codeact_watsonx",
        fail_fast=False,
    )

    # Verify outputs
    assert outputs is not None, "Outputs should not be None"
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"

    # Verify results
    assert results is not None, "Results should not be None"
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.skipif(
    not _custom_api_configured,
    reason="Custom API not configured (CUSTOM_API_BASE and CUSTOM_API_KEY required)",
)
def test_databench_codeact_custom_api():
    """
    E2E test: Run DataBench experiment with CodeAct agent using custom API model.

    This test validates that CodeAct agent can run on DataBench with custom API provider.
    """
    from OpenDsStar.experiments.benchmarks.databench.databench_main import (
        DataBenchExperiment,
    )
    from OpenDsStar.experiments.implementations.agent_factory import AgentType

    # Create experiment with 1 sample
    experiment = DataBenchExperiment(
        qa_split="train",
        semeval_split="train",
        model_agent="tpm/GCP/gemini-2.5-flash",  # Uses custom API provider if configured
        model_file_descriptions="watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        embedding_model="ibm-granite/granite-embedding-english-r2",
        max_steps=3,
        agent_type=AgentType.CODEACT_SMOLAGENTS,
        question_limit=1,
        seed=43,
    )

    # Redirect output/cache to tests/e2e/cache_outputs/databench/ to avoid polluting benchmark dirs
    from tests.e2e.conftest import redirect_experiment_dirs

    redirect_experiment_dirs(experiment, benchmark_name="databench")

    # Run experiment
    outputs, results = experiment.experiment_main(
        run_id="e2e_test_databench_codeact_custom_api",
        fail_fast=False,
    )

    # Verify outputs
    assert outputs is not None, "Outputs should not be None"
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"

    # Verify results
    assert results is not None, "Results should not be None"
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
