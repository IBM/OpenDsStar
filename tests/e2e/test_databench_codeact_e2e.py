"""
End-to-end tests for DataBench with CodeAct agent.

Tests that CodeAct agent can run on DataBench with different model providers.
"""

import subprocess

import pytest


@pytest.mark.e2e
@pytest.mark.slow
def test_databench_codeact_watsonx():
    """Test DataBench with CodeAct agent using WatsonX Llama Maverick."""
    # Run databench_main with codeact agent and watsonx model
    result = subprocess.run(
        [
            ".venv/bin/python",
            "-m",
            "src.experiments.benchmarks.databench.databench_main",
            "--agent-type",
            "codeact_smolagents",
            "--model-agent",
            "wx_llama_maverick",
            "--model-file-descriptions",
            "wx_llama_maverick",
            "--question-limit",
            "1",
            "--max-steps",
            "3",
        ],
        cwd="/Users/yoavkantor/projects/OpenDsStar",
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    # Check that it ran without fatal errors
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that output contains expected success indicators
    assert (
        "stage_end:run_agent" in result.stdout
        or "Experiment completed" in result.stdout
    )


@pytest.mark.e2e
@pytest.mark.slow
def test_databench_codeact_custom_api():
    """Test DataBench with CodeAct agent using custom API model."""
    # Run databench_main with codeact agent and custom API model
    result = subprocess.run(
        [
            ".venv/bin/python",
            "-m",
            "src.experiments.benchmarks.databench.databench_main",
            "--agent-type",
            "codeact_smolagents",
            "--model-agent",
            "tpm/GCP/gemini-2.5-flash",  # Uses custom API provider if configured
            "--model-file-descriptions",
            "wx_llama_maverick",
            "--question-limit",
            "1",
            "--max-steps",
            "3",
        ],
        cwd="/Users/yoavkantor/projects/OpenDsStar",
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    # Check that it ran without fatal errors
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check that output contains expected success indicators
    assert (
        "stage_end:run_agent" in result.stdout
        or "Experiment completed" in result.stdout
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
