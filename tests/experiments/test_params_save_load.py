"""Test experiment params save/load functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from OpenDsStar.experiments.benchmarks.hotpotqa.hotpotqa_main import HotpotQAExperiment
from OpenDsStar.experiments.implementations.agent_factory import AgentType


def test_experiment_params_save_format():
    """Test that experiment params are saved in Recreatable format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a simple experiment
        experiment = HotpotQAExperiment(
            split="test",
            model="watsonx/mistralai/mistral-medium-2505",
            embedding_model="ibm/slate-125m-english-rtrvr",
            max_steps=3,
            agent_type=AgentType.DS_STAR,
            temperature=0.0,
            question_limit=2,
            seed=42,
        )

        # Save config
        params_file = tmpdir_path / "test_params.json"
        experiment.save_config(params_file)

        # Load and verify structure
        with open(params_file, "r") as f:
            config = json.load(f)

        # Check Recreatable format (type + args only, no components)
        assert "type" in config
        assert "args" in config
        assert "components" not in config  # Should NOT have components anymore

        # Verify type
        assert (
            config["type"]
            == "OpenDsStar.experiments.benchmarks.hotpotqa.hotpotqa_main.HotpotQAExperiment"
        )

        # Verify args contain all constructor parameters
        args = config["args"]
        assert "split" in args
        assert "model" in args
        assert "embedding_model" in args
        assert "max_steps" in args
        assert "agent_type" in args
        assert "temperature" in args
        assert "question_limit" in args
        assert args["question_limit"] == 2


def test_experiment_params_contains_all_configs():
    """Test that all constructor parameters are captured in args."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        experiment = HotpotQAExperiment(
            split="test",
            model="watsonx/mistralai/mistral-medium-2505",
            embedding_model="ibm/slate-125m-english-rtrvr",
            max_steps=5,
            agent_type=AgentType.DS_STAR,
            temperature=0.5,
            question_limit=10,
            document_factor=2,
            seed=43,
            output_max_length=1000,
            logs_max_length=50000,
        )

        params_file = tmpdir_path / "comprehensive_params.json"
        experiment.save_config(params_file)

        with open(params_file, "r") as f:
            config = json.load(f)

        # Check all experiment constructor args are in args
        args = config["args"]
        assert args["split"] == "test"
        assert args["model"] == "watsonx/mistralai/mistral-medium-2505"
        assert args["embedding_model"] == "ibm/slate-125m-english-rtrvr"
        assert args["max_steps"] == 5
        assert args["temperature"] == 0.5
        assert args["question_limit"] == 10
        assert args["document_factor"] == 2
        assert args["seed"] == 43
        assert args["output_max_length"] == 1000
        assert args["logs_max_length"] == 50000

        # Verify agent_type is serialized correctly (as string value)
        assert args["agent_type"] == "ds_star"


def test_params_file_is_json_serializable():
    """Test that saved params file is valid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        experiment = HotpotQAExperiment(
            split="test",
            model="watsonx/mistralai/mistral-medium-2505",
            embedding_model="ibm/slate-125m-english-rtrvr",
            max_steps=3,
            agent_type=AgentType.DS_STAR,
            temperature=0.0,
            question_limit=5,
        )

        params_file = tmpdir_path / "json_test_params.json"
        experiment.save_config(params_file)

        # Should be able to load as JSON
        with open(params_file, "r") as f:
            config = json.load(f)

        # Should be able to dump back to JSON
        json_str = json.dumps(config, indent=2)
        assert len(json_str) > 0

        # Should be able to parse again
        reparsed = json.loads(json_str)
        assert reparsed == config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
