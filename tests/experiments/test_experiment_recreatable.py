"""Tests for experiment recreatable functionality."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from OpenDsStar.experiments.benchmarks.demo.experiment_main import DemoExperiment


def test_demo_experiment_save_and_load():
    """Test saving and loading DemoExperiment."""
    # Create experiment
    experiment = DemoExperiment(
        model="watsonx/mistralai/mistral-medium-2505",
        max_steps=3,
        temperature=0.1,
        code_timeout=20,
    )

    # Save config
    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "demo_experiment.json"
        experiment.save_config(config_path)

        # Load experiment
        loaded_experiment = DemoExperiment.load_instance(config_path)

        # Verify config matches
        assert loaded_experiment.get_config() == experiment.get_config()
        assert loaded_experiment.model == "watsonx/mistralai/mistral-medium-2505"
        assert loaded_experiment.max_steps == 3
        assert loaded_experiment.temperature == 0.1
        assert loaded_experiment.code_timeout == 20


def test_demo_experiment_has_factory_methods():
    """Test that loaded experiment has all factory methods."""
    experiment = DemoExperiment(
        model="watsonx/mistralai/mistral-medium-2505",
        max_steps=5,
    )

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "demo_experiment.json"
        experiment.save_config(config_path)

        loaded_experiment = DemoExperiment.load_instance(config_path)

        # Verify factory methods work
        data_reader = loaded_experiment.get_data_reader()
        assert data_reader is not None

        agent_builder = loaded_experiment.get_agent_builder()
        assert agent_builder is not None

        tool_builders = loaded_experiment.get_tools_builder()
        assert len(tool_builders) > 0

        evaluators = loaded_experiment.get_evaluators()
        assert len(evaluators) > 0


def test_experiment_with_security():
    """Test loading experiment with allowed types."""
    import json

    experiment = DemoExperiment(model="test-model", max_steps=5)

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "experiment.json"
        experiment.save_config(config_path)

        # Get actual type from saved config
        with open(config_path, "r") as f:
            config_data = json.load(f)
        actual_type = config_data["type"]

        # Should succeed with correct type
        loaded = DemoExperiment.load_instance(config_path, allowed_types=[actual_type])
        assert isinstance(loaded, DemoExperiment)

        # Should fail with wrong type
        wrong_types = ["some.other.Experiment"]
        with pytest.raises(ValueError, match="not in allowed_types"):
            DemoExperiment.load_instance(config_path, allowed_types=wrong_types)


def test_experiment_config_with_defaults():
    """Test that default values are captured correctly."""
    # Create experiment with only required args
    experiment = DemoExperiment(model="test-model")

    config = experiment.get_config()

    # Verify defaults are captured
    assert config["model"] == "test-model"
    assert config["max_steps"] == 5  # default
    assert config["temperature"] == 0.0  # default
    assert config["code_timeout"] == 30  # default


def test_multiple_experiments_different_configs():
    """Test that multiple experiments can have different configs."""
    exp1 = DemoExperiment(model="model1", max_steps=3, temperature=0.1)
    exp2 = DemoExperiment(model="model2", max_steps=5, temperature=0.5)

    config1 = exp1.get_config()
    config2 = exp2.get_config()

    assert config1 != config2
    assert config1["model"] == "model1"
    assert config2["model"] == "model2"
    assert config1["max_steps"] == 3
    assert config2["max_steps"] == 5
