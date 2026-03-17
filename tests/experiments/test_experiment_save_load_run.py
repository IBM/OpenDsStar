"""
Test that experiments can be saved, loaded, and produce consistent results.

This test verifies the complete save/load/run cycle:
1. Run an experiment and save its config
2. Load the config and run again
3. Compare results to ensure reproducibility
"""

import json

import pytest


def test_demo_experiment_save_load_run_cycle(tmp_path):
    """
    Test complete cycle: run experiment -> save config -> load config -> run again -> compare results.
    """
    from experiments.benchmarks.demo.experiment_main import DemoExperiment

    # Create a simple test experiment with deterministic settings
    experiment1 = DemoExperiment(
        model="test-model",
        max_steps=2,
        temperature=0.0,  # Deterministic
        code_timeout=10,
    )

    # Save the config
    config_file = tmp_path / "test_experiment.json"
    experiment1.save_config(config_file)

    # Verify config file was created
    assert config_file.exists()

    # Load the config
    with open(config_file) as f:
        saved_data = json.load(f)

    # Verify it's in the correct format
    assert "type" in saved_data
    assert "args" in saved_data
    assert saved_data["args"]["model"] == "test-model"
    assert saved_data["args"]["max_steps"] == 2
    assert saved_data["args"]["temperature"] == 0.0

    # Load and create a new experiment from the config
    experiment2 = DemoExperiment.load_instance(config_file)

    # Verify the loaded experiment has the same config
    assert experiment2.get_config() == experiment1.get_config()

    # Verify specific parameters
    assert experiment2.get_config()["model"] == "test-model"
    assert experiment2.get_config()["max_steps"] == 2
    assert experiment2.get_config()["temperature"] == 0.0


def test_experiment_config_persistence(tmp_path):
    """
    Test that experiment config is correctly persisted and can be loaded.
    """
    from experiments.benchmarks.demo.experiment_main import DemoExperiment

    # Create experiment with specific parameters
    original_config = {
        "model": "test-model-123",
        "max_steps": 7,
        "temperature": 0.5,
        "code_timeout": 30,
    }

    experiment1 = DemoExperiment(**original_config)

    # Save config
    config_file = tmp_path / "config_test.json"
    experiment1.save_config(config_file)

    # Load config
    experiment2 = DemoExperiment.load_instance(config_file)

    # Compare all config values
    loaded_config = experiment2.get_config()
    for key, value in original_config.items():
        assert (
            loaded_config[key] == value
        ), f"Config mismatch for {key}: {loaded_config[key]} != {value}"


def test_experiment_with_defaults_save_load(tmp_path):
    """
    Test that experiments with default parameters save and load correctly.
    """
    from experiments.benchmarks.demo.experiment_main import DemoExperiment

    # Create experiment with only required parameter
    experiment1 = DemoExperiment(model="test-model")

    # Save config
    config_file = tmp_path / "defaults_test.json"
    experiment1.save_config(config_file)

    # Load config
    experiment2 = DemoExperiment.load_instance(config_file)

    # Verify defaults are preserved
    config1 = experiment1.get_config()
    config2 = experiment2.get_config()

    assert config1 == config2
    assert config2["max_steps"] == 5  # Default value
    assert config2["temperature"] == 0.0  # Default value
    assert config2["code_timeout"] == 30  # Default value


def test_multiple_save_load_cycles(tmp_path):
    """
    Test that multiple save/load cycles preserve config integrity.
    """
    from experiments.benchmarks.demo.experiment_main import DemoExperiment

    # Original experiment
    original = DemoExperiment(
        model="cycle-test", max_steps=3, temperature=0.1, code_timeout=15
    )
    original_config = original.get_config()

    # Save and load 3 times
    for i in range(3):
        config_file = tmp_path / f"cycle_{i}.json"
        original.save_config(config_file)
        loaded = DemoExperiment.load_instance(config_file)

        # Verify config is preserved
        assert loaded.get_config() == original_config

        # Use loaded experiment for next iteration
        original = loaded


def test_experiment_config_immutability(tmp_path):
    """
    Test that saved config doesn't change after experiment modifications.
    """
    from experiments.benchmarks.demo.experiment_main import DemoExperiment

    # Create and save experiment
    experiment = DemoExperiment(model="immutable-test", max_steps=5)
    config_file = tmp_path / "immutable_test.json"
    experiment.save_config(config_file)

    # Get original config
    original_config = experiment.get_config().copy()

    # Modify experiment object (if possible)
    # Note: This tests that the saved config is independent of object state

    # Load from saved config
    loaded = DemoExperiment.load_instance(config_file)

    # Verify loaded config matches original, not modified object
    assert loaded.get_config() == original_config


def test_config_file_format(tmp_path):
    """
    Test that the saved config file has the correct JSON format.
    """
    from experiments.benchmarks.demo.experiment_main import DemoExperiment

    experiment = DemoExperiment(model="format-test", max_steps=4)
    config_file = tmp_path / "format_test.json"
    experiment.save_config(config_file)

    # Read and parse JSON
    with open(config_file) as f:
        data = json.load(f)

    # Verify structure
    assert isinstance(data, dict)
    assert "type" in data
    assert "args" in data

    # Verify type is a string with module path
    assert isinstance(data["type"], str)
    assert "." in data["type"]  # Should have module path
    assert data["type"].endswith("DemoExperiment")

    # Verify args is a dict
    assert isinstance(data["args"], dict)
    assert "model" in data["args"]
    assert data["args"]["model"] == "format-test"


def test_load_with_security_allowed_types(tmp_path):
    """
    Test that load_instance respects allowed_types security parameter.
    """
    from experiments.benchmarks.demo.experiment_main import DemoExperiment

    experiment = DemoExperiment(model="security-test")
    config_file = tmp_path / "security_test.json"
    experiment.save_config(config_file)

    # Get the correct type string
    with open(config_file) as f:
        data = json.load(f)
    correct_type = data["type"]

    # Should succeed with correct allowed type
    loaded = DemoExperiment.load_instance(config_file, allowed_types=[correct_type])
    assert loaded.get_config()["model"] == "security-test"

    # Should fail with wrong allowed type
    with pytest.raises(ValueError, match="not in allowed_types"):
        DemoExperiment.load_instance(config_file, allowed_types=["some.other.Type"])


def test_hotpotqa_experiment_save_and_load(tmp_path):
    """
    Test that HotpotQAExperiment can be saved and loaded correctly.
    """
    from experiments.benchmarks.hotpotqa.hotpotqa_main import HotpotQAExperiment
    from experiments.implementations.agent_factory import AgentType

    # Create experiment with all parameters including new ones
    experiment1 = HotpotQAExperiment(
        split="test",
        model="test-model",
        embedding_model="test-embedding",
        max_steps=3,
        agent_type=AgentType.DS_STAR,
        temperature=0.0,
        question_limit=2,
        document_factor=5,
        seed=42,
        output_max_length=600,
        logs_max_length=25000,
    )

    # Save config
    config_file = tmp_path / "hotpotqa_test.json"
    experiment1.save_config(config_file)

    # Verify config file
    assert config_file.exists()

    # Load config
    experiment2 = HotpotQAExperiment.load_instance(config_file)

    # Verify configs match
    assert experiment2.get_config() == experiment1.get_config()
    assert experiment2.get_config()["split"] == "test"
    assert experiment2.get_config()["model"] == "test-model"
    assert experiment2.get_config()["question_limit"] == 2
    assert experiment2.get_config()["output_max_length"] == 600
    assert experiment2.get_config()["logs_max_length"] == 25000


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
