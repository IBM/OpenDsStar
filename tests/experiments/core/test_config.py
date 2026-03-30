"""Tests for configuration classes."""

from pathlib import Path

from OpenDsStar.experiments.core.config import AgentConfig, ExperimentConfig


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_to_dict_merges_extra_params(self):
        """Test that to_dict merges extra_params into main dict."""
        config = AgentConfig(
            model="gpt-4",
            temperature=0.5,
            extra_params={"custom": "param", "another": 123},
        )
        result = config.to_dict()

        # Standard fields present
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.5
        # Extra params merged at top level
        assert result["custom"] == "param"
        assert result["another"] == 123
        # extra_params key should not be in result
        assert "extra_params" not in result

    def test_extra_params_default_factory(self):
        """Test that extra_params uses default factory to prevent shared mutable defaults."""
        config1 = AgentConfig()
        config2 = AgentConfig()

        # Should be different dict instances
        assert config1.extra_params is not config2.extra_params

        config1.extra_params["test"] = "value"
        assert "test" not in config2.extra_params


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""

    def test_post_init_fail_fast_overrides_continue_on_error(self):
        """Test that fail_fast=True forces continue_on_error=False (business rule)."""
        config = ExperimentConfig(run_id="test", fail_fast=True, continue_on_error=True)
        # __post_init__ should override continue_on_error
        assert config.fail_fast is True
        assert config.continue_on_error is False

    def test_post_init_converts_string_paths(self):
        """Test that string paths are converted to Path objects."""
        config = ExperimentConfig(
            run_id="test",
            output_dir="/tmp/output",  # type: ignore
            cache_dir="/tmp/cache",  # type: ignore
        )

        assert isinstance(config.output_dir, Path)
        assert isinstance(config.cache_dir, Path)
        assert str(config.output_dir) == "/tmp/output"
        assert str(config.cache_dir) == "/tmp/cache"

    def test_to_dict_converts_paths_to_strings(self, tmp_path):
        """Test to_dict converts Path objects to strings for serialization."""
        config = ExperimentConfig(
            run_id="test",
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "cache",
        )
        result = config.to_dict()

        assert isinstance(result["output_dir"], str)
        assert isinstance(result["cache_dir"], str)
        assert result["output_dir"] == str(tmp_path / "output")
        assert result["cache_dir"] == str(tmp_path / "cache")

    def test_to_dict_includes_nested_agent_config(self):
        """Test to_dict recursively converts nested AgentConfig."""
        agent_config = AgentConfig(model="gpt-4", temperature=0.5)
        config = ExperimentConfig(run_id="test", agent_config=agent_config)
        result = config.to_dict()

        assert result["agent_config"] is not None
        assert isinstance(result["agent_config"], dict)
        assert result["agent_config"]["model"] == "gpt-4"
        assert result["agent_config"]["temperature"] == 0.5


class TestConfigIntegration:
    """Test integration between config classes."""

    def test_to_dict_preserves_nested_structure(self):
        """Test that to_dict preserves and converts nested config structure."""
        agent_config = AgentConfig(
            model="claude-3", temperature=0.5, extra_params={"custom": "value"}
        )
        exp_config = ExperimentConfig(run_id="test", agent_config=agent_config)

        result = exp_config.to_dict()

        assert "agent_config" in result
        assert result["agent_config"]["model"] == "claude-3"
        assert result["agent_config"]["temperature"] == 0.5
        # Extra params should be merged in nested config
        assert result["agent_config"]["custom"] == "value"

    def test_configs_are_independent(self):
        """Test that config instances don't share mutable state."""
        config1 = AgentConfig(model="gpt-4")
        config2 = AgentConfig(model="gpt-4")

        config1.extra_params["test"] = "value"

        assert "test" not in config2.extra_params
        assert config1.model == config2.model
