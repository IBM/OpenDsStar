"""
Configuration classes for experiments and agents.

This module provides a clear separation between agent configuration
and experiment configuration, improving modularity and maintainability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..implementations.agent_factory import AgentType


@dataclass
class AgentConfig:
    """
    Configuration for agent initialization.

    This class encapsulates all agent-specific parameters, making it easy
    to pass agent configuration between components without mixing it with
    experiment configuration.

    Attributes:
        agent_type: Type of agent to use (AgentType.DS_STAR or AgentType.REACT)
        model: Model identifier (e.g., "watsonx/mistralai/mistral-medium-2505", "claude-3-sonnet-20240229")
        temperature: Temperature for generation (0.0 to 1.0)
        max_steps: Maximum number of reasoning/planning steps
        max_debug_tries: Maximum number of debug attempts per step (DS-Star only)
        code_timeout: Timeout for code execution in seconds (DS-Star only)
        code_mode: Code execution mode ("stepwise" or "full") (DS-Star only)
        output_max_length: Maximum length for output truncation in prompts (DS-Star only)
        logs_max_length: Maximum length for log truncation in prompts (DS-Star only)
        system_prompt: Optional system prompt to guide agent behavior
        task_prompt: Optional task-specific prompt (DS-Star only)
        extra_params: Additional agent-specific parameters
    """

    agent_type: "AgentType | str" = (
        "ds_star"  # AgentType enum or string for backwards compatibility
    )
    model: str = "watsonx/mistralai/mistral-medium-2505"
    temperature: float = 0.0
    max_steps: int = 5
    max_debug_tries: int = 5
    code_timeout: int = 30
    code_mode: str = "stepwise"
    output_max_length: int = 500
    logs_max_length: int = 20000
    system_prompt: Optional[str] = None
    task_prompt: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "agent_type": self.agent_type,
            "model": self.model,
            "temperature": self.temperature,
            "max_steps": self.max_steps,
            "max_debug_tries": self.max_debug_tries,
            "code_timeout": self.code_timeout,
            "code_mode": self.code_mode,
            "output_max_length": self.output_max_length,
            "logs_max_length": self.logs_max_length,
            "system_prompt": self.system_prompt,
            "task_prompt": self.task_prompt,
            **self.extra_params,
        }

    def to_cache_key(self) -> str:
        """
        Generate a cache key from this configuration.

        Returns a deterministic hash that uniquely identifies this agent configuration.
        All configuration parameters that affect agent behavior are included.

        Returns:
            16-character hex string representing the configuration hash
        """
        config_dict = self.to_dict()

        # Convert agent_type enum to string if needed
        if hasattr(config_dict.get("agent_type"), "value"):
            config_dict["agent_type"] = config_dict["agent_type"].value

        # Serialize to JSON for consistent hashing
        config_json = json.dumps(config_dict, sort_keys=True)

        # Generate hash
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]


@dataclass
class ExperimentConfig:
    """
    Configuration for experiment execution.

    This class contains experiment-level settings that control how the
    pipeline runs, separate from agent-specific configuration.

    Attributes:
        run_id: Unique identifier for this experiment run
        fail_fast: Whether to stop on first error
        continue_on_error: Whether to continue after errors (opposite of fail_fast)
        output_dir: Directory for saving experiment results
        cache_dir: Directory for caching intermediate results
        agent_config: Agent-specific configuration
        use_cache: Whether to enable caching
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """

    run_id: str
    fail_fast: bool = False
    continue_on_error: bool = True
    output_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    agent_config: Optional[AgentConfig] = None
    use_cache: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure continue_on_error is opposite of fail_fast
        if self.fail_fast:
            self.continue_on_error = False

        # Convert string paths to Path objects
        if self.output_dir and isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.cache_dir and isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "run_id": self.run_id,
            "fail_fast": self.fail_fast,
            "continue_on_error": self.continue_on_error,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "use_cache": self.use_cache,
            "log_level": self.log_level,
            "agent_config": self.agent_config.to_dict() if self.agent_config else None,
        }
