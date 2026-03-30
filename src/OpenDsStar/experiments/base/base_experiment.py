"""
Base Experiment Class.

This module provides an abstract base class for all experiments.
Each experiment must implement methods to provide:
- Data reader (corpus and benchmark data)
- Tools builder (tools for the agent)
- Agent builder (agent configuration)
- Evaluators (evaluation metrics)

The experiment_main method is provided by default but can be overridden.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Sequence, Tuple, TypeVar

from ..core.context import PipelineConfig, PipelineContext
from ..core.types import AgentOutput, EvalResult
from ..interfaces.agent_builder import AgentBuilder
from ..interfaces.agent_runner import AgentRunner
from ..interfaces.data_reader import DataReader
from ..interfaces.evaluator import Evaluator
from ..interfaces.tool_builder import ToolBuilder
from ..pipeline import ExperimentPipeline
from ..utils.recreatable import Recreatable

T = TypeVar("T", bound="BaseExperiment")


class BaseExperiment(Recreatable, ABC):
    """
    Abstract base class for all experiments.

    Each experiment must implement:
    - get_data_reader(): Returns the data reader (corpus and benchmarks)
    - get_tools_builder(): Returns list of tool builders
    - get_agent_builder(): Returns the agent builder
    - get_evaluators(): Returns list of evaluators

    The experiment_main() method orchestrates the pipeline and can be overridden
    if custom behavior is needed.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the experiment.

        Args:
            **kwargs: Experiment-specific configuration parameters
        """
        self.config = kwargs

        # Setup directories early - use the specific experiment's directory
        import inspect

        experiment_module = inspect.getmodule(self.__class__)
        if experiment_module and experiment_module.__file__:
            self._experiment_dir = Path(experiment_module.__file__).parent
        else:
            # Fallback to the experiments directory if we can't determine the specific one
            self._experiment_dir = Path(__file__).parent

        # Setup output and cache directories
        self.output_dir, self.cache_dir = self.setup_directories(self._experiment_dir)

    @property
    def experiment_dir(self) -> Path:
        """Get the experiment directory."""
        return self._experiment_dir

    @abstractmethod
    def get_data_reader(self) -> DataReader:
        """
        Get the data reader for this experiment.

        Use self.cache_dir for caching if needed.

        Returns:
            DataReader instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_tools_builder(self) -> Sequence[ToolBuilder]:
        """
        Get the tool builders for this experiment.

        Use self.cache_dir for caching if needed.

        Returns:
            List of ToolBuilder instances
        """
        raise NotImplementedError

    @abstractmethod
    def get_agent_builder(self) -> AgentBuilder:
        """
        Get the agent builder for this experiment.

        Returns:
            AgentBuilder instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_evaluators(self) -> Sequence[Evaluator]:
        """
        Get the evaluators for this experiment.

        Returns:
            List of Evaluator instances
        """
        raise NotImplementedError

    def get_experiment_params(self) -> dict:
        """
        Get experiment parameters for saving/loading.

        Override this method to provide experiment-specific parameters
        that can be saved and used to reproduce the experiment.

        Returns:
            Dictionary of experiment parameters
        """
        if not hasattr(self, "config") or not self.config:
            raise NotImplementedError("Config is missing")

        # Create a copy of config
        params = self.config.copy()

        # Convert all enum values to their string representations
        for key, value in params.items():
            if isinstance(value, Enum):
                params[key] = value.value

        return params

    def get_agent_runner(self) -> AgentRunner:
        """
        Get the agent runner for this experiment.

        By default, uses InvokeAgentRunner which works with any agent
        that has an invoke() method. Override if needed.

        Returns:
            AgentRunner instance
        """
        from ..implementations import InvokeAgentRunner

        return InvokeAgentRunner()

    def create_pipeline_context(
        self,
        run_id: str,
        output_dir: Path,
        cache_dir: Path,
        fail_fast: bool = False,
    ) -> PipelineContext:
        """
        Create the pipeline context for this experiment.

        Args:
            run_id: Unique identifier for this run
            output_dir: Directory for output files
            cache_dir: Directory for cache files (used by AgentCache and EvaluationCache)
            fail_fast: Whether to stop on first error

        Returns:
            PipelineContext instance
        """
        return PipelineContext(
            config=PipelineConfig(
                run_id=run_id,
                fail_fast=fail_fast,
                continue_on_error=not fail_fast,
                output_dir=output_dir,
                cache_dir=cache_dir,
                agent_config=None,
            ),
        )

    def setup_directories(self, experiment_dir: Path) -> Tuple[Path, Path]:
        """
        Setup output and cache directories.

        Args:
            experiment_dir: Base experiment directory

        Returns:
            Tuple of (output_dir, cache_dir)
        """
        output_dir = experiment_dir / "output"
        cache_dir = experiment_dir / "cache"

        output_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)

        return output_dir, cache_dir

    def print_header(self, title: str, **info: Any) -> None:
        """
        Print experiment header.

        Args:
            title: Experiment title
            **info: Additional information to display
        """
        print("=" * 80)
        print(title)
        print("=" * 80)
        for key, value in info.items():
            print(f"{key}: {value}")
        print("=" * 80)

    def print_summary(
        self,
        outputs: Sequence[AgentOutput],
        results: Sequence[EvalResult],
    ) -> None:
        """
        Print experiment summary.

        Args:
            outputs: Agent outputs
            results: Evaluation results
        """
        print("\n" + "=" * 80)
        print("Experiment Summary")
        print("=" * 80)
        print(f"Total questions: {len(results)}")
        print(f"Total outputs: {len(outputs)}")

        if results:
            avg_score = sum(r.score for r in results) / len(results)
            passed_count = sum(1 for r in results if r.passed)
            print(f"Average score: {avg_score:.2%}")
            print(
                f"Passed: {passed_count}/{len(results)} ({passed_count/len(results):.2%})"
            )

        print("=" * 80)
        print("Experiment Complete!")
        print("=" * 80)

    def experiment_main(
        self,
        run_id: str | None = None,
        fail_fast: bool = False,
    ) -> Tuple[Sequence[AgentOutput], Sequence[EvalResult]]:
        """
        Main experiment execution method.

        This method orchestrates the complete experiment pipeline:
        1. Setup directories
        2. Create pipeline context
        3. Setup logging with file
        4. Create and run pipeline
        5. Print summary

        Should NOT need to be overridden - all customization should be
        done through the four abstract methods.

        Args:
            run_id: Unique identifier for this run (defaults to experiment name)
            fail_fast: Whether to stop on first error

        Returns:
            Tuple of (outputs, results)
        """
        if run_id is None:
            run_id = self.__class__.__name__.lower().replace("experiment", "")

        # Print header
        self.print_header(
            f"{self.__class__.__name__}",
            run_id=run_id,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            **self.config,
        )

        ctx = self.create_pipeline_context(
            run_id=run_id,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            fail_fast=fail_fast,
        )

        # Generate timestamp once for all files (output, params, and log)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup logging with file using the same timestamp
        log_filename = self._generate_filename(run_id, "log", timestamp).replace(
            ".json", ".txt"
        )
        from OpenDsStar.experiments.utils import setup_logging_with_file

        setup_logging_with_file(self.output_dir, log_filename)

        pipeline = ExperimentPipeline(
            ctx=ctx,
            data_reader=self.get_data_reader(),
            tool_builders=self.get_tools_builder(),
            agent_builder=self.get_agent_builder(),
            agent_runner=self.get_agent_runner(),
            evaluators=self.get_evaluators(),
            experiment_params=self.get_experiment_params(),
            timestamp=timestamp,
        )
        outputs, results = pipeline.run()

        # Save experiment parameters alongside results (use same timestamp)
        self._save_experiment_params(self.output_dir, run_id, timestamp)

        self.print_summary(outputs, results)
        return outputs, results

    def _generate_filename(self, run_id: str, suffix: str, timestamp: str) -> str:
        """
        Generate a filename using experiment parameters.

        Naming convention: result_<agent_type>_<model_name>_<experiment_name>[_<question_limit>]_<timestamp>_<suffix>.json

        Args:
            run_id: Run identifier
            suffix: File suffix (e.g., "output" or "params")
            timestamp: Timestamp string

        Returns:
            Generated filename
        """
        from OpenDsStar.core.model_registry import ModelRegistry

        params = self.get_experiment_params()

        # Get agent_type from parameters
        agent_type = params.get("agent_type", "unknown")
        if hasattr(agent_type, "value"):
            agent_type = str(agent_type.value)
        else:
            agent_type = str(agent_type)

        # Get model name from parameters
        model = params.get("model", "unknown")
        model_name = ModelRegistry.get_model_name(model)

        # Get experiment name from run_id
        experiment_name = (
            run_id.split("_")[0]
            if run_id
            else self.__class__.__name__.lower().replace("experiment", "")
        )

        # Get question_limit from parameters
        question_limit = params.get("question_limit")

        # Build filename
        if question_limit is not None:
            return f"result_{agent_type}_{model_name}_{experiment_name}_{question_limit}_{timestamp}_{suffix}.json"
        else:
            return f"result_{agent_type}_{model_name}_{experiment_name}_{timestamp}_{suffix}.json"

    def _save_experiment_params(
        self, output_dir: Path, run_id: str, timestamp: str
    ) -> None:
        """
        Save experiment parameters to a JSON file.

        Naming convention: result_<agent_type>_<experiment_name>[_<question_limit>]_<timestamp>_params.json

        Args:
            output_dir: Directory to save parameters
            run_id: Run identifier
            timestamp: Timestamp string (shared with output file)
        """
        # Generate filename using shared method with provided timestamp
        filename = self._generate_filename(run_id, "params", timestamp)
        params_file = output_dir / filename

        # Save config
        self.save_config(params_file)

        print(f"\nExperiment parameters saved to: {params_file}")
