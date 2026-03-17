"""
Demo Experiment Main Entry Point.

This is the example from the original example.py, now structured as a proper experiment.
Demonstrates the 5-step pipeline with configurable parameters similar to HotpotQA.
"""

from __future__ import annotations

from typing import Sequence

from ...base.base_experiment import BaseExperiment
from ...implementations.agent_factory import AgentFactory, AgentType
from ...interfaces.agent_builder import AgentBuilder
from ...interfaces.data_reader import DataReader
from ...interfaces.evaluator import Evaluator
from ...interfaces.tool_builder import ToolBuilder
from .data_reader import SimpleDataReader, create_sample_benchmarks
from .evaluators_builder import DemoEvaluatorsBuilder
from .tool_builder import EchoToolBuilder


class DemoExperiment(BaseExperiment):
    """
    Demo experiment with configurable parameters.

    Demonstrates the 5-step pipeline with proper agent and evaluator configuration.
    """

    def __init__(
        self,
        model: str = "watsonx/mistralai/mistral-medium-2505",
        max_steps: int = 5,
        temperature: float = 0.0,
        code_timeout: int = 30,
    ) -> None:
        """
        Initialize demo experiment.

        Args:
            model: Model ID to use for the agent
            max_steps: Maximum reasoning steps for the agent
            temperature: Temperature for generation
            code_timeout: Timeout for code execution in seconds
        """
        self._capture_init_args(locals())  # FIRST LINE - capture config
        super().__init__(
            model=model,
            max_steps=max_steps,
            temperature=temperature,
            code_timeout=code_timeout,
        )
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.code_timeout = code_timeout
        # Create sample benchmarks
        self.benchmarks = create_sample_benchmarks()

    def get_data_reader(self) -> DataReader:
        """Get the data reader."""
        # SimpleDataReader doesn't use caching
        return SimpleDataReader(self.benchmarks)

    def get_tools_builder(self) -> Sequence[ToolBuilder]:
        """Get the tool builders."""
        return [EchoToolBuilder()]

    def get_agent_builder(self) -> AgentBuilder:
        """Get the agent builder."""
        return AgentFactory(
            agent_type=AgentType.DS_STAR,
            model=self.model,
            temperature=self.temperature,
            max_steps=self.max_steps,
            code_timeout=self.code_timeout,
        )

    def get_evaluators(self) -> Sequence[Evaluator]:
        """Get the evaluators."""
        return DemoEvaluatorsBuilder.build_evaluators()


def run_demo_experiment(
    model: str = "watsonx/mistralai/mistral-medium-2505",
    max_steps: int = 5,
    temperature: float = 0.0,
    code_timeout: int = 30,
    fail_fast: bool = False,
) -> tuple:
    """
    Run the complete demo experiment.

    Args:
        model: Model ID to use for the agent
        max_steps: Maximum reasoning steps for the agent
        temperature: Temperature for generation
        code_timeout: Timeout for code execution in seconds
        fail_fast: Whether to stop on first error

    Returns:
        Tuple of (outputs, results)
    """
    experiment = DemoExperiment(
        model=model,
        max_steps=max_steps,
        temperature=temperature,
        code_timeout=code_timeout,
    )
    return experiment.experiment_main(fail_fast=fail_fast)


def main():
    """Main entry point for command-line execution."""
    import argparse
    from datetime import datetime
    from pathlib import Path

    from ...utils import setup_custom_api_provider, setup_logging_with_file

    # Register Custom API provider for custom model support
    setup_custom_api_provider()

    parser = argparse.ArgumentParser(description="Run demo experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="watsonx/mistralai/mistral-medium-2505",
        help="Model ID to use",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum reasoning steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--code-timeout",
        type=int,
        default=30,
        help="Timeout for code execution in seconds",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error",
    )

    args = parser.parse_args()

    # Setup output directory and logging with file
    # Use the same directory structure as BaseExperiment
    experiment_dir = Path(__file__).parent
    output_dir = experiment_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate timestamp for log filename (similar to output files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate log filename following the same pattern as output files
    # Format: result_<model_name>_demo_<timestamp>_log.txt
    # For demo, we use a simpler naming since there's no agent_type or question_limit
    log_filename = f"result_demo_{timestamp}_log.txt"

    setup_logging_with_file(output_dir, log_filename)

    run_demo_experiment(
        model=args.model,
        max_steps=args.max_steps,
        temperature=args.temperature,
        code_timeout=args.code_timeout,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
