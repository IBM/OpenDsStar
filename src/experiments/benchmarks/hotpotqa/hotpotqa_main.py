"""
HotpotQA Experiment Main Entry Point.

This module orchestrates the complete HotpotQA experiment:
1. Load HotpotQA data using RagDataLoaderFactory
2. Build retrieval tools from the corpus
3. Create DsStarAgent with those tools
4. Run the agent over the benchmark
5. Evaluate results using text exact match
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from core.model_registry import ModelRegistry

from ...base.base_experiment import BaseExperiment
from ...evaluators.ragbench_llm_judge import RagbenchLLMJudgeEvaluator
from ...implementations import AgentFactory, AgentType
from ...interfaces.agent_builder import AgentBuilder
from ...interfaces.data_reader import DataReader
from ...interfaces.evaluator import Evaluator
from ...interfaces.tool_builder import ToolBuilder
from .data_reader import HotpotQADataReader
from .tools_builder import HotpotQAToolsBuilder


class HotpotQAExperiment(BaseExperiment):
    """
    HotpotQA experiment implementation.

    This experiment loads HotpotQA data, builds retrieval tools from the corpus,
    creates a DsStarAgent, and evaluates it on the benchmark.

    Only overrides the four core methods required by BaseExperiment:
    - get_data_reader()
    - get_tools_builder()
    - get_agent_builder()
    - get_evaluators()
    """

    def __init__(
        self,
        split: Literal["train", "test"],
        model: str,
        embedding_model: str,
        max_steps: int = 5,
        agent_type: AgentType = AgentType.DS_STAR,
        temperature: float = 0.0,
        question_limit: Optional[int] = None,
        document_factor: Optional[int] = None,
        seed: int = 43,
        output_max_length: int = 500,
        logs_max_length: int = 20000,
    ) -> None:
        """
        Initialize HotpotQA experiment.

        Args:
            split: Dataset split to use ("train" or "test")
            model: Model ID to use for the agent
            embedding_model: Embedding model ID for vector store
            max_steps: Maximum reasoning steps for the agent
            agent_type: Type of agent to use ("ds_star" or "react")
            temperature: Temperature for generation
            question_limit: Limit number of questions (for testing)
            document_factor: Document sampling factor (for testing)
            seed: Random seed for reproducibility
            output_max_length: Maximum length for output truncation in prompts
            logs_max_length: Maximum length for log truncation in prompts
        """
        self._capture_init_args(locals())  # FIRST LINE - capture config
        super().__init__(
            split=split,
            model=model,
            embedding_model=embedding_model,
            max_steps=max_steps,
            agent_type=agent_type,
            temperature=temperature,
            question_limit=question_limit,
            document_factor=document_factor,
            seed=seed,
            output_max_length=output_max_length,
            logs_max_length=logs_max_length,
        )
        self.split = split
        self.model = model
        self.embedding_model = embedding_model
        self.max_steps = max_steps
        self.agent_type = agent_type
        self.temperature = temperature
        self.question_limit = question_limit
        self.document_factor = document_factor
        self.seed = seed
        self.output_max_length = output_max_length
        self.logs_max_length = logs_max_length

    def get_data_reader(self) -> DataReader:
        """Get the HotpotQA data reader."""

        # Override data reader with sampling parameters for this sample run
        return HotpotQADataReader(
            split=self.split,  # type: ignore
            question_limit=self.question_limit,
            document_factor=self.document_factor,
            seed=self.seed,
            cache_base_dir=self.cache_dir,
        )

    def get_tools_builder(self) -> Sequence[ToolBuilder]:
        """Get the HotpotQA tools builder."""
        return [
            HotpotQAToolsBuilder(
                embedding_model=self.embedding_model,
                temperature=self.temperature,
            )
        ]

    def get_agent_builder(self) -> AgentBuilder:
        """Get the HotpotQA agent builder."""
        return AgentFactory(
            agent_type=self.agent_type,
            model=self.model,
            max_steps=self.max_steps,
            temperature=self.temperature,
        )

    def get_evaluators(self) -> Sequence[Evaluator]:
        return [RagbenchLLMJudgeEvaluator()]


def run_hotpotqa_experiment(
    split: Literal["train", "test"] = "test",
    model: str = ModelRegistry.WX_MISTRAL_MEDIUM,
    embedding_model: str = ModelRegistry.GRANITE_EMBEDDING,
    max_steps: int = 5,
    agent_type: AgentType = AgentType.DS_STAR,
    temperature: float = 0.0,
    fail_fast: bool = False,
    question_limit: Optional[int] = None,
    document_factor: Optional[int] = None,
    seed: int = 43,
    output_max_length: int = 500,
    logs_max_length: int = 20000,
) -> tuple:
    """
    Run the complete HotpotQA experiment.

    Args:
        split: Dataset split to use ("train" or "test")
        model: Model ID to use for the agent
        embedding_model: Embedding model ID for vector store
        max_steps: Maximum reasoning steps for the agent
        agent_type: Type of agent to use ("ds_star" or "react")
        temperature: Temperature for generation
        fail_fast: Whether to stop on first error
        question_limit: Limit number of questions (for testing)
        document_factor: Document sampling factor (for testing)
        seed: Random seed for reproducibility
        output_max_length: Maximum length for output truncation in prompts
        logs_max_length: Maximum length for log truncation in prompts

    Returns:
        Tuple of (outputs, results)
    """
    experiment = HotpotQAExperiment(
        split=split,
        model=model,
        embedding_model=embedding_model,
        max_steps=max_steps,
        agent_type=agent_type,
        temperature=temperature,
        question_limit=question_limit,
        document_factor=document_factor,
        seed=seed,
        output_max_length=output_max_length,
        logs_max_length=logs_max_length,
    )
    return experiment.experiment_main(fail_fast=fail_fast)


def main():
    """Main entry point for command-line execution."""
    import argparse
    from itertools import product

    from core.model_registry import ModelRegistry

    from ...utils import setup_custom_api_provider

    # Register Custom API provider for custom model support
    setup_custom_api_provider()

    parser = argparse.ArgumentParser(
        description="Run HotpotQA experiment. If question_limit is not specified, runs on all questions."
    )
    parser.add_argument(
        "--load-params",
        type=str,
        default=None,
        help="Load experiment from params file and run it",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=[ModelRegistry.WX_MISTRAL_MEDIUM],
        help=f"Model ID(s) to use (default: {ModelRegistry.WX_MISTRAL_MEDIUM}). Can specify multiple models.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=ModelRegistry.GRANITE_EMBEDDING,
        help=f"Embedding model ID for vector store (default: {ModelRegistry.GRANITE_EMBEDDING})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum reasoning steps",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        nargs="+",
        default=["ds_star"],
        choices=[
            "ds_star",
            "react_langchain",
            "react_smolagents",
            "codeact_smolagents",
        ],
        help="Type(s) of agent to use. Can specify multiple agent types.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Limit number of questions (for testing). If not specified, runs on all questions.",
    )
    parser.add_argument(
        "--document-factor",
        type=int,
        default=None,
        help="Document sampling factor (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error",
    )
    parser.add_argument(
        "--output-max-length",
        type=int,
        default=500,
        help="Maximum length for output truncation in prompts",
    )
    parser.add_argument(
        "--logs-max-length",
        type=int,
        default=20000,
        help="Maximum length for log truncation in prompts",
    )

    args = parser.parse_args()

    # If load-params is specified, load and run from params file
    if args.load_params:
        from pathlib import Path

        print(f"\n{'='*80}")
        print(f"Loading experiment from params file: {args.load_params}")
        print(f"{'='*80}\n")

        # Use Recreatable.load_instance to reconstruct experiment
        experiment = HotpotQAExperiment.load_instance(Path(args.load_params))
        experiment.experiment_main(fail_fast=args.fail_fast)
        return

    # Convert agent_type strings to enums
    from ...implementations.agent_factory import AgentType

    agent_type_enums = [AgentType(at) for at in args.agent_type]

    # Resolve model aliases to full model IDs
    models = [
        ModelRegistry.get_model_id(m) if m else ModelRegistry.WX_MISTRAL_MEDIUM
        for m in args.model
    ]
    embedding_model = (
        ModelRegistry.get_model_id(args.embedding_model)
        if args.embedding_model
        else ModelRegistry.GRANITE_EMBEDDING
    )

    # Generate all combinations of models and agent types
    combinations = list(product(models, agent_type_enums))

    print(f"\n{'='*80}")
    print(f"Running {len(combinations)} experiment combination(s):")
    for i, (model, agent_type) in enumerate(combinations, 1):
        model_name = ModelRegistry.get_model_name(model)
        print(f"  {i}. Model: {model_name}, Agent: {agent_type.value}")
    print(f"{'='*80}\n")

    # Run experiment for each combination
    for i, (model, agent_type_enum) in enumerate(combinations, 1):
        model_name = ModelRegistry.get_model_name(model)
        print(f"\n{'='*80}")
        print(f"Running combination {i}/{len(combinations)}")
        print(f"Model: {model_name}, Agent: {agent_type_enum.value}")
        print(f"{'='*80}\n")

        # Logging is now set up automatically by BaseExperiment.experiment_main()
        # with the same timestamp as output files

        run_hotpotqa_experiment(
            split=args.split,  # type: ignore
            model=model,
            embedding_model=embedding_model,
            max_steps=args.max_steps,
            agent_type=agent_type_enum,
            temperature=args.temperature,
            question_limit=args.question_limit,
            document_factor=args.document_factor,
            seed=args.seed,
            fail_fast=args.fail_fast,
            output_max_length=args.output_max_length,
            logs_max_length=args.logs_max_length,
        )

        print(f"\n{'='*80}")
        print(f"Completed combination {i}/{len(combinations)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
