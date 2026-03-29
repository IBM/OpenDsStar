"""
DataBench Experiment Main Entry Point.

This module orchestrates the complete DataBench experiment:
1. Load DataBench data from HuggingFace (cardiffnlp/databench)
2. Build data analysis tools (Python executor, guidance)
3. Create agent with those tools
4. Run the agent over the benchmark
5. Evaluate results
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from OpenDsStar.core.model_registry import ModelRegistry

from ...base.base_experiment import BaseExperiment
from ...evaluators.ragbench_llm_judge import RagbenchLLMJudgeEvaluator
from ...implementations import AgentFactory, AgentType
from ...interfaces.agent_builder import AgentBuilder
from ...interfaces.data_reader import DataReader
from ...interfaces.evaluator import Evaluator
from ...interfaces.tool_builder import ToolBuilder
from .data_reader import DataBenchDataReader
from .tools_builder import DataBenchToolsBuilder


class DataBenchExperiment(BaseExperiment):
    """
    DataBench experiment implementation.

    This experiment loads DataBench data from HuggingFace, builds data analysis tools,
    creates an agent, and evaluates it on the benchmark.

    Only overrides the four core methods required by BaseExperiment:
    - get_data_reader()
    - get_tools_builder()
    - get_agent_builder()
    - get_evaluators()
    """

    def __init__(
        self,
        qa_split: Literal["train"] = "train",
        semeval_split: Literal["train", "validation", "test"] = "train",
        model_agent: str = ModelRegistry.WX_MISTRAL_MEDIUM,
        model_file_descriptions: str = ModelRegistry.WX_MISTRAL_MEDIUM,
        embedding_model: str = ModelRegistry.GRANITE_EMBEDDING,
        max_steps: int = 10,
        agent_type: AgentType = AgentType.DS_STAR,
        temperature: float = 0.0,
        max_debug_tries: int = 5,
        question_limit: Optional[int] = None,
        seed: int = 43,
        output_max_length: int = 2000,
        logs_max_length: int = 100000,
        code_mode: str = "stepwise",
        parallel_workers: Optional[int] = None,
    ) -> None:
        """
        Initialize DataBench experiment.

        Args:
            qa_split: Dataset split for QA config (typically "train")
            semeval_split: Dataset split for SemEval config ("train", "validation", or "test")
            model_agent: Model ID to use for the agent
            model_file_descriptions: Model ID for file description generation
            embedding_model: Embedding model ID for vector store
            max_steps: Maximum reasoning steps for the agent
            agent_type: Type of agent to use ("ds_star", "react_langchain", etc.)
            temperature: Temperature for generation
            max_debug_tries: Maximum number of debug attempts per step
            question_limit: Limit number of questions (for testing)
            seed: Random seed for reproducibility
            output_max_length: Maximum length for output truncation in prompts
            logs_max_length: Maximum length for log truncation in prompts
            code_mode: Code execution mode - "stepwise" (execute each step separately) or
                "full" (regenerate entire script each iteration).
            parallel_workers: Number of parallel workers (None for sequential, >1 for parallel)
        """
        self._capture_init_args(locals())  # FIRST LINE - capture config
        super().__init__(
            split=qa_split,  # Use qa_split as the main split
            model=model_agent,
            embedding_model=embedding_model,
            max_steps=max_steps,
            agent_type=agent_type,
            temperature=temperature,
            max_debug_tries=max_debug_tries,
            question_limit=question_limit,
            document_factor=None,
            seed=seed,
            output_max_length=output_max_length,
            logs_max_length=logs_max_length,
        )
        self.qa_split = qa_split
        self.semeval_split = semeval_split
        self.model_agent = model_agent
        self.model_file_descriptions = model_file_descriptions
        self.embedding_model = embedding_model
        self.max_steps = max_steps
        self.agent_type = agent_type
        self.temperature = temperature
        self.max_debug_tries = max_debug_tries
        self.question_limit = question_limit
        self.seed = seed
        self.output_max_length = output_max_length
        self.logs_max_length = logs_max_length
        self.code_mode = code_mode
        self.parallel_workers = parallel_workers

    def get_data_reader(self) -> DataReader:
        """Get the DataBench data reader."""
        return DataBenchDataReader(
            qa_split=self.qa_split,  # type: ignore
            semeval_split=self.semeval_split,  # type: ignore
            question_limit=self.question_limit,
            seed=self.seed,
            use_cache=True,
            cache_base_dir=self.cache_dir,
        )

    def get_tools_builder(self) -> Sequence[ToolBuilder]:
        """Get the DataBench tools builder."""
        return [
            DataBenchToolsBuilder(
                cache_dir=str(self.cache_dir),
                llm=self.model_file_descriptions,
                embedding_model=self.embedding_model,
                temperature=self.temperature,
            )
        ]

    def get_agent_builder(self) -> AgentBuilder:
        """Get the DataBench agent builder."""
        return AgentFactory(
            agent_type=self.agent_type,
            model=self.model_agent,
            max_steps=self.max_steps,
            temperature=self.temperature,
            max_debug_tries=self.max_debug_tries,
            code_mode=self.code_mode,
        )

    def get_agent_runner(self):
        """Get the agent runner for this experiment."""
        if self.parallel_workers is not None and self.parallel_workers > 1:
            from ...implementations import ParallelInvokeAgentRunner

            return ParallelInvokeAgentRunner(max_workers=self.parallel_workers)
        else:
            # Use default sequential runner from base class
            return super().get_agent_runner()

    def get_evaluators(self) -> Sequence[Evaluator]:
        """Get evaluators for DataBench."""
        return [RagbenchLLMJudgeEvaluator()]


def main():
    """Main entry point for command-line execution."""
    import argparse
    import itertools

    from ...utils import setup_custom_api_provider
    from ...utils.logging import setup_logging

    # Setup logging first
    setup_logging()

    # Register Custom API provider for custom model support
    setup_custom_api_provider()

    parser = argparse.ArgumentParser(
        description="Run DataBench experiment. If question_limit is not specified, runs on all questions."
    )
    parser.add_argument(
        "--load-params",
        type=str,
        default=None,
        help="Load experiment from params file and run it",
    )
    parser.add_argument(
        "--qa-split",
        type=str,
        default="train",
        choices=["train"],
        help="Dataset split for QA config (default: train)",
    )
    parser.add_argument(
        "--semeval-split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split for SemEval config (default: train)",
    )
    parser.add_argument(
        "--model-agent",
        type=str,
        nargs="+",
        default=[ModelRegistry.WX_MISTRAL_MEDIUM],
        help=f"Model ID(s) for agent (default: {ModelRegistry.WX_MISTRAL_MEDIUM}). Can specify multiple models.",
    )
    parser.add_argument(
        "--model-file-descriptions",
        type=str,
        default=ModelRegistry.WX_MISTRAL_MEDIUM,
        help=f"Model ID for file descriptions (default: {ModelRegistry.WX_MISTRAL_MEDIUM})",
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
        default=10,
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
        "--max-debug-tries",
        type=int,
        default=10,
        help="Maximum number of debug attempts per step",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Limit number of questions (for testing). If not specified, runs on all questions.",
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
        default=2000,
        help="Maximum length for output truncation in prompts",
    )
    parser.add_argument(
        "--logs-max-length",
        type=int,
        default=100000,
        help="Maximum length for log truncation in prompts",
    )
    parser.add_argument(
        "--code-mode",
        type=str,
        default="full",
        choices=["stepwise", "full"],
        help="Code execution mode: 'stepwise' (execute each step separately) or 'full' (regenerate entire script). Default: full",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Number of parallel workers for running questions (default: None for sequential, use 20 for parallel)",
    )

    args = parser.parse_args()

    # If load-params is specified, load and run from params file
    if args.load_params:
        from pathlib import Path

        print(f"\n{'='*80}")
        print(f"Loading experiment from params file: {args.load_params}")
        print(f"{'='*80}\n")

        # Use Recreatable.load_instance to reconstruct experiment
        experiment = DataBenchExperiment.load_instance(Path(args.load_params))
        experiment.experiment_main(fail_fast=args.fail_fast)
        return

    # Convert agent_type strings to enums
    from ...implementations.agent_factory import AgentType

    agent_type_enums = [AgentType(at) for at in args.agent_type]

    # Resolve model aliases to full model IDs
    model_agents = [ModelRegistry.get_model_id(m) for m in args.model_agent]
    model_file_descriptions = ModelRegistry.get_model_id(args.model_file_descriptions)
    embedding_model = ModelRegistry.get_model_id(args.embedding_model)

    # Generate all combinations of models and agent types
    combinations = list(itertools.product(model_agents, agent_type_enums))

    print(f"\n{'='*80}")
    print(f"Running {len(combinations)} experiment combination(s):")
    for i, (model_agent, agent_type) in enumerate(combinations, 1):
        model_name = ModelRegistry.get_model_name(model_agent)
        print(f"  {i}. Model: {model_name}, Agent: {agent_type.value}")
    print(f"{'='*80}\n")

    # Run experiment for each combination
    for i, (model_agent, agent_type_enum) in enumerate(combinations, 1):
        model_name = ModelRegistry.get_model_name(model_agent)
        print(f"\n{'='*80}")
        print(f"Running combination {i}/{len(combinations)}")
        print(f"Model: {model_name}, Agent: {agent_type_enum.value}")
        print(f"{'='*80}\n")

        experiment = DataBenchExperiment(
            qa_split=args.qa_split,
            semeval_split=args.semeval_split,
            model_agent=model_agent,
            model_file_descriptions=model_file_descriptions,
            embedding_model=embedding_model,
            max_steps=args.max_steps,
            agent_type=agent_type_enum,
            temperature=args.temperature,
            max_debug_tries=args.max_debug_tries,
            question_limit=args.question_limit,
            seed=args.seed,
            output_max_length=args.output_max_length,
            logs_max_length=args.logs_max_length,
            code_mode=args.code_mode,
            parallel_workers=args.parallel_workers,
        )

        experiment.experiment_main(fail_fast=args.fail_fast)

        print(f"\n{'='*80}")
        print(f"Completed combination {i}/{len(combinations)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
