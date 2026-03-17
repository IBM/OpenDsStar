"""
Experiment Runner - A modular framework for running RAG experiments.

This package provides a clean, interface-driven architecture for:
1. Reading benchmark data
2. Creating tools
3. Building agents with tools
4. Running agents over benchmarks
5. Evaluating agent outputs

Example usage:
    from experiments import ExperimentPipeline, PipelineContext, PipelineConfig
    from experiments.utils import StdoutLogger
    from experiments.implementations import (
        SimpleBenchmarkReader,
        EchoToolBuilder,
        SimpleAgentBuilder,
        SimpleAgentRunner,
        NumericExactEvaluator,
        TextExactEvaluator,
    )

    # Setup context
    ctx = PipelineContext(
        config=PipelineConfig(run_id="my_experiment"),
        logger=StdoutLogger(),
    )

    # Create pipeline
    pipeline = ExperimentPipeline(
        ctx=ctx,
        benchmark_reader=SimpleBenchmarkReader(benchmarks),
        tool_builders=[EchoToolBuilder()],
        agent_builder=SimpleAgentBuilder(),
        agent_runner=SimpleAgentRunner(),
        evaluators=[NumericExactEvaluator(), TextExactEvaluator()],
    )

    # Run experiment
    outputs, results = pipeline.run()
"""

from .core import (
    AgentOutput,
    BenchmarkEntry,
    Document,
    EvalResult,
    GroundTruth,
    PipelineConfig,
    PipelineContext,
    ProcessedBenchmark,
)
from .pipeline import ExperimentPipeline

__all__ = [
    "ExperimentPipeline",
    "PipelineContext",
    "PipelineConfig",
    "Document",
    "BenchmarkEntry",
    "GroundTruth",
    "ProcessedBenchmark",
    "AgentOutput",
    "EvalResult",
]

__version__ = "0.1.0"
