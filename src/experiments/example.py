"""
Example usage of the refactored experiment runner.

This demonstrates the 5-step pipeline:
1. Read benchmark data
2. Create tools
3. Build agent with tools
4. Run agent over benchmark
5. Evaluate agent's output
"""

from experiments.implementations import (
    EchoToolBuilder,
    NumericExactEvaluator,
    SimpleAgentBuilder,
    SimpleAgentRunner,
    SimpleBenchmarkReader,
    TextExactEvaluator,
    create_sample_benchmarks,
)
from experiments.utils import StdoutLogger
from src.experiments import (
    ExperimentPipeline,
    PipelineConfig,
    PipelineContext,
)


def main() -> None:
    """Run a simple experiment with placeholder implementations."""

    # Step 0: Setup
    print("=" * 60)
    print("Experiment Runner - Example Usage")
    print("=" * 60)

    # Create pipeline context
    ctx = PipelineContext(
        config=PipelineConfig(
            run_id="example_experiment",
            fail_fast=False,
            continue_on_error=True,
        ),
        logger=StdoutLogger(),
    )

    # Create sample benchmarks
    benchmarks = create_sample_benchmarks()
    print(f"\nCreated {len(benchmarks)} sample benchmarks")

    # Create pipeline with all components
    pipeline = ExperimentPipeline(
        ctx=ctx,
        benchmark_reader=SimpleBenchmarkReader(benchmarks),
        tool_builders=[EchoToolBuilder()],
        agent_builder=SimpleAgentBuilder(),
        agent_runner=SimpleAgentRunner(),
        evaluators=[
            NumericExactEvaluator(),
            TextExactEvaluator(),
        ],
    )

    # Run the 5-step pipeline
    print("\n" + "=" * 60)
    print("Running 5-Step Pipeline:")
    print("1. Read benchmark data")
    print("2. Create tools")
    print("3. Build agent with tools")
    print("4. Run agent over benchmark")
    print("5. Evaluate agent's output")
    print("=" * 60 + "\n")

    outputs, results = pipeline.run()

    # Display results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Total questions: {len(results)}")
    print(f"Total outputs: {len(outputs)}")

    if results:
        avg_score = sum(r.score for r in results) / len(results)
        passed_count = sum(1 for r in results if r.passed)
        print(f"Average score: {avg_score:.2f}")
        print(f"Passed: {passed_count}/{len(results)}")

        print("\nDetailed Results:")
        for r in results:
            status = "✓" if r.passed else "✗"
            print(
                f"  {status} {r.question_id}: score={r.score:.2f}, type={r.answer_type}"
            )

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
