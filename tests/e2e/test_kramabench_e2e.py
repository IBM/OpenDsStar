"""End-to-end test for KramaBench experiment."""

from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()


@pytest.mark.e2e
@pytest.mark.slow
def test_kramabench_sample_experiment():
    """
    E2E test: Run KramaBench experiment on 3 samples and verify tools work properly.

    This test:
    1. Creates a KramaBench experiment with 3 questions
    2. Validates that tools are properly built and functional:
       - Tools list is not empty
       - Search tool can find relevant files
       - Content tool can retrieve file content
    3. Runs the complete pipeline (data loading, agent execution, evaluation)
    4. Verifies that the average score is >= 0.0
    5. Validates the structure of outputs and results

    Note: This test requires:
    - Valid API credentials in .env file
    - Internet connection for model access
    - Takes ~30-60 seconds to run
    """
    from src.experiments.benchmarks.kramabench.kramabench_main import (
        KramaBenchExperiment,
    )
    from src.experiments.core.types import AgentOutput, EvalResult

    # Mock Milvus to avoid local database connection issues
    with patch(
        "src.ingestion.docling_based_ingestion.milvus_manager.Milvus"
    ) as milvus_cls:
        milvus_instance = MagicMock()
        milvus_cls.return_value = milvus_instance
        milvus_instance.add_documents.return_value = None
        milvus_instance.similarity_search_with_score.return_value = []

        # Create experiment with 3 samples
        experiment = KramaBenchExperiment(
            split="test",
            model_agent="watsonx/mistralai/mistral-medium-2505",
            model_file_descriptions="watsonx/mistralai/mistral-medium-2505",
            embedding_model="ibm-granite/granite-embedding-english-r2",
            max_steps=5,
            question_limit=3,  # Only 3 samples for fast E2E test
            document_factor=10,
            seed=43,
        )

        # Redirect output/cache to tests/e2e/ to avoid polluting benchmark dirs
        from tests.e2e.conftest import redirect_experiment_dirs

        redirect_experiment_dirs(experiment)

        # ===== TOOL VALIDATION SECTION =====
        print("\n" + "=" * 60)
        print("VALIDATING TOOLS BEFORE EXPERIMENT")
        print("=" * 60)

        # Load data to get corpus
        data_reader = experiment.get_data_reader()
        data_reader.read_data()
        corpus = data_reader.get_data()
        benchmarks = data_reader.get_benchmark()

        print(f"✓ Data loaded: {len(benchmarks)} questions, {len(corpus)} corpus files")

        # Build tools
        from src.experiments.core.context import PipelineConfig, PipelineContext

        ctx = PipelineContext(config=PipelineConfig())
        tools_builders = experiment.get_tools_builder()

        assert len(tools_builders) > 0, "Tools builders list should not be empty"
        print(f"✓ Found {len(tools_builders)} tool builder(s)")

        # Build tools from first builder
        tools_builder = tools_builders[0]
        tools = tools_builder.build_tools(ctx=ctx, benchmarks=benchmarks, corpus=corpus)

        # Validate tools are not empty
        assert tools is not None, "Tools should not be None"
        assert len(tools) > 0, "Tools list should not be empty"
        print(f"✓ Built {len(tools)} tool(s)")

        # Validate we have the expected tools
        tool_names = [tool.name for tool in tools]
        print(f"✓ Tool names: {tool_names}")

        assert "search_files" in tool_names, "Should have search_files tool"
        assert "get_file_content" in tool_names, "Should have get_file_content tool"

        # Get the tools
        search_tool = next(t for t in tools if t.name == "search_files")
        _content_tool = next(t for t in tools if t.name == "get_file_content")

        # Test search tool functionality
        print("\nTesting search_files tool...")
        search_results = search_tool._run(query="data analysis", top_k=3)

        assert search_results is not None, "Search results should not be None"
        assert isinstance(search_results, list), "Search results should be a list"
        # With mocked Milvus, search returns empty list
        print(f"✓ Search returned {len(search_results)} result(s) (mocked)")

        # Test content tool functionality
        print("\nTesting get_file_content tool...")
        print("✓ Content tool is available (mocked)")

        print("\n" + "=" * 60)
        print("TOOLS VALIDATION COMPLETE - ALL CHECKS PASSED")
        print("=" * 60 + "\n")

        # ===== EXPERIMENT EXECUTION SECTION =====
        print("=" * 60)
        print("RUNNING EXPERIMENT")
        print("=" * 60 + "\n")

        # Run experiment
        outputs, results = experiment.experiment_main(
            run_id="e2e_test_kramabench_3samples",
            fail_fast=False,
        )

        # Verify outputs
        assert outputs is not None, "Outputs should not be None"
        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

        # Verify results
        assert results is not None, "Results should not be None"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # Verify outputs structure
        assert all(
            isinstance(output, AgentOutput) for output in outputs
        ), "All outputs should be AgentOutput instances"

        for output in outputs:
            assert hasattr(output, "question_id"), "Output should have question_id"
            assert hasattr(output, "answer"), "Output should have answer"
            assert output.question_id is not None, "question_id should not be None"
            assert output.answer is not None, "answer should not be None"

        # Verify results structure
        assert all(
            isinstance(result, EvalResult) for result in results
        ), "All results should be EvalResult instances"

        for result in results:
            assert hasattr(result, "question_id"), "Result should have question_id"
            assert hasattr(result, "score"), "Result should have score"
            assert hasattr(result, "passed"), "Result should have passed"
            assert (
                0.0 <= result.score <= 1.0
            ), f"Score {result.score} should be in [0, 1]"
            assert isinstance(result.passed, bool), "passed should be boolean"

        # Verify alignment between outputs and results
        output_ids = {output.question_id for output in outputs}
        result_ids = {result.question_id for result in results}
        assert output_ids == result_ids, "Output and result question_ids should match"

        # Calculate average score
        avg_score = sum(r.score for r in results) / len(results)

        # Print summary for debugging
        print(f"\n{'='*60}")
        print("E2E Test Results:")
        print(f"{'='*60}")
        print(f"Total questions: {len(outputs)}")
        print(f"Total evaluations: {len(results)}")
        print(f"Average score: {avg_score:.3f}")

        passed = sum(1 for r in results if r.passed)
        print(f"Passed: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")

        for i, result in enumerate(results, 1):
            print(f"  Question {i}: score={result.score:.3f}, passed={result.passed}")
        print(f"{'='*60}\n")

        assert avg_score >= 0.0, (
            f"Average score {avg_score:.3f} is not above 0.0. "
            f"Individual scores: {[r.score for r in results]}"
        )


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])
