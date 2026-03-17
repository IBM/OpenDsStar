"""Shared pytest fixtures for all tests."""

import io
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from experiments.core.config import AgentConfig, ExperimentConfig
from experiments.core.context import PipelineConfig, PipelineContext
from experiments.core.types import (
    AgentOutput,
    BenchmarkEntry,
    Document,
    EvalResult,
    GroundTruth,
    ProcessedBenchmark,
)
from experiments.interfaces.evaluator import Evaluator
from experiments.utils.logging import StdoutLogger


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        document_id="doc1",
        path="/path/to/doc1.txt",
        mime_type="text/plain",
        extra_metadata={"source": "test"},
        stream_factory=lambda: io.BytesIO(b"This is test document content."),
    )


@pytest.fixture
def sample_documents():
    """Create multiple sample documents for testing."""
    return [
        Document(
            document_id=f"doc{i}",
            path=f"/path/to/doc{i}.txt",
            mime_type="text/plain",
            extra_metadata={"source": "test"},
            stream_factory=lambda i=i: io.BytesIO(
                f"This is test document {i} content.".encode()
            ),
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def sample_ground_truth():
    """Create sample ground truth data."""
    return GroundTruth(
        answers=["4", "four"],
        context_ids=["doc1"],
        extra={"difficulty": "easy"},
    )


@pytest.fixture
def sample_benchmark(sample_ground_truth):
    """Create a sample benchmark entry."""
    return BenchmarkEntry(
        question_id="q1",
        question="What is 2+2?",
        ground_truth=sample_ground_truth,
        additional_information={"category": "math"},
    )


@pytest.fixture
def sample_benchmarks(sample_ground_truth):
    """Create multiple sample benchmark entries."""
    return [
        BenchmarkEntry(
            question_id=f"q{i}",
            question=f"Test question {i}?",
            ground_truth=sample_ground_truth,
            additional_information={"category": "test"},
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def sample_processed_benchmark(sample_ground_truth):
    """Create a sample processed benchmark."""
    return ProcessedBenchmark(
        question_id="q1",
        question="What is 2+2?",
        ground_truth=sample_ground_truth,
        metadata={"category": "math"},
    )


@pytest.fixture
def sample_agent_output():
    """Create a sample agent output."""
    return AgentOutput(
        question_id="q1",
        answer="4",
        metadata={
            "steps_used": 2,
            "input_tokens": 100,
            "output_tokens": 50,
        },
    )


@pytest.fixture
def sample_eval_result():
    """Create a sample evaluation result."""
    return EvalResult(
        question_id="q1",
        score=1.0,
        passed=True,
        details={"method": "exact_match"},
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock(spec=BaseChatModel)
    llm.invoke.return_value = AIMessage(content="Test response from LLM")
    llm.model = "test-model"
    return llm


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    from pydantic import BaseModel, Field

    class MockToolInput(BaseModel):
        query: str = Field(description="Test query parameter")

    tool = Mock(spec=BaseTool)
    tool.name = "test_tool"
    tool.description = "A test tool for testing purposes"
    tool._run.return_value = "tool result"
    tool.args_schema = MockToolInput
    return tool


@pytest.fixture
def mock_tools():
    """Create multiple mock tools for testing."""
    from pydantic import BaseModel, Field

    class MockToolInput(BaseModel):
        query: str = Field(description="Test query parameter")

    tools = []
    for i in range(1, 4):
        tool = Mock(spec=BaseTool)
        tool.name = f"test_tool_{i}"
        tool.description = f"Test tool {i}"
        tool._run.return_value = f"result_{i}"
        tool.args_schema = MockToolInput
        tools.append(tool)
    return tools


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator for testing."""
    evaluator = Mock(spec=Evaluator)
    evaluator.metric_id = "test_metric"
    evaluator.evaluate_one.return_value = EvalResult(
        question_id="q1",
        score=1.0,
        passed=True,
        details={"method": "mock"},
    )
    return evaluator


@pytest.fixture
def agent_config():
    """Create a sample agent configuration."""
    return AgentConfig(
        model="watsonx/mistralai/mistral-medium-2505",
        temperature=0.0,
        max_steps=5,
        code_timeout=30,
        code_mode="stepwise",
    )


@pytest.fixture
def experiment_config(agent_config, tmp_path):
    """Create a sample experiment configuration."""
    return ExperimentConfig(
        run_id="test_run",
        fail_fast=False,
        continue_on_error=True,
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
        agent_config=agent_config,
        use_cache=True,
        log_level="INFO",
    )


@pytest.fixture
def pipeline_config(tmp_path):
    """Create a pipeline configuration for testing."""
    return PipelineConfig(
        run_id="test_run",
        fail_fast=False,
        continue_on_error=True,
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def pipeline_context(pipeline_config):
    """Create a pipeline context for testing."""
    return PipelineContext(
        config=pipeline_config,
        logger=StdoutLogger(),
    )
