"""
Tests for the CodeActAgentSmolagents.
"""

from unittest.mock import Mock, patch

import pytest
from smolagents import RunResult

from agents import CodeActAgentSmolagents


class TestCodeActAgentSmolagents:
    """Test suite for CodeActAgentSmolagents."""

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_initialization_with_litellm_model(self, mock_code_agent):
        """Test initialization with a LiteLLMModel instance."""
        from smolagents import LiteLLMModel

        mock_agent_instance = Mock()
        mock_code_agent.return_value = mock_agent_instance

        # Create LiteLLMModel instance
        litellm_model = LiteLLMModel(model_id="gpt-4o-mini", temperature=0.5)
        agent = CodeActAgentSmolagents(model=litellm_model, temperature=0.5)

        assert agent.model_id == "gpt-4o-mini"
        assert agent.temperature == 0.5
        assert agent.max_steps == 5
        # StringToStreamTool is always added automatically
        assert len(agent.tools) == 1
        from tools.string_to_stream_tool import StringToStreamTool

        assert isinstance(agent.tools[0], StringToStreamTool)

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_initialization_with_custom_params(self, mock_code_agent):
        """Test initialization with custom parameters."""
        from smolagents import LiteLLMModel

        mock_agent_instance = Mock()
        mock_code_agent.return_value = mock_agent_instance

        litellm_model = LiteLLMModel(model_id="gpt-4o", temperature=0.7)
        agent = CodeActAgentSmolagents(
            model=litellm_model,
            temperature=0.7,
            max_steps=10,
            system_prompt="Custom prompt",
            code_timeout=60,
        )

        assert agent.model_id == "gpt-4o"
        assert agent.temperature == 0.7
        assert agent.max_steps == 10
        assert agent.system_prompt == "Custom prompt"
        assert agent.code_timeout == 60

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_initialization_with_tools(self, mock_code_agent):
        """Test initialization with custom tools."""
        from langchain_core.tools import tool
        from smolagents import LiteLLMModel

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return "result"

        mock_agent_instance = Mock()
        mock_code_agent.return_value = mock_agent_instance

        litellm_model = LiteLLMModel(
            model_id="watsonx/mistralai/mistral-medium-2505", temperature=0.0
        )
        agent = CodeActAgentSmolagents(model=litellm_model, tools=[test_tool])
        # Should have test_tool + StringToStreamTool (automatically added)
        assert len(agent.tools) == 2

    def test_invalid_model_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(
            ValueError,
            match="model must be either a model ID string.*or a smolagents LiteLLMModel instance",
        ):
            CodeActAgentSmolagents(model=123)  # type: ignore

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_invoke_with_empty_query(self, mock_code_agent):
        """Test that empty query raises ValueError."""
        from smolagents import LiteLLMModel

        mock_agent_instance = Mock()
        mock_code_agent.return_value = mock_agent_instance

        litellm_model = LiteLLMModel(
            model_id="watsonx/mistralai/mistral-medium-2505", temperature=0.0
        )
        agent = CodeActAgentSmolagents(model=litellm_model)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.invoke("")

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.invoke(None)  # type: ignore

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_invoke_basic(self, mock_code_agent):
        """Test basic invoke functionality."""
        from smolagents import LiteLLMModel

        mock_agent_instance = Mock()
        # Mock RunResult object
        mock_run_result = Mock(spec=RunResult)
        mock_run_result.output = "The answer is 42"
        mock_run_result.steps = ["Step 1: Thinking", "Step 2: Calculating"]
        mock_agent_instance.run.return_value = mock_run_result
        mock_code_agent.return_value = mock_agent_instance

        litellm_model = LiteLLMModel(model_id="gpt-4o-mini", temperature=0.0)
        agent = CodeActAgentSmolagents(model=litellm_model)

        result = agent.invoke("What is the answer?")

        assert "answer" in result
        assert "trajectory" in result
        assert "steps_used" in result
        assert result["answer"] == "The answer is 42"
        assert len(result["trajectory"]) == 2
        assert result["steps_used"] == 2

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_invoke_with_error(self, mock_code_agent):
        """Test invoke when agent raises an error."""
        from smolagents import LiteLLMModel

        mock_agent_instance = Mock()
        mock_agent_instance.run.side_effect = Exception("Test error")
        mock_code_agent.return_value = mock_agent_instance

        litellm_model = LiteLLMModel(model_id="gpt-4o-mini", temperature=0.0)
        agent = CodeActAgentSmolagents(model=litellm_model)

        result = agent.invoke("What is the answer?")

        assert result["answer"] == ""
        assert result["fatal_error"] == "Test error"
        assert result["execution_error"] == "Test error"
        assert result["steps_used"] == 0
        assert result["verifier_sufficient"] is False

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_invoke_return_format(self, mock_code_agent):
        """Test that invoke returns the correct format."""
        from smolagents import LiteLLMModel

        mock_agent_instance = Mock()
        # Mock RunResult object
        mock_run_result = Mock(spec=RunResult)
        mock_run_result.output = "Result"
        mock_run_result.steps = []
        mock_agent_instance.run.return_value = mock_run_result
        mock_code_agent.return_value = mock_agent_instance

        litellm_model = LiteLLMModel(model_id="gpt-4o-mini", temperature=0.0)
        agent = CodeActAgentSmolagents(model=litellm_model)

        result = agent.invoke("Test query")

        # Check all required keys are present
        required_keys = [
            "answer",
            "trajectory",
            "plan",
            "steps_used",
            "max_steps",
            "verifier_sufficient",
            "fatal_error",
            "execution_error",
            "input_tokens",
            "output_tokens",
            "num_llm_calls",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_system_prompt_override(self, mock_code_agent):
        """Test that system prompt is stored but not applied to CodeAgent."""
        from smolagents import LiteLLMModel

        mock_agent_instance = Mock()
        mock_code_agent.return_value = mock_agent_instance

        custom_prompt = "Custom system prompt"
        litellm_model = LiteLLMModel(model_id="gpt-4o-mini", temperature=0.0)
        agent = CodeActAgentSmolagents(model=litellm_model, system_prompt=custom_prompt)

        # System prompt is stored but NOT applied to CodeAgent
        # (CodeAgent needs its default prompt structure to include tool descriptions)
        assert agent.system_prompt == custom_prompt

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_extract_config_from_langchain_model(self, mock_code_agent):
        """Test that we can pass a pre-built LiteLLMModel."""
        from smolagents import LiteLLMModel

        # Create a LiteLLMModel directly
        litellm_model = LiteLLMModel(
            model_id="gpt-4o-mini",
            temperature=0.5,
        )

        mock_agent_instance = Mock()
        mock_code_agent.return_value = mock_agent_instance

        # Initialize agent with LiteLLMModel
        agent = CodeActAgentSmolagents(model=litellm_model, temperature=0.5)

        # Model already built, so ModelBuilder.build should not be called
        assert agent.model_id == "gpt-4o-mini"

    @patch("agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_extract_all_model_kwargs(self, mock_code_agent):
        """Test that we can pass a pre-built LiteLLMModel with custom config."""
        from smolagents import LiteLLMModel

        # Create a LiteLLMModel with custom config
        litellm_model = LiteLLMModel(
            model_id="watsonx/mistralai/mistral-medium-2505",
            temperature=0.5,
            api_key="test-watsonx-key",
            api_base="https://us-south.ml.cloud.ibm.com",
        )

        mock_agent_instance = Mock()
        mock_code_agent.return_value = mock_agent_instance

        # Initialize agent with LiteLLMModel
        agent = CodeActAgentSmolagents(model=litellm_model, temperature=0.5)

        # Model already built, so ModelBuilder.build should not be called
        assert agent.model_id == "watsonx/mistralai/mistral-medium-2505"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
