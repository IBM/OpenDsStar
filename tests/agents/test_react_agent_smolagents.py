"""
Tests for the ReactAgentSmolagents.
"""

from unittest.mock import Mock, patch

import pytest
from smolagents import RunResult

from OpenDsStar.agents import ReactAgentSmolagents


class TestReactAgentSmolagents:
    """Test suite for ReactAgentSmolagents."""

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_initialization_with_string_model(self, mock_tool_agent):
        """Test initialization with a model string."""
        from smolagents import LiteLLMModel

        mock_model_instance = LiteLLMModel(model_id="gpt-4o-mini")
        mock_agent_instance = Mock()
        mock_tool_agent.return_value = mock_agent_instance

        agent = ReactAgentSmolagents(model=mock_model_instance, temperature=0.5)

        assert agent.model_id == "gpt-4o-mini"
        assert agent.temperature == 0.5
        assert agent.max_steps == 5
        assert agent.tools == []

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_initialization_with_custom_params(self, mock_tool_agent):
        """Test initialization with custom parameters."""
        from smolagents import LiteLLMModel

        mock_model_instance = LiteLLMModel(model_id="gpt-4o")
        mock_agent_instance = Mock()
        mock_tool_agent.return_value = mock_agent_instance

        agent = ReactAgentSmolagents(
            model=mock_model_instance,
            temperature=0.7,
            max_steps=10,
            system_prompt="Custom prompt",
        )

        assert agent.model_id == "gpt-4o"
        assert agent.temperature == 0.7
        assert agent.max_steps == 10
        assert agent.system_prompt == "Custom prompt"

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_initialization_with_tools(self, mock_tool_agent):
        """Test initialization with custom tools."""
        from langchain_core.tools import tool
        from smolagents import LiteLLMModel

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return "result"

        mock_model_instance = LiteLLMModel(
            model_id="watsonx/mistralai/mistral-medium-2505"
        )
        mock_agent_instance = Mock()
        mock_tool_agent.return_value = mock_agent_instance

        agent = ReactAgentSmolagents(model=mock_model_instance, tools=[test_tool])
        assert len(agent.tools) == 1

    def test_invalid_model_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="model must be a smolagents LiteLLMModel"):
            ReactAgentSmolagents(model=123)  # type: ignore

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_invoke_with_empty_query(self, mock_tool_agent):
        """Test that empty query raises ValueError."""
        from smolagents import LiteLLMModel

        mock_model_instance = LiteLLMModel(
            model_id="watsonx/mistralai/mistral-medium-2505"
        )
        mock_agent_instance = Mock()
        mock_tool_agent.return_value = mock_agent_instance

        agent = ReactAgentSmolagents(model=mock_model_instance)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.invoke("")

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.invoke(None)  # type: ignore

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_invoke_basic(self, mock_tool_agent):
        """Test basic invoke functionality."""
        from smolagents import LiteLLMModel

        mock_model_instance = LiteLLMModel(model_id="gpt-4o-mini")

        mock_agent_instance = Mock()
        # Mock RunResult object
        mock_run_result = Mock(spec=RunResult)
        mock_run_result.output = "The answer is 42"
        mock_run_result.steps = ["Step 1: Thinking", "Step 2: Reasoning"]
        mock_agent_instance.run.return_value = mock_run_result
        mock_tool_agent.return_value = mock_agent_instance

        agent = ReactAgentSmolagents(model=mock_model_instance)

        result = agent.invoke("What is the answer?")

        assert "answer" in result
        assert "trajectory" in result
        assert "steps_used" in result
        assert result["answer"] == "The answer is 42"
        assert len(result["trajectory"]) == 2
        assert result["steps_used"] == 2

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_invoke_with_error(self, mock_tool_agent):
        """Test invoke when agent raises an error."""
        from smolagents import LiteLLMModel

        mock_model_instance = LiteLLMModel(model_id="gpt-4o-mini")

        mock_agent_instance = Mock()
        mock_agent_instance.run.side_effect = Exception("Test error")
        mock_tool_agent.return_value = mock_agent_instance

        agent = ReactAgentSmolagents(model=mock_model_instance)

        result = agent.invoke("What is the answer?")

        assert result["answer"] == ""
        assert result["fatal_error"] == "Test error"
        assert result["execution_error"] == "Test error"
        assert result["steps_used"] == 0
        assert result["verifier_sufficient"] is False

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_invoke_return_format(self, mock_tool_agent):
        """Test that invoke returns the correct format."""
        from smolagents import LiteLLMModel

        mock_model_instance = LiteLLMModel(model_id="gpt-4o-mini")

        mock_agent_instance = Mock()
        # Mock RunResult object
        mock_run_result = Mock(spec=RunResult)
        mock_run_result.output = "Result"
        mock_run_result.steps = []
        mock_agent_instance.run.return_value = mock_run_result
        mock_tool_agent.return_value = mock_agent_instance

        agent = ReactAgentSmolagents(model=mock_model_instance)

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

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_system_prompt_override(self, mock_tool_agent):
        """Test that system prompt can be overridden."""
        from smolagents import LiteLLMModel

        mock_model_instance = LiteLLMModel(model_id="gpt-4o-mini")

        mock_agent_instance = Mock()
        mock_agent_instance.system_prompt = None
        mock_tool_agent.return_value = mock_agent_instance

        custom_prompt = "Custom system prompt"
        agent = ReactAgentSmolagents(
            model=mock_model_instance, system_prompt=custom_prompt
        )

        assert agent.system_prompt == custom_prompt
        # Verify the agent's system_prompt was set
        assert mock_agent_instance.system_prompt == custom_prompt

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_invoke_with_empty_steps(self, mock_tool_agent):
        """Test invoke when RunResult has empty steps."""
        from smolagents import LiteLLMModel

        mock_model_instance = LiteLLMModel(model_id="gpt-4o-mini")
        mock_agent_instance = Mock()
        # Mock RunResult object with empty steps
        mock_run_result = Mock(spec=RunResult)
        mock_run_result.output = "Result"
        mock_run_result.steps = []
        mock_agent_instance.run.return_value = mock_run_result
        mock_tool_agent.return_value = mock_agent_instance

        agent = ReactAgentSmolagents(model=mock_model_instance)

        result = agent.invoke("Test query")

        # Should still work, with default steps_used
        assert result["answer"] == "Result"
        assert result["steps_used"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
