"""
Tests for the ReactAgentLangchain.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from OpenDsStar.agents import ReactAgentLangchain


class TestReactAgentLangchain:
    """Test suite for ReactAgentLangchain."""

    def test_initialization_with_string_model(self):
        """Test initialization with a model string."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        agent = ReactAgentLangchain(model=mock_model, temperature=0.5)
        assert agent.temperature == 0.5
        assert agent.max_steps == 5
        assert agent.tools == []

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        agent = ReactAgentLangchain(
            model=mock_model,
            temperature=0.7,
            max_steps=10,
            system_prompt="Custom prompt",
        )
        assert agent.temperature == 0.7
        assert agent.max_steps == 10
        assert agent.system_prompt == "Custom prompt"

    @patch("OpenDsStar.agents.react_langchain.react_agent_langchain.create_react_agent")
    def test_initialization_with_tools(self, mock_create_react):
        """Test initialization with custom tools."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return "result"

        mock_model = FakeChatModel()
        mock_graph = Mock()
        mock_create_react.return_value = mock_graph

        agent = ReactAgentLangchain(model=mock_model, tools=[test_tool])
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "test_tool"

    def test_invalid_model_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="model must be a LangChain BaseChatModel"):
            ReactAgentLangchain(model=123)  # type: ignore

    def test_invoke_with_empty_query(self):
        """Test that empty query raises ValueError."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()
        agent = ReactAgentLangchain(model=mock_model)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.invoke("")

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.invoke(None)  # type: ignore

    def test_invoke_basic(self):
        """Test basic invoke functionality."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        # Create agent
        agent = ReactAgentLangchain(model=mock_model)

        # Mock the graph invoke
        mock_response = AIMessage(content="The answer is 42", tool_calls=[])
        agent._graph.invoke = Mock(
            return_value={
                "messages": [HumanMessage(content="What is the answer?"), mock_response]
            }
        )

        # Test invoke
        result = agent.invoke("What is the answer?")

        assert "answer" in result
        assert "messages" in result
        assert "trajectory" in result
        assert "steps_used" in result
        assert result["answer"] == "The answer is 42"

    def test_invoke_with_tool_calls(self):
        """Test invoke with tool calls in trajectory."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        # Create agent
        agent = ReactAgentLangchain(model=mock_model)

        # Mock the graph invoke with tool calls
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "calculator", "args": {"a": 2, "b": 3}, "id": "call_123"}
            ],
        )
        tool_result_msg = ToolMessage(
            content="5", name="calculator", tool_call_id="call_123"
        )
        final_msg = AIMessage(content="The result is 5", tool_calls=[])

        agent._graph.invoke = Mock(
            return_value={
                "messages": [
                    HumanMessage(content="What is 2+3?"),
                    tool_call_msg,
                    tool_result_msg,
                    final_msg,
                ]
            }
        )

        # Test invoke
        result = agent.invoke("What is 2+3?")

        assert result["answer"] == "The result is 5"
        assert len(result["trajectory"]) == 4
        assert result["trajectory"][1]["type"] == "tool_call"
        assert result["trajectory"][2]["type"] == "tool_result"

    def test_invoke_with_custom_config(self):
        """Test invoke with custom configuration."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        # Create agent
        agent = ReactAgentLangchain(model=mock_model)

        # Mock the graph invoke
        agent._graph.invoke = Mock(
            return_value={
                "messages": [
                    HumanMessage(content="Test"),
                    AIMessage(content="Response", tool_calls=[]),
                ]
            }
        )

        # Test invoke with custom config
        custom_config = {
            "configurable": {"thread_id": "test_thread"},
            "recursion_limit": 50,
        }
        _ = agent.invoke("Test query", config=custom_config)

        # Verify the config was passed
        agent._graph.invoke.assert_called_once()
        call_args = agent._graph.invoke.call_args
        assert call_args[1]["config"]["recursion_limit"] == 50

    def test_invoke_return_state(self):
        """Test invoke with return_state=True."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        # Create agent
        agent = ReactAgentLangchain(model=mock_model)

        # Mock the graph invoke
        mock_state = {
            "messages": [
                HumanMessage(content="Test"),
                AIMessage(content="Response", tool_calls=[]),
            ],
            "is_last_step": False,
        }
        agent._graph.invoke = Mock(return_value=mock_state)

        # Test invoke with return_state=True
        result = agent.invoke("Test query", return_state=True)

        # Should return the raw state
        assert result == mock_state
        assert "is_last_step" in result

    def test_graph_building(self):
        """Test that the graph is built correctly."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()
        agent = ReactAgentLangchain(model=mock_model)

        # Verify graph exists
        assert hasattr(agent, "_graph")
        assert agent._graph is not None

    def test_system_prompt_with_time_placeholder(self):
        """Test that system prompt with {system_time} is formatted."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        agent = ReactAgentLangchain(
            model=mock_model,
            system_prompt="Current time: {system_time}",
        )

        # The system prompt should contain the placeholder
        assert "{system_time}" in agent.system_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
