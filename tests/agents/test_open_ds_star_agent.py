"""Tests for OpenDsStarAgent - meaningful tests only."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field

from OpenDsStar.agents.ds_star.open_ds_star_agent import OpenDsStarAgent


class TestOpenDsStarAgentValidation:
    """Test agent validation and error handling."""

    def test_initialization_invalid_model_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="model must be a LangChain BaseChatModel"):
            OpenDsStarAgent(model=123)  # type: ignore

    def test_initialization_invalid_code_mode(self):
        """Test that invalid code_mode raises ValueError."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        with pytest.raises(
            ValueError, match=r"code_mode must be either 'stepwise' or 'full'"
        ):
            OpenDsStarAgent(model=mock_model, code_mode="invalid")

    def test_initialization_wrong_case_code_mode(self):
        """Test that wrong case code_mode raises ValueError."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        with pytest.raises(
            ValueError, match=r"code_mode must be either 'stepwise' or 'full'"
        ):
            OpenDsStarAgent(model=mock_model, code_mode="STEPWISE")


class TestOpenDsStarAgentInvoke:
    """Test agent invoke method - business logic and error handling."""

    @patch("OpenDsStar.agents.ds_star.open_ds_star_agent.DSStarGraph")
    def test_invoke_with_empty_query(self, mock_graph_class):
        """Test that empty query raises ValueError."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = OpenDsStarAgent(model=mock_model, cache_dir=Path(tmpdir))
            with pytest.raises(ValueError, match="non-empty string"):
                agent.invoke("")

    @patch("OpenDsStar.agents.ds_star.open_ds_star_agent.DSStarGraph")
    def test_invoke_with_none_query(self, mock_graph_class):
        """Test that None query raises ValueError."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = OpenDsStarAgent(model=mock_model, cache_dir=Path(tmpdir))
            with pytest.raises(ValueError, match="non-empty string"):
                agent.invoke(None)  # type: ignore

    @patch("OpenDsStar.agents.ds_star.open_ds_star_agent.DSStarGraph")
    def test_invoke_with_tool_update_description(self, mock_graph_class):
        """Test invoke updates tool descriptions if available."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_graph = Mock()
            mock_graph_class.return_value = mock_graph
            mock_graph.invoke.return_value = {
                "user_query": "Test query",
                "tools": {},
                "final_answer": "Test answer",
                "steps_used": 2,
                "trajectory": [],
                "token_usage": [],
                "fatal_error": None,
                "steps": [],
                "max_steps": 5,
                "code_mode": "stepwise",
            }

            # Create tool with update_description method
            class MockToolInput(BaseModel):
                query: str = Field(description="Test query parameter")

            mock_tool = Mock()
            mock_tool.name = "test_tool"
            mock_tool.description = "Test tool"
            mock_tool.args_schema = MockToolInput
            mock_tool.update_description = Mock()

            agent = OpenDsStarAgent(
                model=mock_model,
                tools=[mock_tool],
                cache_dir=Path(tmpdir),
            )
            agent.invoke("Test query")

            mock_tool.update_description.assert_called_once_with("Test query")
            mock_graph.update_tools_spec.assert_called_once()

    @patch("OpenDsStar.agents.ds_star.open_ds_star_agent.DSStarGraph")
    def test_invoke_error_handling(self, mock_graph_class):
        """Test that invoke propagates exceptions."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_graph = Mock()
            mock_graph_class.return_value = mock_graph
            mock_graph.invoke.side_effect = Exception("Test error")

            agent = OpenDsStarAgent(model=mock_model, cache_dir=Path(tmpdir))
            with pytest.raises(Exception, match="Test error"):
                agent.invoke("Test query")
