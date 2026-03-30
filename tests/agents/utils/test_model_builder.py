"""Tests for ModelBuilder utility class."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

from OpenDsStar.agents.utils.model_builder import ModelBuilder


class TestModelBuilder:
    """Test ModelBuilder class."""

    def test_build_with_string_standard_model(self):
        """Test building a model from a standard model string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, model_id = ModelBuilder.build(
                "gpt-4o-mini", temperature=0.5, cache_dir=Path(tmpdir)
            )

            assert isinstance(model, ChatLiteLLM)
            assert model_id == "gpt-4o-mini"
            assert model.temperature == 0.5

    def test_build_with_existing_model_instance(self):
        """Test building from an existing BaseChatModel instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_model = Mock(spec=BaseChatModel)
            existing_model.model = "test-model"

            model, model_id = ModelBuilder.build(existing_model, cache_dir=Path(tmpdir))

            # Model should be bound with cache, so it's not the same instance
            assert model_id == "test-model"

    def test_build_with_model_instance_no_model_attr(self):
        """Test building from model instance without model attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_model = Mock(spec=BaseChatModel)
            existing_model.__class__.__name__ = "MockChatModel"
            del existing_model.model  # Remove model attribute

            model, model_id = ModelBuilder.build(existing_model, cache_dir=Path(tmpdir))

            assert model_id == "MockChatModel"

    def test_build_with_invalid_type(self):
        """Test that invalid model type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="model must be str or BaseChatModel"):
                ModelBuilder.build(123, cache_dir=Path(tmpdir))  # type: ignore

    def test_build_default_temperature(self):
        """Test that default temperature is 0.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, _ = ModelBuilder.build("gpt-4o-mini", cache_dir=Path(tmpdir))
            assert isinstance(model, ChatLiteLLM)
            assert model.temperature == 0.0
