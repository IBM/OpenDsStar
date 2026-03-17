"""Unified model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel


@dataclass
class ModelConfig:
    """Configuration for building models."""

    model_id: str
    temperature: float = 0.0
    api_base: str | None = None
    api_key: str | None = None
    custom_llm_provider: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_string(cls, model_string: str, temperature: float = 0.0) -> ModelConfig:
        """Create config from model string."""
        return cls(model_id=model_string, temperature=temperature)

    @classmethod
    def from_langchain_model(
        cls, model: BaseChatModel, temperature: float = 0.0
    ) -> ModelConfig:
        """Extract config from LangChain BaseChatModel."""
        # Extract model_id
        model_id = (
            getattr(model, "model_id", None)
            or getattr(model, "model", None)
            or getattr(model, "model_name", None)
            or model.__class__.__name__
        )

        # Create config
        config = cls(model_id=model_id, temperature=temperature)

        # Extract from model_kwargs if present (ChatLiteLLM)
        if hasattr(model, "model_kwargs"):
            model_kwargs = getattr(model, "model_kwargs", {})
            if isinstance(model_kwargs, dict):
                for key, value in model_kwargs.items():
                    if value is not None and key != "cache":
                        extracted_value = cls._get_secret_value(value)
                        if extracted_value is not None:
                            config.extra_params[key] = extracted_value

        # Extract standard attributes (may override model_kwargs)
        for attr in ["api_base", "base_url", "openai_api_base"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if val:
                    config.api_base = cls._get_secret_value(val)
                    break

        for attr in ["api_key", "apikey", "openai_api_key"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if val:
                    config.api_key = cls._get_secret_value(val)
                    break

        for attr in ["custom_llm_provider", "provider"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if val:
                    config.custom_llm_provider = val
                    break

        return config

    @staticmethod
    def _get_secret_value(value: Any) -> str | None:
        """Extract string from SecretStr."""
        if value is None:
            return None
        if hasattr(value, "get_secret_value"):
            return value.get_secret_value()
        return str(value) if value else None
