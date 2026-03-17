"""
Unified model builder for LangChain and smolagents.

This module provides a simplified, unified interface for creating models
for both LangChain and smolagents frameworks.

Key features:
- Single entry point for all model building
- Automatic provider handling (e.g., tpm/ prefix)
- Centralized cache management
- Support for both string model IDs and pre-built model instances

Usage:
    LangChain models:
        >>> from pathlib import Path
        >>> model, model_id = ModelBuilder.build(
        ...     model="gpt-4o-mini",
        ...     temperature=0.0,
        ...     cache_dir=Path("./cache"),
        ...     framework="langchain"
        ... )

    Smolagents models:
        >>> model, model_id = ModelBuilder.build(
        ...     model="tpm/GCP/gemini-2.5-flash",
        ...     temperature=0.0,
        ...     cache_dir=Path("./cache"),
        ...     framework="smolagents"
        ... )

    Custom providers:
        >>> from agents.utils.providers import ProviderRegistry
        >>> from agents.utils.model_config import ModelConfig
        >>>
        >>> def my_transformer(config: ModelConfig) -> ModelConfig:
        ...     config.model_id = config.model_id.removeprefix("my_prefix/")
        ...     config.api_base = "https://my-api.com"
        ...     return config
        >>>
        >>> ProviderRegistry.register("my_prefix/", my_transformer)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

from agents.utils.cache_manager import CacheManager
from agents.utils.model_config import ModelConfig
from agents.utils.model_provider_registry import get_global_registry
from agents.utils.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)


class ModelBuilder:
    """Unified model builder for LangChain and smolagents."""

    @staticmethod
    def build(
        model: str | BaseChatModel,
        temperature: float = 0.0,
        cache_dir: Path | None = None,
        framework: Literal["langchain", "smolagents"] = "langchain",
    ) -> tuple[Any, str]:
        """
        Build a model for specified framework.

        Args:
            model: Model ID string (e.g., "gpt-4o-mini", "tpm/GCP/gemini-2.5-flash")
                or pre-built BaseChatModel instance
            temperature: Generation temperature (0.0-1.0)
            cache_dir: Cache directory path. If None, uses in-memory cache.
            framework: Target framework - "langchain" or "smolagents"

        Returns:
            Tuple of (model_instance, model_id)

        Raises:
            ValueError: If model type is invalid or framework is not recognized

        Examples:
            >>> # LangChain model with cache
            >>> model, model_id = ModelBuilder.build(
            ...     "gpt-4o-mini",
            ...     temperature=0.0,
            ...     cache_dir=Path("./cache"),
            ...     framework="langchain"
            ... )

            >>> # Smolagents model with custom API provider
            >>> model, model_id = ModelBuilder.build(
            ...     "custom_prefix/provider/model-name",
            ...     temperature=0.7,
            ...     cache_dir=Path("./cache"),
            ...     framework="smolagents"
            ... )
        """
        # 1. Create config from input
        if isinstance(model, str):
            model_string = model
            config = ModelConfig.from_string(model, temperature)
        elif isinstance(model, BaseChatModel):
            model_string = None
            config = ModelConfig.from_langchain_model(model, temperature)
        else:
            raise ValueError(
                f"model must be str or BaseChatModel, got {type(model).__name__}"
            )

        # 2. Try new ModelProviderRegistry first (for CustomAPIProvider, etc.)
        registry = get_global_registry()
        provider = registry.get_provider(config.model_id) if model_string else None

        if provider and framework == "langchain":
            # Use new provider system for LangChain
            cache = CacheManager.get_cache(cache_dir, config.model_id)
            model_instance, resolved_model_id = provider.build_model(
                config.model_id, temperature, cache
            )
            return model_instance, resolved_model_id

        # 3. Fall back to old ProviderRegistry transformer system
        config = ProviderRegistry.apply(config)

        # 4. Get cache instance
        cache = CacheManager.get_cache(cache_dir, config.model_id)

        # 5. Build framework-specific model
        if framework == "langchain":
            return ModelBuilder._build_langchain(config, cache)
        elif framework == "smolagents":
            return ModelBuilder._build_smolagents(config, cache_dir)
        else:
            raise ValueError(
                f"framework must be 'langchain' or 'smolagents', got {framework}"
            )

    @staticmethod
    def _build_langchain(config: ModelConfig, cache: Any) -> tuple[BaseChatModel, str]:
        """
        Build LangChain ChatLiteLLM model.

        Args:
            config: Model configuration
            cache: Cache instance to attach

        Returns:
            Tuple of (ChatLiteLLM instance, model_id)
        """
        model_kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "cache": cache,
        }

        # Add optional parameters
        if config.api_base:
            model_kwargs["api_base"] = config.api_base
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        if config.custom_llm_provider:
            model_kwargs["custom_llm_provider"] = config.custom_llm_provider

        # Add any extra parameters from config
        model_kwargs.update(config.extra_params)

        model = ChatLiteLLM(model=config.model_id, **model_kwargs)

        logger.debug(
            f"Built LangChain model: {config.model_id} "
            f"(api_base: {config.api_base}, provider: {config.custom_llm_provider})"
        )

        return model, config.model_id

    @staticmethod
    def _build_smolagents(
        config: ModelConfig, cache_dir: Path | None
    ) -> tuple[Any, str]:
        """
        Build smolagents LiteLLMModel.

        Args:
            config: Model configuration
            cache_dir: Cache directory for global LiteLLM cache configuration

        Returns:
            Tuple of (LiteLLMModel instance, model_id)

        Raises:
            ImportError: If smolagents is not installed
        """
        try:
            from smolagents import LiteLLMModel
        except ImportError:
            raise ImportError(
                "smolagents is required to build LiteLLMModel. "
                "Install it with: pip install smolagents"
            )

        # Configure global LiteLLM cache if cache_dir provided
        if cache_dir:
            from agents.utils.litellm_cache_config import configure_litellm_cache

            configure_litellm_cache(cache_dir)

        model_kwargs: dict[str, Any] = {
            "model_id": config.model_id,
            "temperature": config.temperature,
        }

        # Add optional parameters
        if config.api_base:
            model_kwargs["api_base"] = config.api_base
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        if config.custom_llm_provider:
            model_kwargs["custom_llm_provider"] = config.custom_llm_provider

        # Add any extra parameters from config
        model_kwargs.update(config.extra_params)

        model = LiteLLMModel(**model_kwargs)

        logger.debug(
            f"Built smolagents model: {config.model_id} "
            f"(api_base: {config.api_base}, provider: {config.custom_llm_provider})"
        )

        return model, config.model_id
