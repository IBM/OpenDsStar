"""
Custom API Provider - Generic provider for custom model endpoints.

This provider handles models with a configurable prefix, connecting to any
custom API endpoint. All configuration is read from environment variables,
making it suitable for open-source projects without hardcoded credentials.

Environment Variables:
    CUSTOM_API_PREFIX: Model prefix to handle (default: "custom")
    CUSTOM_API_BASE: Base URL for the API (required)
    CUSTOM_API_KEY: API key for authentication (required)
    CUSTOM_API_PROVIDER: LiteLLM provider type (default: "openai")
    CUSTOM_API_NAME: Display name for the provider (default: "CustomAPI")

Example:
    >>> # Set environment variables
    >>> os.environ["CUSTOM_API_PREFIX"] = "myapi"
    >>> os.environ["CUSTOM_API_BASE"] = "https://api.example.com"
    >>> os.environ["CUSTOM_API_KEY"] = "sk-..."
    >>>
    >>> # Register the provider
    >>> CustomAPIProvider.register()
    >>>
    >>> # Now ModelBuilder can handle myapi/ models
    >>> model, model_id = ModelBuilder.build("myapi/my-model", ...)
"""

from __future__ import annotations

import logging
import os

from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

from OpenDsStar.agents.utils.model_provider import ModelProvider

logger = logging.getLogger(__name__)

_custom_api_registered = False


class CustomAPIProvider(ModelProvider):
    """
    Generic provider for custom API endpoints.

    All configuration is read from environment variables, making this
    provider suitable for any custom API without hardcoding credentials.

    Configuration via environment variables:
        - CUSTOM_API_PREFIX: Model prefix (default: "custom")
        - CUSTOM_API_BASE: API base URL (required)
        - CUSTOM_API_KEY: API key (required)
        - CUSTOM_API_PROVIDER: LiteLLM provider type (default: "openai")
        - CUSTOM_API_NAME: Display name (default: "CustomAPI")
    """

    def __init__(
        self,
        prefix: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        custom_llm_provider: str | None = None,
        name: str | None = None,
    ):
        """
        Initialize Custom API provider.

        All parameters are optional and will be read from environment
        variables if not provided.

        Args:
            prefix: Model prefix to handle (from CUSTOM_API_PREFIX)
            api_base: Base URL for API (from CUSTOM_API_BASE)
            api_key: API key (from CUSTOM_API_KEY)
            custom_llm_provider: LiteLLM provider type (from CUSTOM_API_PROVIDER)
            name: Display name (from CUSTOM_API_NAME)
        """
        # Read from environment variables with fallbacks
        self._prefix = prefix or os.getenv("CUSTOM_API_PREFIX", "custom")
        self._api_base = api_base or os.getenv("CUSTOM_API_BASE")
        self._api_key = api_key or os.getenv("CUSTOM_API_KEY")
        self._custom_llm_provider = custom_llm_provider or os.getenv(
            "CUSTOM_API_PROVIDER", "openai"
        )
        self._name = name or os.getenv("CUSTOM_API_NAME", "CustomAPI")

        # Validate required configuration
        if not self._api_base:
            logger.warning(
                "CUSTOM_API_BASE not set. Provider will fail if used. "
                "Set CUSTOM_API_BASE environment variable."
            )

    @property
    def name(self) -> str:
        """Provider name."""
        return self._name

    def can_handle(self, model_string: str) -> bool:
        """Check if model string has the configured prefix."""
        return model_string.startswith(f"{self._prefix}/")

    def build_model(
        self,
        model_string: str,
        temperature: float,
        cache: BaseCache,
    ) -> tuple[BaseChatModel, str]:
        """
        Build a custom API model instance.

        Args:
            model_string: Model string with configured prefix
            temperature: Temperature for generation
            cache: Cache instance

        Returns:
            Tuple of (model_instance, resolved_model_id)

        Raises:
            ValueError: If model string format is invalid or config is missing
        """
        expected_prefix = f"{self._prefix}/"
        if not model_string.startswith(expected_prefix):
            raise ValueError(
                f"{self.name} provider requires '{expected_prefix}' prefix, "
                f"got: {model_string}"
            )

        # Validate API base is configured
        if not self._api_base:
            raise ValueError(
                f"{self.name} provider requires CUSTOM_API_BASE to be set. "
                "Please set the environment variable."
            )

        # Remove prefix to get actual model ID
        prefix_len = len(expected_prefix)
        model_id = model_string[prefix_len:]

        # Get API key
        api_key = self._api_key

        if not api_key:
            logger.warning(
                "No API key configured for %s provider. "
                "Set CUSTOM_API_KEY environment variable. "
                "Model may fail to authenticate.",
                self.name,
            )

        # Build model with custom configuration
        model_instance = ChatLiteLLM(
            model=model_id,
            temperature=temperature,
            api_base=self._api_base,
            api_key=api_key,
            custom_llm_provider=self._custom_llm_provider,
            cache=cache,
        )

        logger.debug("Built %s model: %s", self.name, model_id)
        return model_instance, model_id

    @staticmethod
    def register() -> None:
        """
        Register Custom API provider with the global registry.

        Reads configuration from environment variables:
        - CUSTOM_API_PREFIX: Model prefix (default: "custom")
        - CUSTOM_API_BASE: API base URL (required)
        - CUSTOM_API_KEY: API key (required)
        - CUSTOM_API_PROVIDER: LiteLLM provider type (default: "openai")
        - CUSTOM_API_NAME: Display name (default: "CustomAPI")

        This function is idempotent - calling it multiple times is safe.

        Example:
            >>> # Set environment variables first
            >>> os.environ["CUSTOM_API_BASE"] = "https://api.example.com"
            >>> os.environ["CUSTOM_API_KEY"] = "sk-..."
            >>>
            >>> from OpenDsStar.agents.utils.providers.custom_api_provider import CustomAPIProvider
            >>> CustomAPIProvider.register()
            >>> # Now ModelBuilder can handle custom/ models
        """
        global _custom_api_registered

        if _custom_api_registered:
            logger.debug("Custom API provider already registered, skipping")
            return

        from OpenDsStar.agents.utils.model_provider_registry import get_global_registry

        registry = get_global_registry()
        provider = CustomAPIProvider()

        # Warn if not properly configured
        if not provider._api_base:
            logger.warning(
                "Registering Custom API provider without CUSTOM_API_BASE set. "
                "Provider will fail if used. Set environment variable before use."
            )

        registry.register(provider)

        _custom_api_registered = True
        logger.info(
            "Custom API provider registered (prefix: %s, name: %s)",
            provider._prefix,
            provider.name,
        )


# Made with Bob
