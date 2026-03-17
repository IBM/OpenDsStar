"""
Model Provider Interface - Extensible model building system.

This module defines the interface for custom model providers that can be
registered with ModelBuilder to handle specific model prefixes or types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel


class ModelProvider(ABC):
    """
    Abstract base class for model providers.

    Custom providers can implement this interface to add support for
    specific model types or providers (e.g., custom APIs, proprietary endpoints, etc.).
    """

    @abstractmethod
    def can_handle(self, model_string: str) -> bool:
        """
        Check if this provider can handle the given model string.

        Args:
            model_string: Model identifier string (e.g., "tpm/GCP/gemini-2.5-flash")

        Returns:
            True if this provider can build this model, False otherwise
        """
        pass

    @abstractmethod
    def build_model(
        self,
        model_string: str,
        temperature: float,
        cache: BaseCache,
    ) -> tuple[BaseChatModel, str]:
        """
        Build a model instance from the model string.

        Args:
            model_string: Model identifier string
            temperature: Temperature for generation
            cache: Cache instance to attach to the model

        Returns:
            Tuple of (model_instance, resolved_model_id)

        Raises:
            ValueError: If model cannot be built
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provider name for logging and debugging.

        Returns:
            Human-readable provider name
        """
        pass
