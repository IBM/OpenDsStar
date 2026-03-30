"""
Recreatable base class for config-based object persistence.

This module provides a base class that allows objects to be saved and loaded
from JSON configuration files while maintaining normal Python __init__ signatures.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

T = TypeVar("T", bound="Recreatable")


class Recreatable:
    """
    Base class for objects that can be recreated from configuration.

    Subclasses automatically capture their __init__ arguments and can:
    - Export configuration as JSON-serializable dict
    - Save configuration with class identity to file
    - Load and recreate instances from saved configuration

    Usage:
        class MyClass(Recreatable):
            def __init__(self, arg1: int, arg2: str, scale: float = 1.0):
                self._capture_init_args(locals())  # FIRST LINE
                self.arg1 = arg1
                self.arg2 = arg2
                self.scale = scale
    """

    def __init_subclass__(cls) -> None:
        """Cache the __init__ signature for each subclass."""
        cls._init_signature = inspect.signature(cls.__init__)

    def _capture_init_args(self, locals_: Dict[str, Any]) -> None:
        """
        Capture constructor arguments from locals().

        Must be called as the FIRST line in __init__:
            self._capture_init_args(locals())

        Args:
            locals_: The locals() dict from __init__
        """
        sig = type(self)._init_signature
        # Filter out special variables that locals() includes in Python 3.11+
        filtered_locals = {
            k: v for k, v in locals_.items() if not k.startswith("__") and k != "self"
        }
        bound = sig.bind_partial(**filtered_locals)
        bound.apply_defaults()
        self._config = {k: v for k, v in bound.arguments.items() if k != "self"}

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dict for this instance.

        Returns:
            Dictionary of constructor arguments
        """
        if not hasattr(self, "_config"):
            raise RuntimeError(
                f"{self.__class__.__name__} did not call _capture_init_args() in __init__"
            )
        return dict(self._config)

    @classmethod
    def _type_id(cls) -> str:
        """
        Get the type identifier for this class.

        Returns:
            String in format "module.qualname"
        """
        return f"{cls.__module__}.{cls.__qualname__}"

    @staticmethod
    def _resolve_type(type_id: str) -> Type[Recreatable]:
        """
        Resolve a type identifier to a class.

        Args:
            type_id: Type identifier in format "module.qualname"

        Returns:
            The resolved class

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If class cannot be found
            TypeError: If class is not a Recreatable subclass
        """
        module_name, class_name = type_id.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not issubclass(cls, Recreatable):
            raise TypeError(f"{type_id} is not a Recreatable subclass")
        return cls

    def save_config(self, path: Path | str) -> None:
        """
        Save configuration to a JSON file.

        File format:
            {
                "type": "module.qualname",
                "args": {<constructor arguments>}
            }

        Args:
            path: Path to save configuration file
        """
        path = Path(path)
        config_data = {"type": self._type_id(), "args": self.get_config()}

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_instance(
        cls: Type[T], path: Path | str, allowed_types: Optional[List[str]] = None
    ) -> T:
        """
        Load an instance from a configuration file.

        Args:
            path: Path to configuration file
            allowed_types: Optional list of allowed type identifiers for security.
                          If provided, only these types can be loaded.

        Returns:
            Recreated instance

        Raises:
            ValueError: If type is not in allowed_types
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        type_id = config_data["type"]
        args = config_data["args"]

        # Security check
        if allowed_types is not None and type_id not in allowed_types:
            raise ValueError(f"Type {type_id} is not in allowed_types: {allowed_types}")

        # Convert agent_type string back to enum if needed (for experiments)
        if "agent_type" in args and isinstance(args["agent_type"], str):
            try:
                from OpenDsStar.experiments.implementations.agent_factory import AgentType

                args["agent_type"] = AgentType(args["agent_type"])
            except (ImportError, ValueError):
                # If AgentType can't be imported or value is invalid, leave as string
                pass

        # Resolve and instantiate
        target_cls = cls._resolve_type(type_id)
        return target_cls(**args)  # type: ignore[return-value]

    @staticmethod
    def load_config_dict(path: Path | str) -> Dict[str, Any]:
        """
        Load configuration dict from file without instantiating.

        Args:
            path: Path to configuration file

        Returns:
            Configuration dictionary with "type" and "args" keys
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
