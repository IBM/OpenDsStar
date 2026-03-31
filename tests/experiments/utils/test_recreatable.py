"""Tests for the Recreatable base class."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from OpenDsStar.experiments.utils.recreatable import Recreatable


class SimpleClass(Recreatable):
    """Simple test class with basic types."""

    def __init__(self, arg1: int, arg2: str, arg3: float = 1.0):
        self._capture_init_args(locals())
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3


class ComplexClass(Recreatable):
    """Test class with more complex types."""

    def __init__(
        self, name: str, values: list[int], config: dict, optional: str | None = None
    ):
        self._capture_init_args(locals())
        self.name = name
        self.values = values
        self.config = config
        self.optional = optional


def test_capture_init_args():
    """Test that init args are captured correctly."""
    obj = SimpleClass(arg1=42, arg2="test", arg3=2.5)

    config = obj.get_config()
    assert config == {"arg1": 42, "arg2": "test", "arg3": 2.5}


def test_capture_with_defaults():
    """Test that default values are captured."""
    obj = SimpleClass(arg1=10, arg2="hello")

    config = obj.get_config()
    assert config == {"arg1": 10, "arg2": "hello", "arg3": 1.0}


def test_save_config():
    """Test saving configuration to file."""
    obj = SimpleClass(arg1=100, arg2="world", arg3=3.14)

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        obj.save_config(config_path)

        # Verify file exists and contains correct data
        assert config_path.exists()

        with open(config_path, "r") as f:
            data = json.load(f)

        # Type ID will be test_recreatable.SimpleClass when run via pytest
        assert data["type"].endswith("SimpleClass")
        assert "test_recreatable" in data["type"]
        assert data["args"] == {"arg1": 100, "arg2": "world", "arg3": 3.14}


def test_load_instance():
    """Test loading instance from configuration file."""
    original = SimpleClass(arg1=42, arg2="test", arg3=2.5)

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        original.save_config(config_path)

        # Load the instance
        loaded = SimpleClass.load_instance(config_path)

        # Verify it's a new instance with same config
        assert loaded is not original
        assert isinstance(loaded, SimpleClass)
        assert loaded.arg1 == 42
        assert loaded.arg2 == "test"
        assert loaded.arg3 == 2.5
        assert loaded.get_config() == original.get_config()


def test_load_instance_with_complex_types():
    """Test loading instance with complex types."""
    original = ComplexClass(
        name="test",
        values=[1, 2, 3],
        config={"key": "value", "nested": {"a": 1}},
        optional="optional_value",
    )

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        original.save_config(config_path)

        loaded = ComplexClass.load_instance(config_path)

        assert loaded.name == "test"
        assert loaded.values == [1, 2, 3]
        assert loaded.config == {"key": "value", "nested": {"a": 1}}
        assert loaded.optional == "optional_value"


def test_load_instance_with_allowed_types():
    """Test security feature with allowed types."""
    obj = SimpleClass(arg1=1, arg2="test")

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        obj.save_config(config_path)

        # Get the actual type ID from the saved config
        import json

        with open(config_path, "r") as f:
            config_data = json.load(f)
        actual_type = config_data["type"]

        # Should succeed with correct type
        loaded = SimpleClass.load_instance(config_path, allowed_types=[actual_type])
        assert isinstance(loaded, SimpleClass)

        # Should fail with wrong type
        wrong_types = ["some.other.Class"]
        with pytest.raises(ValueError, match="not in allowed_types"):
            SimpleClass.load_instance(config_path, allowed_types=wrong_types)


def test_type_id():
    """Test type identifier generation."""
    obj = SimpleClass(arg1=1, arg2="test")
    type_id = obj._type_id()

    # Type ID should end with the class name
    assert type_id.endswith("SimpleClass")
    assert "test_recreatable" in type_id


def test_resolve_type():
    """Test type resolution from identifier."""
    # Get the actual type ID
    obj = SimpleClass(arg1=1, arg2="test")
    type_id = obj._type_id()

    cls = Recreatable._resolve_type(type_id)

    assert cls.__name__ == "SimpleClass"
    assert issubclass(cls, Recreatable)


def test_resolve_type_invalid():
    """Test type resolution with invalid identifier."""
    with pytest.raises(ModuleNotFoundError):
        Recreatable._resolve_type("nonexistent.module.Class")

    with pytest.raises(AttributeError):
        Recreatable._resolve_type(
            "OpenDsStar.experiments.utils.recreatable.NonexistentClass"
        )


def test_missing_capture_init_args():
    """Test error when _capture_init_args is not called."""

    class BadClass(Recreatable):
        def __init__(self, arg: int):
            # Forgot to call _capture_init_args!
            self.arg = arg

    obj = BadClass(arg=42)

    with pytest.raises(RuntimeError, match="did not call _capture_init_args"):
        obj.get_config()


def test_load_config_dict():
    """Test loading config dict without instantiating."""
    obj = SimpleClass(arg1=42, arg2="test")

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        obj.save_config(config_path)

        config_dict = Recreatable.load_config_dict(config_path)

        # Type ID should end with the class name
        assert config_dict["type"].endswith("SimpleClass")
        assert "test_recreatable" in config_dict["type"]
        assert config_dict["args"] == {"arg1": 42, "arg2": "test", "arg3": 1.0}


def test_nested_directory_creation():
    """Test that save_config creates nested directories."""
    obj = SimpleClass(arg1=1, arg2="test")

    with TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "a" / "b" / "c" / "config.json"
        obj.save_config(nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()
