"""
Benchmarks directory.

Each benchmark should be in its own subdirectory with the following structure:
- data_reader.py: Implements DataReader
- tools_builder.py: Implements ToolBuilder
- <benchmark>_main.py: Main entry point that inherits from BaseExperiment
- output/: Directory for benchmark results
- cache/: Directory for benchmark cache
"""

from ..base.base_experiment import BaseExperiment

__all__ = ["BaseExperiment"]
