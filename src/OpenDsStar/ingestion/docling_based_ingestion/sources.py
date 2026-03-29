"""Source file handling utilities for docling-based ingestion."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class SourceFile:
    """Represents a source file for processing."""

    display_name: str
    path_hint: str
    stream_factory: Optional[Callable[[], bytes]] = None
    temp_path: Optional[str] = None


class TempMaterializer:
    """Manages temporary files for stream-based sources."""

    def __init__(self):
        self._temp_files: list[Path] = []

    def materialize(self, path_hint: str, stream_factory: Callable[[], bytes]) -> str:
        """
        Materialize a stream to a temporary file.

        Args:
            path_hint: Hint for the file extension
            stream_factory: Function that returns file bytes

        Returns:
            Path to the temporary file
        """
        suffix = Path(path_hint).suffix or ".tmp"
        temp_file = tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False)
        try:
            data = stream_factory()
            temp_file.write(data)
            temp_file.flush()
            temp_path = Path(temp_file.name)
            self._temp_files.append(temp_path)
            return str(temp_path)
        finally:
            temp_file.close()

    def cleanup(self):
        """Remove all temporary files."""
        for temp_path in self._temp_files:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
        self._temp_files.clear()
