"""Logging utilities."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def setup_logging(level: int = logging.INFO) -> None:
    """
    Setup logging configuration for experiments.

    Configures the root logger to output to stdout with a simple format.
    This ensures that INFO level messages from data readers and other
    components are visible.

    Args:
        level: Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,  # Override any existing configuration
    )


def setup_logging_with_file(
    output_dir: Path,
    log_filename: str,
    level: int = logging.INFO,
) -> None:
    """
    Setup logging configuration for experiments with both console and file output.

    Configures the root logger to output to both stdout and a file in the output directory.
    This ensures that INFO level messages from data readers and other components are
    visible in both console and saved to a log file.

    Args:
        output_dir: Directory where log file will be saved
        log_filename: Name of the log file (e.g., "result_ds_star_hotpotqa_20260127_161619_log.txt")
        level: Logging level (default: logging.INFO)
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / log_filename

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Log the log file location
    root_logger.info(f"Logging to file: {log_file}")


class StdoutLogger:
    """Simple stdout logger implementation."""

    def info(self, msg: str, **kwargs: Any) -> None:
        print(f"[INFO] {msg} {kwargs if kwargs else ''}".rstrip())

    def warning(self, msg: str, **kwargs: Any) -> None:
        print(f"[WARN] {msg} {kwargs if kwargs else ''}".rstrip())

    def error(self, msg: str, **kwargs: Any) -> None:
        print(f"[ERROR] {msg} {kwargs if kwargs else ''}".rstrip())


@dataclass(frozen=True)
class StageTimer:
    """Context manager for timing pipeline stages."""

    logger: logging.Logger
    stage: str
    start: float = field(default_factory=time.time)

    def __enter__(self) -> StageTimer:
        self.logger.info(f"stage_start:{self.stage}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        dur = time.time() - self.start
        if exc is None:
            self.logger.info(f"stage_end:{self.stage} seconds={dur}")
            return False
        self.logger.error(f"stage_fail:{self.stage} seconds={dur} error={str(exc)}")
        return False  # don't suppress exceptions
