"""Docling-based document ingestion and description generation."""

from .batching import iter_batches
from .docling_converter import DoclingConverter, FallbackTextDoc
from .docling_description_builder import DoclingDescriptionBuilder
from .markdown_shortener import MarkdownShortener
from .prompts import build_file_description_prompt
from .sources import SourceFile, TempMaterializer

__all__ = [
    "DoclingConverter",
    "DoclingDescriptionBuilder",
    "MarkdownShortener",
    "FallbackTextDoc",
    "SourceFile",
    "TempMaterializer",
    "build_file_description_prompt",
    "iter_batches",
]
