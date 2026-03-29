"""Cache utilities for docling analyzer."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from experiments.utils.cache import FileCache

logger = logging.getLogger(__name__)


class DoclingAnalysisCache:
    """
    Cache for file -> docling analysis results.

    Caches the markdown conversion output from docling for each file.
    Cache key is based on file path and file modification time to detect changes.

    Cache structure:
    ingestion/cache/docling_analysis_<hash>/

    Where hash is computed from:
    - max_content_length
    - max_table_rows
    - max_list_items
    - max_fallback_bytes
    """

    def __init__(
        self,
        cache_base_dir: Path | str,
        max_content_length: int = 50000,
        max_table_rows: int = 50,
        max_list_items: int = 100,
        max_fallback_bytes: int = 2_000_000,
        enabled: bool = True,
    ):
        self.cache_base_dir = Path(cache_base_dir)
        self.enabled = enabled

        # Parameters that affect the analysis result
        self.max_content_length = max_content_length
        self.max_table_rows = max_table_rows
        self.max_list_items = max_list_items
        self.max_fallback_bytes = max_fallback_bytes

        if self.enabled:
            self.cache_base_dir.mkdir(parents=True, exist_ok=True)
            cache_key = self._generate_cache_key()
            self.cache_path = self.cache_base_dir / f"docling_analysis_{cache_key}"
            self.cache_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Docling analysis cache initialized at: {self.cache_path}")

    def _generate_cache_key(self) -> str:
        """
        Generate cache key based on parameters that affect analysis results.

        Returns:
            Hash string for cache directory name
        """
        params_str = (
            f"content{self.max_content_length}_"
            f"table{self.max_table_rows}_"
            f"list{self.max_list_items}_"
            f"fallback{self.max_fallback_bytes}"
        )
        hash_obj = hashlib.md5(params_str.encode())
        return hash_obj.hexdigest()[:16]

    def _get_file_cache_key(self, file_path: Path) -> str:
        """
        Generate cache key for a specific file.

        Includes file path and modification time to detect changes.

        Args:
            file_path: Path to the file

        Returns:
            Cache key string
        """
        abs_path = str(file_path.resolve())
        try:
            mtime = file_path.stat().st_mtime
        except Exception:
            mtime = 0

        key_str = f"{abs_path}_{mtime}"
        hash_obj = hashlib.sha256(key_str.encode())
        return hash_obj.hexdigest()

    def get(self, file_path: Path) -> Tuple[str, Dict[str, Any]] | None:
        """
        Retrieve cached analysis result for a file.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (markdown_content, stats) if cached, None otherwise
        """
        if not self.enabled:
            return None

        cache_key = self._get_file_cache_key(file_path)

        try:
            with FileCache(self.cache_path) as cache:
                result = cache.get(cache_key)
                if result is not None:
                    logger.debug(f"Cache hit for {file_path.name}")
                    return result
        except Exception as e:
            logger.warning(f"Failed to load cache for {file_path.name}: {e}")

        return None

    def put(
        self,
        file_path: Path,
        markdown_content: str,
        stats: Dict[str, Any],
    ) -> None:
        """
        Store analysis result in cache.

        Args:
            file_path: Path to the file
            markdown_content: Markdown content from docling
            stats: Statistics about the analysis (truncation info, etc.)
        """
        if not self.enabled:
            return

        cache_key = self._get_file_cache_key(file_path)

        try:
            with FileCache(self.cache_path) as cache:
                cache.put(cache_key, (markdown_content, stats))
            logger.debug(f"Cached analysis for {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to cache analysis for {file_path.name}: {e}")

    def clear(self) -> None:
        """Clear all cached analysis results."""
        if not self.enabled:
            return

        try:
            with FileCache(self.cache_path) as cache:
                cache.clear()
            logger.info("Cleared docling analysis cache")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")


class FileDescriptionCache:
    """
    Cache for docling analysis -> file description results.

    Caches the LLM-generated descriptions for each file's markdown content.
    Cache key is based on the markdown content hash, LLM model, and prompt template.

    Cache structure:
    ingestion/cache/file_descriptions_<model_name>_<prompt_hash>/

    Where model_name is the sanitized LLM model name and prompt_hash is a hash
    of the prompt template to ensure cache invalidation when prompts change.
    """

    def __init__(
        self,
        cache_base_dir: Path | str,
        llm_model: str,
        prompt_template: str = "",
        enabled: bool = True,
    ):
        self.cache_base_dir = Path(cache_base_dir)
        self.enabled = enabled

        # Parameters that affect the description result
        self.llm_model = llm_model
        self.prompt_template = prompt_template

        if self.enabled:
            self.cache_base_dir.mkdir(parents=True, exist_ok=True)

            # Use model name directly in cache path for better readability
            safe_model_name = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in self.llm_model
            )

            # Hash the prompt template to detect changes
            prompt_hash = ""
            if prompt_template:
                prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:8]
                cache_dir_name = f"file_descriptions_{safe_model_name}_{prompt_hash}"
            else:
                cache_dir_name = f"file_descriptions_{safe_model_name}"

            self.cache_path = self.cache_base_dir / cache_dir_name
            self.cache_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"File description cache initialized at: {self.cache_path}")

    def _get_content_cache_key(self, doc_name: str, markdown_content: str) -> str:
        """
        Generate cache key for specific content.

        Based on document name and markdown content hash.

        Args:
            doc_name: Name of the document
            markdown_content: Markdown content to generate description for

        Returns:
            Cache key string
        """
        # Hash the content to create a stable key
        content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:32]
        # Include doc name for readability
        safe_name = "".join(c if c.isalnum() else "_" for c in doc_name[:50])
        return f"{safe_name}_{content_hash}"

    def get(self, doc_name: str, markdown_content: str) -> str | None:
        """
        Retrieve cached description for markdown content.

        Args:
            doc_name: Name of the document
            markdown_content: Markdown content

        Returns:
            Description string if cached, None otherwise
        """
        if not self.enabled:
            return None

        cache_key = self._get_content_cache_key(doc_name, markdown_content)

        try:
            with FileCache(self.cache_path) as cache:
                result = cache.get(cache_key)
                if result is not None:
                    logger.debug(f"Cache hit for description of {doc_name}")
                    return result
        except Exception as e:
            logger.warning(f"Failed to load cached description for {doc_name}: {e}")

        return None

    def put(
        self,
        doc_name: str,
        markdown_content: str,
        description: str,
    ) -> None:
        """
        Store description in cache.

        Args:
            doc_name: Name of the document
            markdown_content: Markdown content
            description: LLM-generated description
        """
        if not self.enabled:
            return

        cache_key = self._get_content_cache_key(doc_name, markdown_content)

        try:
            with FileCache(self.cache_path) as cache:
                cache.put(cache_key, description)
            logger.debug(f"Cached description for {doc_name}")
        except Exception as e:
            logger.warning(f"Failed to cache description for {doc_name}: {e}")

    def clear(self) -> None:
        """Clear all cached descriptions."""
        if not self.enabled:
            return

        try:
            with FileCache(self.cache_path) as cache:
                cache.clear()
            logger.info("Cleared file description cache")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")


class AnalyzerDescriptionCache:
    """
    Cache for analyzer agent description results.

    Caches the final description (logs output) produced by the AnalyzerGraph
    for each file. Cache key is based on file path and modification time so
    that changed files are re-analyzed automatically.

    Cache structure:
        <cache_base_dir>/analyzer_descriptions_<llm_hash>/
    """

    def __init__(
        self,
        cache_base_dir: Path | str,
        llm_model: str = "",
        code_timeout: int = 30,
        max_debug_tries: int = 3,
        enabled: bool = True,
    ):
        self.cache_base_dir = Path(cache_base_dir)
        self.enabled = enabled
        self.llm_model = llm_model
        self.code_timeout = code_timeout
        self.max_debug_tries = max_debug_tries

        if self.enabled:
            self.cache_base_dir.mkdir(parents=True, exist_ok=True)
            config_hash = self._generate_config_hash()
            cache_dir_name = f"analyzer_descriptions_{config_hash}"
            self.cache_path = self.cache_base_dir / cache_dir_name
            self.cache_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Analyzer description cache initialized at: %s", self.cache_path
            )

    def _generate_config_hash(self) -> str:
        """
        Generate a hash from all parameters that affect analysis results:
        LLM model, code_timeout, and max_debug_tries.
        """
        config_str = (
            f"model={self.llm_model}_"
            f"timeout={self.code_timeout}_"
            f"debug={self.max_debug_tries}"
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_file_cache_key(self, file_path: Path) -> str:
        """
        Generate cache key for a specific file based on path and mtime.

        Args:
            file_path: Path to the file

        Returns:
            Cache key string
        """
        abs_path = str(file_path.resolve())
        try:
            mtime = file_path.stat().st_mtime
        except Exception:
            mtime = 0

        key_str = f"{abs_path}_{mtime}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, file_path: Path) -> Dict[str, Any] | None:
        """
        Retrieve cached analysis result for a file.

        Args:
            file_path: Path to the file

        Returns:
            The cached result dict if present, None otherwise.
        """
        if not self.enabled:
            return None

        cache_key = self._get_file_cache_key(file_path)
        try:
            with FileCache(self.cache_path) as cache:
                result = cache.get(cache_key)
                if result is not None:
                    logger.debug("Analyzer cache hit for %s", file_path.name)
                    return result
        except Exception as e:
            logger.warning(
                "Failed to load analyzer cache for %s: %s", file_path.name, e
            )

        return None

    def put(self, file_path: Path, result: Dict[str, Any]) -> None:
        """
        Store an analysis result in the cache.

        Args:
            file_path: Path to the analyzed file
            result: The result dict produced by the analyzer graph
        """
        if not self.enabled:
            return

        cache_key = self._get_file_cache_key(file_path)
        try:
            with FileCache(self.cache_path) as cache:
                cache.put(cache_key, result)
            logger.debug("Cached analyzer result for %s", file_path.name)
        except Exception as e:
            logger.warning(
                "Failed to cache analyzer result for %s: %s", file_path.name, e
            )

    def clear(self) -> None:
        """Clear all cached analyzer results."""
        if not self.enabled:
            return

        try:
            with FileCache(self.cache_path) as cache:
                cache.clear()
            logger.info("Cleared analyzer description cache")
        except Exception as e:
            logger.warning("Failed to clear analyzer cache: %s", e)
