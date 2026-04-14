import hashlib
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from logging import Logger
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from langchain_core.documents import Document as LangchainDocument
from langchain_milvus.vectorstores.milvus import Milvus

from OpenDsStar.agents.utils.model_builder import ModelBuilder
from OpenDsStar.core.model_registry import ModelRegistry
from OpenDsStar.experiments.core.types import Document

from ..docling_cache import DoclingAnalysisCache, FileDescriptionCache
from ..document_description_builder import DocumentDescriptionBuilder
from .docling_converter import DoclingConverter
from .file_description_generator import DescriptionInput, FileDescriptionGenerator
from .markdown_shortener import MarkdownShortener
from .milvus_manager import MilvusConfig, MilvusManager
from .prompts import build_file_description_prompt
from .sources import SourceFile, TempMaterializer

logger: Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyzedItem:
    doc_id: str
    display_name: str
    file_path: str
    md_clean: str
    md_for_prompt: str
    truncation_stats: dict[str, Any]


@dataclass(frozen=True)
class AnalysisMiss:
    src: SourceFile
    file_path_on_disk: str
    cache_key_path: str
    raw_bytes: Optional[bytes]
    doc_id: str


TABULAR_EXTENSIONS: set[str] = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}


class DoclingDescriptionBuilder(DocumentDescriptionBuilder):
    def __init__(
        self,
        cache_dir: str | Path,
        model: str = ModelRegistry.WX_MISTRAL_MEDIUM,
        temperature: float = 0.0,
        embedding_model: str = ModelRegistry.GRANITE_EMBEDDING,
        batch_size: int = 8,
        collection_name: str = "doc_descriptions",
        max_content_length: int = 20000,
        max_table_rows: int = 50,
        max_list_items: int = 100,
        max_fallback_bytes: int = 2_000_000,
        enable_caching: bool = True,
        progress_every: int = 5,
        llm: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        _ = kwargs

        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.batch_size = int(batch_size)
        self.collection_name = collection_name
        self.progress_every = max(1, int(progress_every))
        self.cache_dir = Path(cache_dir)

        self.llm, llm_model_name = self._build_llm(llm)
        prompt_template = build_file_description_prompt("", "", "")
        config_hash = self._config_hash(llm_model_name, prompt_template)

        vector_store_cache_dir = (
            self.cache_dir / f"vector_store_tool_cache_{config_hash}"
        )
        self.db_uri = str(vector_store_cache_dir / f"milvus_analyzer_{config_hash}.db")

        if enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            vector_store_cache_dir.mkdir(parents=True, exist_ok=True)

        self.analysis_cache = DoclingAnalysisCache(
            cache_base_dir=self.cache_dir,
            max_content_length=max_content_length,
            max_table_rows=max_table_rows,
            max_list_items=max_list_items,
            max_fallback_bytes=max_fallback_bytes,
            enabled=enable_caching,
        )

        self.description_cache = FileDescriptionCache(
            cache_base_dir=self.cache_dir,
            llm_model=llm_model_name,
            prompt_template=prompt_template,
            enabled=enable_caching,
        )

        self.converter = DoclingConverter(max_fallback_bytes=max_fallback_bytes)
        self.md_shortener = MarkdownShortener(
            max_content_length=max_content_length,
            max_table_rows=max_table_rows,
            max_list_items=max_list_items,
        )

        self._milvus_config = MilvusConfig(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        self._milvus_manager: Optional[MilvusManager] = None

        self.desc_generator = FileDescriptionGenerator(
            llm=self.llm,
            description_cache=self.description_cache,
            batch_size=self.batch_size,
            progress_every=self.progress_every,
            use_batch=self._should_use_batch(),
        )

        logger.info(
            "DoclingDescriptionBuilder init: llm=%s embedding=%s db_uri=%s collection=%s "
            "max_content_length=%d cache_dir=%s caching=%s llm_batch_size=%d",
            llm_model_name,
            self.embedding_model,
            self.db_uri,
            self.collection_name,
            max_content_length,
            self.cache_dir,
            enable_caching,
            self.batch_size,
        )

    # ----------------------------
    # Init helpers
    # ----------------------------

    def _build_llm(self, llm: Optional[Any]) -> tuple[Any, str]:
        if llm is not None:
            llm_model_name = getattr(llm, "model_name", None) or type(llm).__name__
            return llm, llm_model_name

        built_llm, llm_model_name = ModelBuilder.build(
            self.model,
            temperature=self.temperature,
            cache_dir=self.cache_dir,
        )
        return built_llm, llm_model_name

    @staticmethod
    def _config_hash(llm_model_name: str, prompt_template: str) -> str:
        config_str = f"{llm_model_name}_{prompt_template}"
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:12]

    def _should_use_batch(self) -> bool:
        custom_prefix = os.getenv("CUSTOM_API_PREFIX", "")
        return not (custom_prefix and self.model.startswith(f"{custom_prefix}/"))

    # ----------------------------
    # Stream normalization
    # ----------------------------

    @staticmethod
    def _stream_factory_to_bytes(stream_factory: Any) -> bytes:
        stream = stream_factory()
        if isinstance(stream, (bytes, bytearray)):
            return bytes(stream)

        data = stream.read()
        try:
            stream.seek(0)
        except Exception:
            pass
        return data

    # ----------------------------
    # Small utilities
    # ----------------------------

    @staticmethod
    def _read_file_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def _add_truncation_banner(md: str, stats: dict[str, Any]) -> str:
        if stats.get("chars_after", 0) < stats.get("chars_before", 0):
            return (
                f"[Note: content truncated from {stats['chars_before']} "
                f"to {stats['chars_after']} characters]\n\n{md}"
            )
        return md

    @staticmethod
    def _failure_result(
        *,
        fatal_error: str,
        file_path: str,
        doc_id: str,
        filename: str,
    ) -> dict[str, Any]:
        return {
            "success": False,
            "fatal_error": fatal_error,
            "answer": "",
            "logs": "",
            "outputs": {"truncation": {}},
            "file_path": file_path,
            "doc_id": doc_id,
            "filename": filename,
        }

    def _success_result(
        self,
        *,
        answer: str,
        file_path: str,
        doc_id: str,
        filename: str,
        truncation_stats: Optional[dict[str, Any]] = None,
        md_for_fingerprint: Optional[str] = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "success": True,
            "fatal_error": "",
            "answer": answer,
            "logs": answer,
            "outputs": {"truncation": truncation_stats or {}},
            "file_path": file_path,
            "doc_id": doc_id,
            "filename": filename,
        }
        if md_for_fingerprint is not None:
            result["md_fingerprint"] = self.desc_generator.md_fingerprint(
                md_for_fingerprint
            )
        return result

    @staticmethod
    def _description_to_document(
        *,
        doc_id: str,
        filename: str,
        file_path: str,
        description: str,
    ) -> LangchainDocument:
        return LangchainDocument(
            page_content=description,
            metadata={
                "doc_id": doc_id,
                "filename": filename,
                "file_path": file_path,
                "kind": "description",
            },
        )

    # ----------------------------
    # Tabular fast path
    # ----------------------------

    @staticmethod
    def _read_tabular_metadata(
        file_path: str,
        *,
        sample_rows: int = 5,
    ) -> tuple[int, int, list[tuple[str, str]], pd.DataFrame]:
        ext = Path(file_path).suffix.lower()

        if ext == ".parquet":
            df_full = pd.read_parquet(file_path)
            row_count = len(df_full)
            sample_df = df_full.head(sample_rows)
        elif ext in {".xlsx", ".xls"}:
            df_full = pd.read_excel(file_path)
            row_count = len(df_full)
            sample_df = df_full.head(sample_rows)
        else:
            sep = "\t" if ext == ".tsv" else ","
            df_count = pd.read_csv(file_path, sep=sep, usecols=[0])
            row_count = len(df_count)
            sample_df = pd.read_csv(file_path, sep=sep, nrows=sample_rows)

        dtype_display = {
            "object": "text",
            "bool": "boolean",
        }
        col_dtype_pairs = [
            (
                column,
                dtype_display.get(
                    str(sample_df[column].dtype), str(sample_df[column].dtype)
                ),
            )
            for column in sample_df.columns
        ]
        return row_count, len(sample_df.columns), col_dtype_pairs, sample_df

    @classmethod
    def _build_tabular_summary(cls, file_path: str, *, sample_rows: int = 5) -> str:
        row_count, col_count, col_dtypes, sample_df = cls._read_tabular_metadata(
            file_path,
            sample_rows=sample_rows,
        )

        file_name = Path(file_path).name
        columns_lines = "\n".join(f"- '{name}' ({dtype})" for name, dtype in col_dtypes)
        sample_md = sample_df.to_markdown(index=False)

        return (
            f"## File Name\n{file_name}\n\n"
            f"## File Path\n{file_path}\n\n"
            f"## Overview\nTabular data file with {row_count} rows and {col_count} columns.\n\n"
            f"## Columns\n{columns_lines}\n\n"
            f"## Sample Data (first {sample_rows} rows)\n{sample_md}"
        )

    @staticmethod
    def _extract_section(summary: str, marker: str) -> str:
        """Extract a section from a tabular summary, up to the next ## heading."""
        idx = summary.find(marker)
        if idx == -1:
            return ""
        rest = summary[idx + len(marker) :]
        # Find the next heading
        next_heading = rest.find("\n## ")
        if next_heading != -1:
            return rest[:next_heading].strip()
        return rest.strip()

    @staticmethod
    def _replace_or_append_section(desc: str, heading: str, replacement: str) -> str:
        """Replace an existing ## section in desc, or append if not present."""
        idx = desc.find(heading)
        if idx == -1:
            return desc.rstrip() + "\n\n" + replacement

        # Find the end of the section (next ## heading or end of string)
        after = desc[idx + len(heading) :]
        next_heading = after.find("\n## ")
        if next_heading != -1:
            return desc[:idx] + replacement + "\n\n" + after[next_heading + 1 :]
        return desc[:idx] + replacement

    @classmethod
    def _extract_sample_section(cls, summary: str) -> str:
        """Extract the '## Sample Data ...' section from a tabular summary."""
        marker = "## Sample Data"
        idx = summary.find(marker)
        if idx == -1:
            return ""
        section = summary[idx:].strip()
        return section.replace("## Sample Data", "## Sampled rows/data", 1)

    @classmethod
    def _extract_columns_section(cls, summary: str) -> str:
        """Build a '## Structured Data - Exact Column Names' section from the summary."""
        columns_text = cls._extract_section(summary, "## Columns\n")
        if not columns_text:
            return ""
        # columns_text lines are: "- 'name' (dtype)"
        lines = []
        for i, line in enumerate(columns_text.splitlines(), 1):
            line = line.strip().lstrip("- ")
            if line:
                lines.append(f"{i}. {line}")
        return "## Structured Data - Exact Column Names\n" + "\n".join(lines)

    def _analyze_tabular_files(
        self,
        sources: Sequence[SourceFile],
        *,
        progress_label: str,
    ) -> tuple[list[AnalyzedItem], dict[str, dict[str, Any]]]:
        """Analyze tabular files without docling: read metadata and build summary markdown.

        Returns (analyzed_items, failures) in the same format as _analyze_missing,
        so the results feed directly into the standard describe + build_results pipeline.
        """
        logger.info("Fast path: analyzing %d tabular file(s)", len(sources))

        analyzed: list[AnalyzedItem] = []
        failures: dict[str, dict[str, Any]] = {}

        for src in sources:
            file_path_on_disk = src.temp_path or src.path_hint
            relative_path = src.path_hint
            doc_id = self._doc_id_for_source(src, file_path_on_disk, None)

            # Check analysis cache first (avoids re-reading file from disk)
            cached_analysis = self._load_cached_analysis(file_path_on_disk)
            if cached_analysis is not None:
                summary, _stats = cached_analysis
            else:
                try:
                    summary = self._build_tabular_summary(file_path_on_disk)
                except Exception as exc:
                    logger.warning(
                        "%s: fast path failed for %s: %s",
                        progress_label,
                        src.display_name,
                        exc,
                    )
                    failures[doc_id] = self._failure_result(
                        fatal_error=f"Tabular read error: {exc}",
                        file_path=relative_path,
                        doc_id=doc_id,
                        filename=src.display_name,
                    )
                    continue

                self._store_cached_analysis(file_path_on_disk, summary, {})

            analyzed.append(
                AnalyzedItem(
                    doc_id=doc_id,
                    display_name=src.display_name,
                    file_path=relative_path,
                    md_clean=summary,
                    md_for_prompt=summary,
                    truncation_stats={},
                )
            )

        return analyzed, failures

    # ----------------------------
    # Source handling
    # ----------------------------

    def _doc_id_for_source(
        self,
        src: SourceFile,
        file_path_on_disk: str,
        raw_bytes: Optional[bytes],
    ) -> str:
        base_name = src.display_name

        if raw_bytes is not None:
            digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
            return f"{base_name}::{digest}"

        path = Path(file_path_on_disk or src.path_hint)
        try:
            stat = path.stat()
            key = f"{path.resolve()}::{stat.st_size}::{int(stat.st_mtime_ns)}"
        except Exception:
            key = str(path)

        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return f"{base_name}::{digest}"

    def _stream_cache_key_path(self, *, src: SourceFile, raw_bytes: bytes) -> str:
        digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
        suffix = Path(src.path_hint).suffix
        key_dir = self.cache_dir / "stream_cache_keys"
        return str(key_dir / f"{src.display_name}__{digest}{suffix}")

    def _resolve_inputs(
        self,
        src: SourceFile,
        materializer: TempMaterializer,
    ) -> tuple[str, str, str, Optional[bytes]]:
        _ = materializer

        if src.temp_path:
            return src.temp_path, src.temp_path, src.path_hint, None

        if src.stream_factory is not None:
            raw_bytes = self._stream_factory_to_bytes(src.stream_factory)
            cache_key_path = self._stream_cache_key_path(src=src, raw_bytes=raw_bytes)
            return cache_key_path, cache_key_path, src.path_hint, raw_bytes

        return src.path_hint, src.path_hint, src.path_hint, None

    # ----------------------------
    # Analyze
    # ----------------------------

    def _load_cached_analysis(
        self,
        cache_key_path: str,
    ) -> Optional[tuple[str, dict[str, Any]]]:
        return self.analysis_cache.get(Path(cache_key_path))

    def _store_cached_analysis(
        self,
        cache_key_path: str,
        md: str,
        stats: dict[str, Any],
    ) -> None:
        self.analysis_cache.put(Path(cache_key_path), md, stats)

    def _collect_analysis_hits_and_misses(
        self,
        *,
        sources: Sequence[SourceFile],
        materializer: TempMaterializer,
    ) -> tuple[list[AnalyzedItem], list[AnalysisMiss], dict[str, dict[str, Any]]]:
        cached_items: list[AnalyzedItem] = []
        misses: list[AnalysisMiss] = []
        failures: dict[str, dict[str, Any]] = {}

        for src in sources:
            file_path_on_disk, cache_key_path, relative_path, raw_bytes = (
                self._resolve_inputs(src, materializer)
            )
            doc_id = self._doc_id_for_source(src, file_path_on_disk, raw_bytes)

            cached = self._load_cached_analysis(cache_key_path)
            if cached is None:
                misses.append(
                    AnalysisMiss(
                        src=src,
                        file_path_on_disk=file_path_on_disk,
                        cache_key_path=cache_key_path,
                        raw_bytes=raw_bytes,
                        doc_id=doc_id,
                    )
                )
                continue

            md_clean, stats = cached
            if not (md_clean or "").strip():
                failures[doc_id] = self._failure_result(
                    fatal_error="Empty/unsupported document",
                    file_path=relative_path,
                    doc_id=doc_id,
                    filename=src.display_name,
                )
                continue

            cached_items.append(
                AnalyzedItem(
                    doc_id=doc_id,
                    display_name=src.display_name,
                    file_path=relative_path,
                    md_clean=md_clean,
                    md_for_prompt=self._add_truncation_banner(md_clean, stats),
                    truncation_stats=stats,
                )
            )

        return cached_items, misses, failures

    def _log_analyze_plan(
        self,
        *,
        progress_label: str,
        total_sources: int,
        cached_cnt: int,
        missing_cnt: int,
        failures_cnt: int,
    ) -> None:
        logger.info(
            "%s: analyze plan | total=%d cached_md=%d missing_md=%d pre_failures=%d | will_convert=%d",
            progress_label,
            total_sources,
            cached_cnt,
            missing_cnt,
            failures_cnt,
            missing_cnt,
        )

    def _analyze_missing(
        self,
        *,
        misses: list[AnalysisMiss],
        progress_label: str,
    ) -> tuple[list[AnalyzedItem], dict[str, dict[str, Any]]]:
        analyzed: list[AnalyzedItem] = []
        failures: dict[str, dict[str, Any]] = {}
        miss_relative_paths = {miss.doc_id: miss.src.path_hint for miss in misses}

        total = len(misses)
        for index, miss in enumerate(misses, start=1):
            if index == 1 or index == total or index % self.progress_every == 0:
                logger.info(
                    "%s: analyze missing progress %d/%d",
                    progress_label,
                    index,
                    total,
                )

            doc = self.converter.convert_one(
                display_name=miss.src.display_name,
                path=Path(miss.file_path_on_disk),
                raw_bytes=miss.raw_bytes,
                suffix=Path(miss.src.path_hint).suffix,
            )
            if not doc:
                failures[miss.doc_id] = self._failure_result(
                    fatal_error="Empty/unsupported document",
                    file_path=miss_relative_paths[miss.doc_id],
                    doc_id=miss.doc_id,
                    filename=miss.src.display_name,
                )
                continue

            try:
                md_raw = doc.export_to_markdown()
            except Exception as exc:
                logger.exception(
                    "export_to_markdown failed: %s (%s)",
                    miss.src.display_name,
                    exc,
                )
                failures[miss.doc_id] = self._failure_result(
                    fatal_error=f"Error exporting markdown: {exc}",
                    file_path=miss_relative_paths[miss.doc_id],
                    doc_id=miss.doc_id,
                    filename=miss.src.display_name,
                )
                continue

            md_clean, stats = self.md_shortener.shorten(md_raw)
            self._store_cached_analysis(miss.cache_key_path, md_clean, stats)

            if not (md_clean or "").strip():
                failures[miss.doc_id] = self._failure_result(
                    fatal_error="Empty/unsupported document",
                    file_path=miss_relative_paths[miss.doc_id],
                    doc_id=miss.doc_id,
                    filename=miss.src.display_name,
                )
                continue

            analyzed.append(
                AnalyzedItem(
                    doc_id=miss.doc_id,
                    display_name=miss.src.display_name,
                    file_path=miss_relative_paths[miss.doc_id],
                    md_clean=md_clean,
                    md_for_prompt=self._add_truncation_banner(md_clean, stats),
                    truncation_stats=stats,
                )
            )

        return analyzed, failures

    # ----------------------------
    # Results + vector DB
    # ----------------------------

    def _build_analysis_results(
        self,
        *,
        items: list[AnalyzedItem],
        descriptions: dict[str, str],
        failures: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], list[LangchainDocument], int]:
        results: dict[str, dict[str, Any]] = dict(failures)
        docs_to_add: list[LangchainDocument] = []
        success_count = 0

        for item in items:
            desc = descriptions.get(item.doc_id, "")
            success = not self.desc_generator.is_error_description(desc)

            if success:
                success_count += 1
                results[item.doc_id] = self._success_result(
                    answer=desc,
                    file_path=item.file_path,
                    doc_id=item.doc_id,
                    filename=item.display_name,
                    truncation_stats=item.truncation_stats,
                    md_for_fingerprint=item.md_clean,
                )
                docs_to_add.append(
                    self._description_to_document(
                        doc_id=item.doc_id,
                        filename=item.display_name,
                        file_path=item.file_path,
                        description=desc,
                    )
                )
            else:
                results[item.doc_id] = self._failure_result(
                    fatal_error=(
                        desc[:500] if desc else "Failed to generate description"
                    ),
                    file_path=item.file_path,
                    doc_id=item.doc_id,
                    filename=item.display_name,
                )

        return results, docs_to_add, success_count

    @property
    def milvus_manager(self) -> MilvusManager:
        if self._milvus_manager is None:
            self._milvus_manager = MilvusManager(self._milvus_config)
        return self._milvus_manager

    def _open_vector_db(self) -> Milvus:
        return self.milvus_manager.open()

    def _add_to_vector_db(
        self,
        vector_db: Milvus,
        docs_to_add: list[LangchainDocument],
        progress_label: str,
    ) -> None:
        if not docs_to_add:
            logger.warning(
                "%s: vector_db | no successful descriptions to add",
                progress_label,
            )
            return

        started = time.perf_counter()
        self.milvus_manager.add_documents_if_missing(
            vector_db,
            docs_to_add,
            progress_label=progress_label,
            key_field="doc_id",
            scalar_field_preference=["doc_id", "source_key"],
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "%s: vector_db | operation complete (%.1fms)",
            progress_label,
            duration_ms,
        )

    # ----------------------------
    # Main pipeline
    # ----------------------------

    def _split_sources(
        self,
        sources: Sequence[SourceFile],
    ) -> tuple[list[SourceFile], list[SourceFile]]:
        tabular_sources: list[SourceFile] = []
        docling_sources: list[SourceFile] = []

        for src in sources:
            ext = Path(src.path_hint).suffix.lower()
            if ext in TABULAR_EXTENSIONS and src.stream_factory is None:
                tabular_sources.append(src)
            else:
                docling_sources.append(src)

        return tabular_sources, docling_sources

    def _build_path_to_bytes_factory(
        self,
        *,
        sources: Sequence[SourceFile],
        materializer: TempMaterializer,
    ) -> dict[str, Any]:
        path_to_bytes_factory: dict[str, Any] = {}

        for src in sources:
            file_path_on_disk, _, relative_path, _ = self._resolve_inputs(
                src,
                materializer,
            )
            path_to_bytes_factory[relative_path] = (
                src.stream_factory
                if src.stream_factory is not None
                else partial(self._read_file_bytes, file_path_on_disk)
            )

        return path_to_bytes_factory

    def _process_sources(
        self,
        sources: Sequence[SourceFile],
        *,
        progress_label: str = "Process",
        skip_vectordb: bool = False,
    ) -> tuple[Optional[Milvus], dict[str, dict[str, Any]], dict[str, Any]]:
        started = time.perf_counter()
        total_sources = len(sources)

        materializer = TempMaterializer()
        try:
            logger.info("%s: start | files=%d", progress_label, total_sources)

            tabular_sources, docling_sources = self._split_sources(sources)

            # Fast path: analyze tabular files (skip docling, read metadata only)
            tabular_items: list[AnalyzedItem] = []
            tabular_failures: dict[str, dict[str, Any]] = {}
            if tabular_sources:
                tabular_items, tabular_failures = self._analyze_tabular_files(
                    tabular_sources,
                    progress_label=progress_label,
                )

            path_to_bytes_factory = self._build_path_to_bytes_factory(
                sources=sources,
                materializer=materializer,
            )

            # Docling path: analyze non-tabular files
            cached_items, misses, pre_failures = self._collect_analysis_hits_and_misses(
                sources=docling_sources,
                materializer=materializer,
            )
            self._log_analyze_plan(
                progress_label=progress_label,
                total_sources=len(docling_sources),
                cached_cnt=len(cached_items),
                missing_cnt=len(misses),
                failures_cnt=len(pre_failures),
            )

            new_items, miss_failures = self._analyze_missing(
                misses=misses,
                progress_label=progress_label,
            )

            # Merge all analyzed items and failures into a single pipeline
            analyzed = [*tabular_items, *cached_items, *new_items]
            failures = {**tabular_failures, **pre_failures, **miss_failures}

            desc_inputs = [
                DescriptionInput(
                    doc_id=item.doc_id,
                    display_name=item.display_name,
                    md_clean=item.md_clean,
                    md_for_prompt=item.md_for_prompt,
                    file_path=item.file_path,
                )
                for item in analyzed
            ]
            descriptions, desc_stats = self.desc_generator.generate(
                progress_label=progress_label,
                items=desc_inputs,
            )

            # For tabular items, replace/append deterministic columns and sample rows
            tabular_doc_ids = {item.doc_id for item in tabular_items}
            for item in analyzed:
                if item.doc_id not in tabular_doc_ids:
                    continue
                desc = descriptions.get(item.doc_id, "")
                if not desc:
                    continue
                columns_section = self._extract_columns_section(item.md_clean)
                if columns_section:
                    desc = self._replace_or_append_section(
                        desc,
                        "## Structured Data - Exact Column Names",
                        columns_section,
                    )
                sample_section = self._extract_sample_section(item.md_clean)
                if sample_section:
                    desc = self._replace_or_append_section(
                        desc,
                        "## Sampled rows/data",
                        sample_section,
                    )
                descriptions[item.doc_id] = desc

            analysis_results, docs_to_add, ok_desc_count = self._build_analysis_results(
                items=analyzed,
                descriptions=descriptions,
                failures=failures,
            )

            logger.info(
                "%s: results | ok_descriptions=%d failed_descriptions=%d docs_to_add=%d",
                progress_label,
                ok_desc_count,
                len(analyzed) - ok_desc_count,
                len(docs_to_add),
            )

            vector_db: Optional[Milvus] = None
            if not skip_vectordb:
                vector_db = self._open_vector_db()
                self._add_to_vector_db(vector_db, docs_to_add, progress_label)

            total_duration_s = time.perf_counter() - started
            logger.info(
                "%s: done | sources=%d analyzed=%d cached_desc=%d generated=%d failed=%d total=%.2fs",
                progress_label,
                total_sources,
                len(analyzed),
                desc_stats["cached"],
                desc_stats["generated"],
                desc_stats["failed"],
                total_duration_s,
            )

            return vector_db, analysis_results, path_to_bytes_factory

        finally:
            materializer.cleanup()

    # ----------------------------
    # Public entrypoints
    # ----------------------------

    def describe_files(
        self,
        file_paths: list[Path],
        *,
        progress_label: str = "DescribeFiles",
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        sources: list[SourceFile] = []
        for path in file_paths:
            path = Path(path)
            sources.append(
                SourceFile(
                    display_name=path.name,
                    path_hint=str(path),
                    temp_path=str(path.resolve()),
                    stream_factory=None,
                )
            )

        _, analysis_results, path_to_bytes_factory = self._process_sources(
            sources,
            progress_label=progress_label,
            skip_vectordb=True,
        )
        return analysis_results, path_to_bytes_factory

    def process_directory(
        self,
        dir_path: Path,
        *,
        limit: Optional[int] = None,
    ) -> tuple[Milvus, dict[str, dict[str, Any]], dict[str, Any]]:
        files = self._iter_files(dir_path)
        if limit is not None:
            files = files[:limit]

        dir_path_resolved = Path(dir_path).resolve()
        sources: list[SourceFile] = []
        for path in files:
            relative_path = str(path.relative_to(dir_path_resolved))
            sources.append(
                SourceFile(
                    display_name=path.name,
                    path_hint=relative_path,
                    temp_path=str(path),
                    stream_factory=None,
                )
            )

        return self._process_sources(sources, progress_label="Process")

    def _iter_files(self, dir_path: Path) -> list[Path]:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning("Directory does not exist: %s", dir_path)
            return []

        files = [
            path
            for path in dir_path.rglob("*")
            if path.is_file() and path.name != ".DS_Store"
        ]
        files.sort(key=lambda path: str(path).lower())
        logger.info("Discovered %d files under %s", len(files), dir_path)
        return files

    def process_corpus(
        self,
        corpus: Sequence[Document],
    ) -> tuple[Milvus, dict[str, dict[str, Any]], dict[str, Any]]:
        sources: list[SourceFile] = []

        for doc in corpus:
            stream_factory = None
            if doc.stream_factory:

                def make_bytes_factory(sf):
                    def bytes_factory():
                        return sf().read()

                    return bytes_factory

                stream_factory = make_bytes_factory(doc.stream_factory)

            sources.append(
                SourceFile(
                    display_name=doc.document_id,
                    path_hint=doc.path,
                    stream_factory=stream_factory,
                )
            )

        return self._process_sources(sources, progress_label="Process")
