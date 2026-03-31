import hashlib
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
from .sources import SourceFile, TempMaterializer

logger: Logger = logging.getLogger(__name__)


@dataclass
class AnalyzedItem:
    doc_id: str
    display_name: str
    file_path: str
    md_clean: str
    md_for_prompt: str
    truncation_stats: Dict[str, Any]


@dataclass(frozen=True)
class AnalysisMiss:
    src: SourceFile
    file_path_on_disk: str
    cache_key_path: str
    raw_bytes: Optional[bytes]
    doc_id: str


class DoclingDescriptionBuilder(DocumentDescriptionBuilder):
    def __init__(
        self,
        cache_dir: Union[str, Path],
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
    ):
        """
        Initialize DoclingDescriptionBuilder.

        Args:
            cache_dir: Cache directory (Milvus DB will be stored here)
            model: Model for file description generation (default: WX_MISTRAL_MEDIUM)
            temperature: Temperature for generation (default: 0.0)
            embedding_model: Embedding model (default: GRANITE_EMBEDDING)
            batch_size: Batch size for processing (default: 8)
            collection_name: Milvus collection name
            max_content_length: Max content length for markdown
            max_table_rows: Max table rows
            max_list_items: Max list items
            max_fallback_bytes: Max fallback bytes
            enable_caching: Enable caching
            progress_every: Progress logging frequency
            llm: Optional pre-built BaseChatModel instance. If provided,
                skips ModelBuilder.build() and uses this LLM directly.
        """
        _ = kwargs

        # Set parameters with defaults
        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.batch_size = batch_size

        # Use ModelBuilder to create the model instance
        # Set cache_dir first so we can pass it to ModelBuilder
        self.cache_dir = Path(cache_dir)

        if llm is not None:
            self.llm = llm
            llm_model_name = getattr(llm, "model_name", None) or type(llm).__name__
        else:
            self.llm, llm_model_name = ModelBuilder.build(
                self.model, temperature=self.temperature, cache_dir=self.cache_dir
            )

        # Create a hash of model + prompt to ensure different models/prompts use different Milvus DBs
        # Get prompt template for hashing
        from .prompts import build_file_description_prompt

        prompt_template = build_file_description_prompt("", "", "")

        import hashlib

        config_str = f"{llm_model_name}_{prompt_template}"
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        # Create subdirectory for vector store cache (matching VectorStoreTool pattern)
        vector_store_cache_dir = (
            self.cache_dir / f"vector_store_tool_cache_{config_hash}"
        )

        # Place Milvus DB inside the vector store cache directory
        db_uri = str(vector_store_cache_dir / f"milvus_analyzer_{config_hash}.db")

        # Detect if using custom API model (some don't support batching)
        # Check if model uses custom prefix from environment
        custom_prefix = os.getenv("CUSTOM_API_PREFIX", "")
        use_batch = not (custom_prefix and self.model.startswith(f"{custom_prefix}/"))
        self.db_uri = db_uri
        self.collection_name = collection_name
        self.progress_every = max(1, int(progress_every))

        # Ensure cache directories exist
        if enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            vector_store_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Analysis and Description caches with proper cache_dir
        self.analysis_cache = DoclingAnalysisCache(
            cache_base_dir=self.cache_dir,
            max_content_length=max_content_length,
            max_table_rows=max_table_rows,
            max_list_items=max_list_items,
            max_fallback_bytes=max_fallback_bytes,
            enabled=enable_caching,
        )

        # Get prompt template for cache key (extract the template without variables)
        from .prompts import build_file_description_prompt

        # Extract just the template structure by calling with placeholder values
        prompt_template = build_file_description_prompt("", "", "")

        self.description_cache = FileDescriptionCache(
            cache_base_dir=self.cache_dir,
            llm_model=llm_model_name,
            prompt_template=prompt_template,
            enabled=enable_caching,
        )

        # Initialize tools
        self.converter = DoclingConverter(max_fallback_bytes=max_fallback_bytes)
        self.md_shortener = MarkdownShortener(
            max_content_length=max_content_length,
            max_table_rows=max_table_rows,
            max_list_items=max_list_items,
        )

        # --- Integration: MilvusManager created lazily to avoid loading
        #     embedding models when only describe_files() is needed ---
        self._milvus_config = MilvusConfig(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        self._milvus_manager: Optional[MilvusManager] = None

        self.desc_generator = FileDescriptionGenerator(
            llm=self.llm,
            description_cache=self.description_cache,
            model=self.model,
            temperature=self.temperature,
            batch_size=self.batch_size,
            progress_every=self.progress_every,
            use_batch=use_batch,
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
    # Stream normalization FIX
    # ----------------------------

    @staticmethod
    def _stream_factory_to_bytes(sf) -> bytes:
        s = sf()
        if isinstance(s, (bytes, bytearray)):
            return bytes(s)
        data = s.read()
        try:
            s.seek(0)
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
    def _add_truncation_banner(md: str, stats: Dict[str, Any]) -> str:
        if stats.get("chars_after", 0) < stats.get("chars_before", 0):
            return (
                f"[Note: content truncated from {stats['chars_before']} to {stats['chars_after']} characters]\n\n"
                + md
            )
        return md

    # ----------------------------
    # Source handling
    # ----------------------------

    def _doc_id_for_source(
        self, src: SourceFile, file_path_on_disk: str, raw_bytes: Optional[bytes]
    ) -> str:
        base_name = src.display_name
        if raw_bytes is not None:
            digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
            return f"{base_name}::{digest}"

        p = Path(file_path_on_disk or src.path_hint)
        try:
            st = p.stat()
            key = f"{str(p.resolve())}::{st.st_size}::{int(st.st_mtime_ns)}"
        except Exception:
            key = str(p)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return f"{base_name}::{digest}"

    def _stream_cache_key_path(self, *, src: SourceFile, raw_bytes: bytes) -> str:
        """Generate a cache key path for stream-based sources.

        Note: This returns a virtual path used only as a cache key.
        No actual files are created at this location.
        """
        digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
        suffix = Path(src.path_hint).suffix
        # Use the cache_dir from the instance
        key_dir = self.cache_dir / "stream_cache_keys"
        return str(key_dir / f"{src.display_name}__{digest}{suffix}")

    def _resolve_inputs(
        self, src: SourceFile, materializer: TempMaterializer
    ) -> Tuple[str, str, str, Optional[bytes]]:
        """Resolve inputs for a source file.

        Returns:
            Tuple of (file_path_on_disk, cache_key_path, relative_path, raw_bytes)
            - file_path_on_disk: absolute path for reading the file
            - cache_key_path: path used for caching
            - relative_path: relative path from Document.path (for metadata)
            - raw_bytes: optional raw bytes for stream sources
        """
        if src.temp_path:
            return src.temp_path, src.temp_path, src.path_hint, None

        if src.stream_factory is not None:
            raw_bytes = self._stream_factory_to_bytes(src.stream_factory)
            cache_key_path = self._stream_cache_key_path(src=src, raw_bytes=raw_bytes)
            # For streams, use path_hint as the relative path
            return cache_key_path, cache_key_path, src.path_hint, raw_bytes

        return src.path_hint, src.path_hint, src.path_hint, None

    # ----------------------------
    # Analyze (cache pre-scan -> convert missing -> shorten)
    # ----------------------------

    def _load_cached_analysis(
        self, cache_key_path: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        return self.analysis_cache.get(Path(cache_key_path))

    def _store_cached_analysis(
        self, cache_key_path: str, md: str, stats: Dict[str, Any]
    ) -> None:
        self.analysis_cache.put(Path(cache_key_path), md, stats)

    def _collect_analysis_hits_and_misses(
        self,
        *,
        sources: Sequence[SourceFile],
        materializer: TempMaterializer,
    ) -> Tuple[List[AnalyzedItem], List[AnalysisMiss], Dict[str, Dict[str, Any]]]:
        """
        Pre-scan sources: resolve inputs, compute doc_id, and split into:
          - cached analyzed items (ready)
          - misses (need convert_one/export/shorten)
          - failures (e.g., cached empty)
        """
        cached_items: List[AnalyzedItem] = []
        misses: List[AnalysisMiss] = []
        failures: Dict[str, Dict[str, Any]] = {}

        for src in sources:
            file_path_on_disk, cache_key_path, relative_path, raw_bytes = (
                self._resolve_inputs(src, materializer)
            )
            doc_id = self._doc_id_for_source(src, file_path_on_disk, raw_bytes)

            cached = self._load_cached_analysis(cache_key_path)
            if cached is not None:
                md_clean, stats = cached
                if not (md_clean or "").strip():
                    failures[doc_id] = {
                        "success": False,
                        "fatal_error": "Empty/unsupported document",
                        "answer": "",
                        "logs": "",
                        "outputs": {"truncation": {}},
                        "file_path": relative_path,  # Use relative path
                        "doc_id": doc_id,
                        "filename": src.display_name,
                    }
                    continue

                md_for_prompt = self._add_truncation_banner(md_clean, stats)
                cached_items.append(
                    AnalyzedItem(
                        doc_id=doc_id,
                        display_name=src.display_name,
                        file_path=relative_path,  # Use relative path
                        md_clean=md_clean,
                        md_for_prompt=md_for_prompt,
                        truncation_stats=stats,
                    )
                )
            else:
                misses.append(
                    AnalysisMiss(
                        src=src,
                        file_path_on_disk=file_path_on_disk,
                        cache_key_path=cache_key_path,
                        raw_bytes=raw_bytes,
                        doc_id=doc_id,
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
        misses: List[AnalysisMiss],
        progress_label: str,
    ) -> Tuple[List[AnalyzedItem], Dict[str, Dict[str, Any]]]:
        """
        Convert/export/shorten ONLY for cache misses.
        Returns analyzed_items and failures.
        """
        analyzed: List[AnalyzedItem] = []
        failures: Dict[str, Dict[str, Any]] = {}

        # Get relative path for each miss (from src.path_hint which is Document.path)
        miss_relative_paths = {miss.doc_id: miss.src.path_hint for miss in misses}

        total = len(misses)
        for i, m in enumerate(misses, start=1):
            if i == 1 or i == total or (i % self.progress_every == 0):
                logger.info(
                    "%s: analyze missing progress %d/%d", progress_label, i, total
                )

            doc = self.converter.convert_one(
                display_name=m.src.display_name,
                path=Path(m.file_path_on_disk),
                raw_bytes=m.raw_bytes,
                suffix=Path(m.src.path_hint).suffix,
            )
            if not doc:
                failures[m.doc_id] = {
                    "success": False,
                    "fatal_error": "Empty/unsupported document",
                    "answer": "",
                    "logs": "",
                    "outputs": {"truncation": {}},
                    "file_path": miss_relative_paths[m.doc_id],  # Use relative path
                    "doc_id": m.doc_id,
                    "filename": m.src.display_name,
                }
                continue

            try:
                md_raw = doc.export_to_markdown()
            except Exception as e:
                logger.exception(
                    "export_to_markdown failed: %s (%s)", m.src.display_name, e
                )
                failures[m.doc_id] = {
                    "success": False,
                    "fatal_error": f"Error exporting markdown: {e}",
                    "answer": "",
                    "logs": "",
                    "outputs": {"truncation": {}},
                    "file_path": miss_relative_paths[m.doc_id],  # Use relative path
                    "doc_id": m.doc_id,
                    "filename": m.src.display_name,
                }
                continue

            md_clean, stats = self.md_shortener.shorten(md_raw)
            self._store_cached_analysis(m.cache_key_path, md_clean, stats)

            if not (md_clean or "").strip():
                failures[m.doc_id] = {
                    "success": False,
                    "fatal_error": "Empty/unsupported document",
                    "answer": "",
                    "logs": "",
                    "outputs": {"truncation": {}},
                    "file_path": miss_relative_paths[m.doc_id],  # Use relative path
                    "doc_id": m.doc_id,
                    "filename": m.src.display_name,
                }
                continue

            md_for_prompt = self._add_truncation_banner(md_clean, stats)
            analyzed.append(
                AnalyzedItem(
                    doc_id=m.doc_id,
                    display_name=m.src.display_name,
                    file_path=miss_relative_paths[m.doc_id],  # Use relative path
                    md_clean=md_clean,
                    md_for_prompt=md_for_prompt,
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
        items: List[AnalyzedItem],
        descriptions: Dict[str, str],
        failures: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[LangchainDocument], int]:
        results: Dict[str, Dict[str, Any]] = dict(failures)
        docs_to_add: List[LangchainDocument] = []
        success_cnt = 0

        for it in items:
            desc = descriptions.get(it.doc_id, "")
            success = not self.desc_generator.is_error_description(desc)
            if success:
                success_cnt += 1

            results[it.doc_id] = {
                "success": success,
                "fatal_error": (
                    ""
                    if success
                    else (desc[:500] if desc else "Failed to generate description")
                ),
                "answer": desc if success else "",
                "logs": desc if success else "",
                "outputs": {"truncation": it.truncation_stats},
                "file_path": it.file_path,
                "doc_id": it.doc_id,
                "filename": it.display_name,
                "md_fingerprint": self.desc_generator.md_fingerprint(it.md_clean),
            }

            if success:
                docs_to_add.append(
                    LangchainDocument(
                        page_content=desc,
                        metadata={
                            "doc_id": it.doc_id,
                            "filename": it.display_name,
                            "file_path": it.file_path,
                            "kind": "description",
                        },
                    )
                )

        return results, docs_to_add, success_cnt

    @property
    def milvus_manager(self) -> MilvusManager:
        """Lazily create MilvusManager on first access."""
        if self._milvus_manager is None:
            self._milvus_manager = MilvusManager(self._milvus_config)
        return self._milvus_manager

    def _open_vector_db(self) -> Milvus:
        """
        Delegates to MilvusManager to open the database.
        """
        return self.milvus_manager.open()

    def _add_to_vector_db(
        self,
        vector_db: Milvus,
        docs_to_add: List[LangchainDocument],
        progress_label: str,
    ) -> None:
        """
        Delegates to MilvusManager to add documents, filtering out existing ones
        based on the 'doc_id' found in the metadata.
        """
        if not docs_to_add:
            logger.warning(
                "%s: vector_db | no successful descriptions to add", progress_label
            )
            return

        t0 = time.perf_counter()

        # Use the manager's smart insertion logic
        # We use "doc_id" as the unique key to check against existence in Milvus.
        self.milvus_manager.add_documents_if_missing(
            vector_db,
            docs_to_add,
            progress_label=progress_label,
            key_field="doc_id",
            scalar_field_preference=["doc_id", "source_key"],
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "%s: vector_db | operation complete (%.1fms)",
            progress_label,
            dt_ms,
        )

    # ----------------------------
    # Main pipeline
    # ----------------------------

    def _process_sources(
        self,
        sources: Sequence[SourceFile],
        *,
        progress_label: str = "Process",
        skip_vectordb: bool = False,
    ) -> Tuple[Optional[Milvus], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        started = time.perf_counter()
        total_sources = len(sources)

        materializer = TempMaterializer()
        path_to_bytes_factory: Dict[str, Any] = {}
        analyzed: List[AnalyzedItem] = []
        failures: Dict[str, Dict[str, Any]] = {}

        try:
            logger.info("%s: start | files=%d", progress_label, total_sources)

            # Always keep a way to reload original bytes later
            # Use path_hint (relative path from Document.path) as key for uniqueness
            for src in sources:
                file_path_on_disk, _, relative_path, _ = self._resolve_inputs(
                    src, materializer
                )
                # Use relative_path (which is Document.path) as the key
                path_to_bytes_factory[relative_path] = (
                    src.stream_factory
                    if src.stream_factory is not None
                    else partial(self._read_file_bytes, file_path_on_disk)
                )

            # Stage 1: analyze (pre-scan cache, print stats, then process misses)
            cached_items, misses, pre_failures = self._collect_analysis_hits_and_misses(
                sources=sources,
                materializer=materializer,
            )
            self._log_analyze_plan(
                progress_label=progress_label,
                total_sources=total_sources,
                cached_cnt=len(cached_items),
                missing_cnt=len(misses),
                failures_cnt=len(pre_failures),
            )

            new_items, miss_failures = self._analyze_missing(
                misses=misses,
                progress_label=progress_label,
            )

            analyzed = [*cached_items, *new_items]
            failures = {**failures, **pre_failures, **miss_failures}

            # Stage 2: describe (delegated)
            desc_inputs = [
                DescriptionInput(
                    doc_id=it.doc_id,
                    display_name=it.display_name,
                    md_clean=it.md_clean,
                    md_for_prompt=it.md_for_prompt,
                    file_path=it.file_path,
                )
                for it in analyzed
            ]
            descriptions, desc_stats = self.desc_generator.generate(
                progress_label=progress_label,
                items=desc_inputs,
            )

            # Stage 3: results (+ docs to add)
            analysis_results, docs_to_add, ok_desc_cnt = self._build_analysis_results(
                items=analyzed,
                descriptions=descriptions,
                failures=failures,
            )

            logger.info(
                "%s: results | ok_descriptions=%d failed_descriptions=%d docs_to_add=%d",
                progress_label,
                ok_desc_cnt,
                len(analyzed) - ok_desc_cnt,
                len(docs_to_add),
            )

            # Stage 4: vector db (optional)
            vector_db: Optional[Milvus] = None
            if not skip_vectordb:
                vector_db = self._open_vector_db()
                self._add_to_vector_db(vector_db, docs_to_add, progress_label)

            total_dt_s = time.perf_counter() - started
            logger.info(
                "%s: done | sources=%d analyzed=%d cached_desc=%d generated=%d failed=%d total=%.2fs",
                progress_label,
                total_sources,
                len(analyzed),
                desc_stats["cached"],
                desc_stats["generated"],
                desc_stats["failed"],
                total_dt_s,
            )

            return vector_db, analysis_results, path_to_bytes_factory

        finally:
            materializer.cleanup()

    # ----------------------------
    # Public entrypoints
    # ----------------------------

    def describe_files(
        self,
        file_paths: List[Path],
        *,
        progress_label: str = "DescribeFiles",
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Run ingestion stages 1-3 (analyze + describe) without vector DB insertion.

        This is useful when the caller manages its own vector store (e.g., Langflow).

        Args:
            file_paths: List of file paths to process.
            progress_label: Label for progress logging.

        Returns:
            Tuple of (analysis_results, path_to_bytes_factory).
            analysis_results maps doc_id to metadata dict with keys:
                success, answer (description), file_path, filename, etc.
            path_to_bytes_factory maps file paths to callables returning bytes.
        """
        sources: List[SourceFile] = []
        for p in file_paths:
            p = Path(p)
            sources.append(
                SourceFile(
                    display_name=p.name,
                    path_hint=str(p),
                    temp_path=str(p.resolve()),
                    stream_factory=None,
                )
            )

        _, analysis_results, path_to_bytes_factory = self._process_sources(
            sources, progress_label=progress_label, skip_vectordb=True
        )
        return analysis_results, path_to_bytes_factory

    def process_directory(
        self,
        dir_path: Path,
        *,
        limit: Optional[int] = None,
    ) -> Tuple[Milvus, Dict[str, Dict[str, Any]], Dict[str, Any]]:
        files = self._iter_files(dir_path)
        if limit is not None:
            files = files[:limit]

        # Use absolute paths for file reading, but store relative paths in metadata
        dir_path_resolved = Path(dir_path).resolve()
        sources: List[SourceFile] = []
        for p in files:
            # Create a temp materialized file with absolute path for reading
            # but use relative path as path_hint for metadata
            relative_path = str(p.relative_to(dir_path_resolved))
            sources.append(
                SourceFile(
                    display_name=p.name,
                    path_hint=relative_path,  # Relative path for metadata
                    temp_path=str(p),  # Absolute path for file reading
                    stream_factory=None,
                )
            )
        return self._process_sources(sources, progress_label="Process")

    def _iter_files(self, dir_path: Path) -> List[Path]:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning("Directory does not exist: %s", dir_path)
            return []
        files = [
            p for p in dir_path.rglob("*") if p.is_file() and p.name != ".DS_Store"
        ]
        files.sort(key=lambda x: str(x).lower())
        logger.info("Discovered %d files under %s", len(files), dir_path)
        return files

    def process_corpus(
        self, corpus: Sequence[Document]
    ) -> Tuple[Milvus, Dict[str, Dict[str, Any]], Dict[str, Any]]:
        sources: List[SourceFile] = []
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
