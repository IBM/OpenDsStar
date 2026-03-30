import hashlib
import logging
import re
from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict, List, Tuple

from langchain_core.language_models import BaseChatModel

from OpenDsStar.core.model_registry import ModelRegistry

from ..docling_cache import FileDescriptionCache
from .batching import iter_batches
from .prompts import build_file_description_prompt

logger: Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DescriptionInput:
    doc_id: str
    display_name: str
    md_clean: str
    md_for_prompt: str
    file_path: str = ""  # Relative path for file retrieval


class FileDescriptionGenerator:
    """
    Generates file descriptions from markdown using an LLM (batched), with caching.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        description_cache: FileDescriptionCache,
        model: str = ModelRegistry.WX_MISTRAL_MEDIUM,
        temperature: float = 0.0,
        batch_size: int = 8,
        progress_every: int = 10,
        use_batch: bool = True,
    ):
        self.llm = llm
        self.description_cache = description_cache
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self.llm_batch_size = int(self.batch_size)
        self.progress_every = max(1, int(progress_every))
        self.use_batch = use_batch

    # ----------------------------
    # Output normalization + heuristics
    # ----------------------------

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        if not text:
            return text
        t = text.strip()
        t = re.sub(r"^```[a-zA-Z]*\s*\r?\n", "", t)
        t = re.sub(r"\r?\n```[\s\r\n]*$", "", t)
        return t.strip()

    def _normalize_llm_output(
        self, out: Any, doc_id: str = "", display_name: str = "", file_path: str = ""
    ) -> str:
        if isinstance(out, str):
            text = out
        elif hasattr(out, "content"):
            text = str(out.content)
        else:
            text = str(out)
        text = self._strip_markdown_fences(text)

        # Add File Name and File Path prefix if provided
        if doc_id and display_name:
            # Use file_path (relative path) if available, otherwise fall back to doc_id
            path_to_show = file_path if file_path else doc_id
            prefix = f"## File Name\n{display_name}\n\n## File Path\n{path_to_show}\n\n"
            text = prefix + text

        return text

    @staticmethod
    def is_error_description(text: str) -> bool:
        t = (text or "").strip()
        return (not t) or len(t) < 80 or t.startswith("Error")

    @staticmethod
    def md_fingerprint(md: str) -> str:
        return hashlib.sha256((md or "").encode("utf-8")).hexdigest()[:16]

    # ----------------------------
    # Cache + prompt collection
    # ----------------------------

    def _collect_prompts(
        self, items: List[DescriptionInput]
    ) -> Tuple[Dict[str, str], List[Tuple[str, str, str, str, str, str]]]:
        """
        Returns:
          cached_descriptions, misses: (doc_id, display_name, md_clean, prompt, file_path, doc_id)
        """
        cached: Dict[str, str] = {}
        misses: List[Tuple[str, str, str, str, str, str]] = []

        for it in items:
            cached_desc = self.description_cache.get(it.doc_id, it.md_clean)
            # If cached description is an error, treat it as a miss and regenerate
            if cached_desc is not None and not self.is_error_description(cached_desc):
                cached[it.doc_id] = cached_desc
                continue

            prompt = build_file_description_prompt(
                it.display_name, it.doc_id, it.md_for_prompt
            )
            misses.append(
                (
                    it.doc_id,
                    it.display_name,
                    it.md_clean,
                    prompt,
                    it.file_path,
                    it.doc_id,
                )
            )

        return cached, misses

    # ----------------------------
    # Batched LLM generation
    # ----------------------------

    def _generate_descs_for_misses(
        self, *, progress_label: str, misses: List[Tuple[str, str, str, str, str, str]]
    ) -> Tuple[Dict[str, str], int, int]:
        """
        Returns: (generated_descriptions, generated_count, failed_count)
        misses format: (doc_id, display_name, md_clean, prompt, file_path, doc_id)
        """
        out: Dict[str, str] = {}
        generated = 0
        failed = 0

        for b_idx, batch in enumerate(
            iter_batches(misses, self.llm_batch_size), start=1
        ):
            doc_ids = [x[0] for x in batch]
            display_names = [x[1] for x in batch]
            md_cleans = [x[2] for x in batch]
            prompts = [x[3] for x in batch]
            file_paths = [x[4] for x in batch]

            if b_idx == 1 or (b_idx % max(1, self.progress_every // 2) == 0):
                logger.info(
                    "%s: LLM progress batches=%d generated=%d/%d failed=%d, using batch=%s",
                    progress_label,
                    b_idx,
                    generated,
                    len(misses),
                    failed,
                    str(self.use_batch),
                )

            # Use batch or invoke based on configuration
            if self.use_batch:
                try:
                    raw = self.llm.batch([[("user", p)] for p in prompts])
                    descs = [
                        self._normalize_llm_output(x, doc_id, display_name, file_path)
                        for x, doc_id, display_name, file_path in zip(
                            raw, doc_ids, display_names, file_paths
                        )
                    ]

                    for doc_id, md_clean, desc in zip(doc_ids, md_cleans, descs):
                        out[doc_id] = desc
                        # Only cache successful descriptions, not errors
                        if not self.is_error_description(desc):
                            self.description_cache.put(doc_id, md_clean, desc)

                    generated += len(prompts)

                except Exception:
                    logger.exception(
                        "%s: LLM batch %d failed; falling back per-item",
                        progress_label,
                        b_idx,
                    )
                    # Fall through to per-item processing
                    for doc_id, display_name, md_clean, prompt, file_path, _ in batch:
                        try:
                            desc = self._normalize_llm_output(
                                self.llm.invoke(prompt), doc_id, display_name, file_path
                            )
                            out[doc_id] = desc
                            # Only cache successful descriptions, not errors
                            if not self.is_error_description(desc):
                                self.description_cache.put(doc_id, md_clean, desc)
                            generated += 1
                        except Exception as e2:
                            failed += 1
                            err = f"Error generating description: {e2}"
                            out[doc_id] = err
                            # Don't cache error descriptions
                            logger.warning(
                                "%s: LLM invoke failed | doc_id=%s (%s)",
                                progress_label,
                                doc_id,
                                e2,
                            )
            else:
                # Use invoke directly (no batch) - for models that don't support batching
                for doc_id, display_name, md_clean, prompt, file_path, _ in batch:
                    try:
                        logger.info(
                            f"Generating file description, prompt length: {len(prompt)}"
                        )

                        # logger.info(f"Prompt:\n{prompt}")

                        desc = self._normalize_llm_output(
                            self.llm.invoke(prompt), doc_id, display_name, file_path
                        )

                        # logger.info(f"Generated file description:\n{desc}")
                        out[doc_id] = desc
                        # Only cache successful descriptions, not errors
                        if not self.is_error_description(desc):
                            self.description_cache.put(doc_id, md_clean, desc)
                        generated += 1
                    except Exception as e2:
                        failed += 1
                        err = f"Error generating description: {e2}"
                        out[doc_id] = err
                        # Don't cache error descriptions
                        logger.warning(
                            "%s: LLM invoke failed | doc_id=%s (%s)",
                            progress_label,
                            doc_id,
                            e2,
                        )

        return out, generated, failed

    # ----------------------------
    # Public API
    # ----------------------------

    def generate(
        self,
        *,
        progress_label: str,
        items: List[DescriptionInput],
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Returns:
          descriptions: Dict[doc_id -> description]
          stats: { "cached": int, "generated": int, "failed": int, "total": int }
        """
        cached, misses = self._collect_prompts(items)
        logger.info(
            "%s: descriptions | total=%d cached=%d to_generate=%d batch_size=%d",
            progress_label,
            len(items),
            len(cached),
            len(misses),
            self.llm_batch_size,
        )

        generated_descs, generated_cnt, failed_cnt = self._generate_descs_for_misses(
            progress_label=progress_label,
            misses=misses,
        )
        descriptions = {**cached, **generated_descs}

        stats = {
            "total": len(items),
            "cached": len(cached),
            "generated": generated_cnt,
            "failed": failed_cnt,
        }
        return descriptions, stats
