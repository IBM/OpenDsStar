import hashlib
import logging
import re
from dataclasses import dataclass
from logging import Logger
from typing import Any

from langchain_core.language_models import BaseChatModel

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


@dataclass(frozen=True)
class PromptMiss:
    doc_id: str
    display_name: str
    md_clean: str
    prompt: str
    file_path: str


class FileDescriptionGenerator:
    """
    Generates file descriptions from markdown using an LLM, with caching.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        description_cache: FileDescriptionCache,
        batch_size: int = 8,
        progress_every: int = 10,
        use_batch: bool = True,
    ) -> None:
        self.llm = llm
        self.description_cache = description_cache
        self.batch_size = int(batch_size)
        self.llm_batch_size = self.batch_size
        self.progress_every = max(1, int(progress_every))
        self.use_batch = use_batch

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        if not text:
            return text

        stripped = text.strip()
        stripped = re.sub(r"^```[a-zA-Z]*\s*\r?\n", "", stripped)
        stripped = re.sub(r"\r?\n```[\s\r\n]*$", "", stripped)
        return stripped.strip()

    def _normalize_llm_output(
        self,
        out: Any,
        doc_id: str = "",
        display_name: str = "",
        file_path: str = "",
    ) -> str:
        if isinstance(out, str):
            text = out
        elif hasattr(out, "content"):
            text = str(out.content)
        else:
            text = str(out)

        text = self._strip_markdown_fences(text)

        if doc_id and display_name:
            path_to_show = file_path or doc_id
            prefix = (
                f"## File Name\n{display_name}\n\n" f"## File Path\n{path_to_show}\n\n"
            )
            text = prefix + text

        return text

    @staticmethod
    def is_error_description(text: str) -> bool:
        stripped = (text or "").strip()
        return not stripped or len(stripped) < 80 or stripped.startswith("Error")

    @staticmethod
    def md_fingerprint(md: str) -> str:
        return hashlib.sha256((md or "").encode("utf-8")).hexdigest()[:16]

    def _collect_prompts(
        self,
        items: list[DescriptionInput],
    ) -> tuple[dict[str, str], list[PromptMiss]]:
        """
        Returns:
            cached_descriptions, misses
        """
        cached: dict[str, str] = {}
        misses: list[PromptMiss] = []

        for item in items:
            cached_desc = self.description_cache.get(item.doc_id, item.md_clean)

            if cached_desc is not None and not self.is_error_description(cached_desc):
                cached[item.doc_id] = cached_desc
                continue

            prompt = build_file_description_prompt(
                item.display_name,
                item.doc_id,
                item.md_for_prompt,
            )
            misses.append(
                PromptMiss(
                    doc_id=item.doc_id,
                    display_name=item.display_name,
                    md_clean=item.md_clean,
                    prompt=prompt,
                    file_path=item.file_path,
                )
            )

        return cached, misses

    def _generate_descs_for_misses(
        self,
        *,
        progress_label: str,
        misses: list[PromptMiss],
    ) -> tuple[dict[str, str], int, int]:
        """
        Returns:
            (generated_descriptions, generated_count, failed_count)
        """
        generated_descriptions: dict[str, str] = {}
        generated = 0
        failed = 0

        for batch_index, batch in enumerate(
            iter_batches(misses, self.llm_batch_size),
            start=1,
        ):
            if batch_index == 1 or batch_index % max(1, self.progress_every // 2) == 0:
                logger.info(
                    "%s: LLM progress batches=%d generated=%d/%d failed=%d use_batch=%s",
                    progress_label,
                    batch_index,
                    generated,
                    len(misses),
                    failed,
                    self.use_batch,
                )

            if self.use_batch:
                try:
                    raw_outputs = self.llm.batch(
                        [[("user", miss.prompt)] for miss in batch]
                    )

                    for miss, raw_output in zip(batch, raw_outputs):
                        desc = self._normalize_llm_output(
                            raw_output,
                            miss.doc_id,
                            miss.display_name,
                            miss.file_path,
                        )
                        generated_descriptions[miss.doc_id] = desc

                        if not self.is_error_description(desc):
                            self.description_cache.put(
                                miss.doc_id,
                                miss.md_clean,
                                desc,
                            )

                    generated += len(batch)
                    continue

                except Exception:
                    logger.exception(
                        "%s: LLM batch %d failed; falling back to per-item generation",
                        progress_label,
                        batch_index,
                    )

            for miss in batch:
                try:
                    desc = self._normalize_llm_output(
                        self.llm.invoke(miss.prompt),
                        miss.doc_id,
                        miss.display_name,
                        miss.file_path,
                    )
                    generated_descriptions[miss.doc_id] = desc

                    if not self.is_error_description(desc):
                        self.description_cache.put(
                            miss.doc_id,
                            miss.md_clean,
                            desc,
                        )

                    generated += 1

                except Exception as exc:
                    failed += 1
                    generated_descriptions[miss.doc_id] = (
                        f"Error generating description: {exc}"
                    )
                    logger.warning(
                        "%s: LLM invoke failed | doc_id=%s (%s)",
                        progress_label,
                        miss.doc_id,
                        exc,
                    )

        return generated_descriptions, generated, failed

    def generate(
        self,
        *,
        progress_label: str,
        items: list[DescriptionInput],
    ) -> tuple[dict[str, str], dict[str, int]]:
        """
        Returns:
            descriptions: dict[doc_id, description]
            stats: {"cached": int, "generated": int, "failed": int, "total": int}
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

        generated_descs, generated_count, failed_count = (
            self._generate_descs_for_misses(
                progress_label=progress_label,
                misses=misses,
            )
        )

        descriptions = {**cached, **generated_descs}
        stats = {
            "total": len(items),
            "cached": len(cached),
            "generated": generated_count,
            "failed": failed_count,
        }
        return descriptions, stats
