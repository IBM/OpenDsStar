import logging
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream

from src.ingestion.utils import parquet_to_csv_bytes

logger = logging.getLogger(__name__)


@dataclass
class FallbackTextDoc:
    name: str
    _markdown: str

    def export_to_markdown(self) -> str:
        return self._markdown


class DoclingConverter:
    def __init__(
        self,
        *,
        max_fallback_bytes: int = 2_000_000,
        docling_suffixes: Optional[set[str]] = None,
        text_fallback_suffixes: Optional[set[str]] = None,
        progress_bytes_mb: float = 25.0,
    ):
        self.max_fallback_bytes = max_fallback_bytes
        self._converter = DocumentConverter()
        self.progress_bytes = int(progress_bytes_mb * 1024 * 1024)

        self.docling_suffixes = docling_suffixes or {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".xml",
            ".html",
            ".htm",
            ".xhtml",
            ".md",
            ".markdown",
            ".rst",
            ".json",
            ".jsonl",
            ".ndjson",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".csv",
            ".tsv",
            ".txt",
            ".text",
            ".log",
            ".vtt",
            ".srt",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".cpp",
            ".c",
            ".h",
            ".sh",
            ".bash",
            ".zsh",
            ".sql",
        }
        self.parquet_suffixes = {".parquet"}
        self.text_fallback_suffixes = text_fallback_suffixes or {
            ".txt",
            ".text",
            ".log",
            ".md",
            ".rst",
        }

        self._code_suffixes = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".cpp",
            ".c",
            ".h",
            ".sh",
            ".bash",
            ".zsh",
            ".sql",
        }
        self._json_suffixes = {".json", ".jsonl", ".ndjson"}
        self._yaml_suffixes = {".yaml", ".yml"}
        self._xmlish_suffixes = {".xml", ".html", ".htm", ".xhtml"}

    def _fmt_bytes(self, n: int) -> str:
        if n < 1024:
            return f"{n} B"
        if n < 1024 * 1024:
            return f"{n / 1024:.1f} KB"
        if n < 1024 * 1024 * 1024:
            return f"{n / (1024 * 1024):.2f} MB"
        return f"{n / (1024 * 1024 * 1024):.2f} GB"

    def classify_suffix(self, suffix: str) -> str:
        s = (suffix or "").lower()
        if s in self.text_fallback_suffixes:
            return "fallback_text"
        if s in self.parquet_suffixes:
            return "parquet_to_csv"
        if s in self.docling_suffixes:
            return "docling"
        return "skip"

    def _read_text_bytes(self, raw: bytes, suffix: str) -> str:
        truncated = len(raw) > self.max_fallback_bytes
        raw = raw[: self.max_fallback_bytes]

        text = ""
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                pass
        if not text:
            text = raw.decode("utf-8", errors="replace")

        note = (
            f"\n\n[Note: file truncated to first {self.max_fallback_bytes} bytes]"
            if truncated
            else ""
        )

        sfx = (suffix or "").lower()
        if sfx in self._code_suffixes:
            return f"```{sfx.lstrip('.')}\n{text}\n```{note}"
        if sfx in self._json_suffixes:
            return f"```json\n{text}\n```{note}"
        if sfx in self._yaml_suffixes:
            return f"```yaml\n{text}\n```{note}"
        if sfx in self._xmlish_suffixes:
            return f"```xml\n{text}\n```{note}"
        return text + note

    def _fallback_from_bytes(
        self, display_name: str, raw: bytes, suffix: str
    ) -> FallbackTextDoc:
        return FallbackTextDoc(display_name, self._read_text_bytes(raw, suffix))

    def _fallback_from_path(self, display_name: str, path: Path) -> FallbackTextDoc:
        with path.open("rb") as f:
            raw = f.read(self.max_fallback_bytes + 1)  # +1 to detect truncation
        return self._fallback_from_bytes(display_name, raw, path.suffix)

    def _parquet_to_csv_bytes(
        self,
        display_name: str,
        path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> bytes:
        """
        Convert parquet file to CSV bytes in-memory.

        This method now delegates to the standalone utility function
        in ingestion.utils for better code reusability.
        """
        return parquet_to_csv_bytes(display_name, path, raw_bytes)

    def _maybe_log_large(self, display_name: str, size: int) -> None:
        if self.progress_bytes > 0 and size >= self.progress_bytes:
            logger.info(
                "Docling convert: %s | large input (%s) — conversion may take a while",
                display_name,
                self._fmt_bytes(size),
            )

    def convert_one(
        self,
        *,
        display_name: str,
        path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
        suffix: str = "",
    ) -> Union[FallbackTextDoc, DoclingDocument, None]:
        if path is None and raw_bytes is None:
            raise RuntimeError("Either path or raw_bytes must be provided")

        suffix = (suffix or (path.suffix if path else "")).lower()
        mode = self.classify_suffix(suffix)

        if mode == "skip":
            logger.debug(
                "Docling convert: SKIP suffix=%s | %s", suffix or "<none>", display_name
            )
            return None

        if mode == "parquet_to_csv":
            try:
                logger.debug(
                    "Docling convert: PARQUET_TO_CSV suffix=%s | %s",
                    suffix or "<none>",
                    display_name,
                )
                # Convert parquet to CSV bytes
                csv_bytes = self._parquet_to_csv_bytes(display_name, path, raw_bytes)

                # Now process the CSV bytes through Docling
                csv_display_name = display_name.replace(".parquet", ".csv")
                return self.convert_one(
                    display_name=csv_display_name, raw_bytes=csv_bytes, suffix=".csv"
                )
            except Exception:
                logger.error(
                    "Docling convert: PARQUET_TO_CSV failed | %s", display_name
                )
                return None

        if mode == "fallback_text":
            try:
                if raw_bytes is not None:
                    logger.debug(
                        "Docling convert: FALLBACK_TEXT bytes=%s suffix=%s | %s",
                        self._fmt_bytes(len(raw_bytes)),
                        suffix or "<none>",
                        display_name,
                    )
                    return self._fallback_from_bytes(display_name, raw_bytes, suffix)
                assert path is not None
                size = path.stat().st_size if path.exists() else 0
                logger.debug(
                    "Docling convert: FALLBACK_TEXT size=%s suffix=%s | %s",
                    self._fmt_bytes(size),
                    suffix or "<none>",
                    display_name,
                )
                return self._fallback_from_path(display_name, path)
            except Exception:
                logger.error("Docling convert: FALLBACK_TEXT failed | %s", display_name)
                return None

        # mode == "docling"
        t0 = time.perf_counter()
        errors: list[BaseException] = []

        # Try Docling from bytes first, then from path.
        attempts: list[tuple[str, object]] = []
        if raw_bytes is not None:
            attempts.append(("bytes", raw_bytes))
        if path is not None:
            attempts.append(("path", path))

        for src, payload in attempts:
            try:
                if src == "bytes":
                    b = payload  # type: ignore[assignment]
                    self._maybe_log_large(display_name, len(b))  # type: ignore[arg-type]
                    ds_name = (
                        display_name
                        if not suffix or display_name.lower().endswith(suffix)
                        else f"{display_name}{suffix}"
                    )
                    doc = self._converter.convert(
                        DocumentStream(name=ds_name, stream=BytesIO(b))  # type: ignore[arg-type]
                    ).document
                else:
                    p = payload  # type: ignore[assignment]
                    size = p.stat().st_size if p.exists() else 0  # type: ignore[union-attr]
                    self._maybe_log_large(display_name, size)
                    doc = self._converter.convert(p).document  # type: ignore[arg-type]

                try:
                    doc.name = display_name  # type: ignore[attr-defined]
                except Exception:
                    pass

                dt_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    "Docling convert: OK %s (%.1fms)",
                    display_name,
                    dt_ms,
                )
                return doc

            except ConversionError as e:
                errors.append(e)
                logger.debug(
                    "Docling convert: FAIL source=%s suffix=%s | %s (%s)",
                    src,
                    suffix or "<none>",
                    display_name,
                    e,
                )
            except Exception as e:
                errors.append(e)
                logger.error(
                    "Docling convert: ERROR source=%s suffix=%s | %s",
                    src,
                    suffix or "<none>",
                    display_name,
                )

        # Docling failed everywhere -> fallback to text best-effort (bytes preferred).
        logger.warning(
            "Docling convert: falling back to text after docling failures suffix=%s | %s (errors=%d)",
            suffix or "<none>",
            display_name,
            len(errors),
        )
        try:
            if raw_bytes is not None:
                return self._fallback_from_bytes(display_name, raw_bytes, suffix)
            assert path is not None
            return self._fallback_from_path(display_name, path)
        except Exception:
            logger.exception("Docling convert: fallback failed | %s", display_name)
            return None
