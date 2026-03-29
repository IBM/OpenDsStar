"""
markdown_shortener.py

A markdown shortener optimized for LLM context windows while preserving structure.

Features:
  - Truncates long Markdown tables to max_table_rows (keeps header + separator).
  - Truncates long contiguous lists to max_list_items (keeps first + last few).
  - Preserves fenced code blocks verbatim (no shortening inside).
  - Final guard: hard truncate to max_content_length chars:
      * prefers cutting at paragraph boundary
      * closes unclosed fences if needed
      * avoids mid-word cuts when possible
  - Returns stats about what was shortened.

Notes:
  - This is heuristic/deterministic and intentionally conservative to avoid false positives.
  - Table detection is stricter than “contains |” to reduce accidental matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ShortenStats:
    chars_before: int
    chars_after: int
    tables_truncated: List[Dict[str, Any]]
    lists_truncated: int


class MarkdownShortener:
    """
    Shortens markdown for LLM context while preserving important structure.

    Strategy (heuristic + deterministic):
      - Truncate long Markdown tables to max_table_rows (keeps header + separator).
      - Truncate long contiguous lists to max_list_items (keeps first + last few).
      - Preserve fenced code blocks verbatim (no shortening inside).
      - Final guard: hard truncate to max_content_length chars (tries not to break fences).
    """

    # Fence line like: ``` or ```python
    _FENCE_RE = re.compile(r"^\s*```")
    _LIST_ITEM_RE = re.compile(r"""^(\s*)([-*+]|(\d+)[.)])\s+""")

    # Table separator cell like: ---  :---  ---:  :---:
    _TABLE_SEP_CELL_RE = re.compile(r":?-{3,}:?$")

    def __init__(
        self,
        *,
        max_content_length: int = 50_000,
        max_table_rows: int = 50,
        max_list_items: int = 100,
        keep_last_list_items: int = 5,
        hard_truncate_ellipsis: str = "\n\n[...truncated...]\n",
        hard_truncate_boundary_window: int = 1200,
    ):
        self.max_content_length = int(max_content_length)
        self.max_table_rows = int(max_table_rows)
        self.max_list_items = int(max_list_items)
        self.keep_last_list_items = int(keep_last_list_items)
        self.hard_truncate_ellipsis = hard_truncate_ellipsis
        self.hard_truncate_boundary_window = int(hard_truncate_boundary_window)

    # ---------- Public API ----------

    def shorten(self, md: str) -> Tuple[str, Dict[str, Any]]:
        md = md or ""
        chars_before = len(md)

        lines = md.splitlines()
        out_lines: List[str] = []

        tables_truncated: List[Dict[str, Any]] = []
        lists_truncated = 0

        i = 0
        in_fence = False

        while i < len(lines):
            line = lines[i]

            # Fence toggling (``` or ```lang)
            if self._is_fence(line):
                in_fence = not in_fence
                out_lines.append(line)
                i += 1
                continue

            # Preserve fenced content verbatim
            if in_fence:
                out_lines.append(line)
                i += 1
                continue

            # Try table shortening first
            table_out, next_i, table_stats = self._maybe_shorten_table(lines, i)
            if next_i != i:
                out_lines.extend(table_out)
                if table_stats:
                    tables_truncated.append(table_stats)
                i = next_i
                continue

            # Try list shortening
            list_out, next_i, list_stats = self._maybe_shorten_list(lines, i)
            if next_i != i:
                out_lines.extend(list_out)
                if list_stats:
                    lists_truncated += 1
                i = next_i
                continue

            # Default: keep line
            out_lines.append(line)
            i += 1

        shortened = "\n".join(out_lines).rstrip() + "\n"
        shortened = self._hard_truncate(shortened)

        stats: Dict[str, Any] = ShortenStats(
            chars_before=chars_before,
            chars_after=len(shortened),
            tables_truncated=tables_truncated,
            lists_truncated=lists_truncated,
        ).__dict__

        return shortened, stats

    # ---------- Core helpers ----------

    def _is_fence(self, line: str) -> bool:
        return bool(self._FENCE_RE.match(line or ""))

    def _hard_truncate(self, md: str) -> str:
        """
        Ensures final output <= max_content_length characters.
        Tries to cut at a paragraph boundary, then closes unclosed fences if needed.
        """
        if self.max_content_length <= 0:
            return ""

        if len(md) <= self.max_content_length:
            return md

        cut = md[: self.max_content_length]

        # Prefer cutting at a paragraph boundary near the end of the cut.
        # This preserves sections better than rsplit on space alone.
        window = max(0, len(cut) - max(200, self.hard_truncate_boundary_window))
        boundary = cut.rfind("\n\n", window)
        if boundary != -1 and boundary > 0:
            cut = cut[:boundary].rstrip()

        # If we still end with a partial word, try to back up to last space/newline.
        if cut and not cut.endswith((" ", "\n")) and " " in cut:
            # Only back up a bit; don't destroy too much content
            head, _, _ = cut.rpartition(" ")
            if len(head) >= int(0.9 * len(cut)):
                cut = head

        # Close unclosed fences if needed
        if cut.count("```") % 2 == 1:
            cut = cut.rstrip() + "\n```"

        return (cut.rstrip() + self.hard_truncate_ellipsis).rstrip() + "\n"

    # ---------- Table shortening ----------

    @staticmethod
    def _looks_like_table_row(line: str) -> bool:
        """
        Conservative: require a typical table row shape.
        Most markdown table rows start/end with | and contain at least 2 pipes.
        """
        s = (line or "").strip()
        return s.startswith("|") and s.endswith("|") and s.count("|") >= 2

    def _is_table_sep(self, line: str) -> bool:
        """
        Stricter separator detection to reduce false positives.

        Example:
          | --- | :---: | ---: |
        """
        s = (line or "").strip()
        if not self._looks_like_table_row(s):
            return False

        inner = s.strip("|").strip()
        if not inner:
            return False

        cells = [c.strip() for c in inner.split("|")]
        if not cells:
            return False

        return all(
            self._TABLE_SEP_CELL_RE.fullmatch(c or "") is not None for c in cells
        )

    def _maybe_shorten_table(
        self, lines: List[str], start: int
    ) -> Tuple[List[str], int, Optional[Dict[str, Any]]]:
        if start + 1 >= len(lines):
            return [], start, None

        header = lines[start]
        sep = lines[start + 1]

        if not (self._looks_like_table_row(header) and self._is_table_sep(sep)):
            return [], start, None

        i = start + 2
        rows: List[str] = []

        # Consume contiguous table body rows (stop on blank, non-row, or fence)
        while i < len(lines):
            ln = lines[i]
            if self._is_fence(ln):
                break
            if ln.strip() == "":
                break
            if not self._looks_like_table_row(ln):
                break
            rows.append(ln)
            i += 1

        total_rows = len(rows)
        if total_rows <= self.max_table_rows:
            return [header, sep] + rows, i, None

        kept = rows[: self.max_table_rows]
        omitted = total_rows - len(kept)

        out = (
            [header, sep] + kept + ["", f"[Table truncated: omitted {omitted} row(s)]"]
        )
        stats = {
            "start_line": start,
            "rows_before": total_rows,
            "rows_after": len(kept),
            "omitted": omitted,
        }
        return out, i, stats

    # ---------- List shortening ----------

    def _maybe_shorten_list(
        self, lines: List[str], start: int
    ) -> Tuple[List[str], int, Optional[Dict[str, Any]]]:
        if start >= len(lines):
            return [], start, None

        m0 = self._LIST_ITEM_RE.match(lines[start] or "")
        if not m0:
            return [], start, None

        i = start
        items: List[List[str]] = []
        cur: List[str] = []

        def flush() -> None:
            nonlocal cur
            if cur:
                items.append(cur)
                cur = []

        base_indent = m0.group(1) or ""

        while i < len(lines):
            line = lines[i]

            if self._is_fence(line):
                break

            m = self._LIST_ITEM_RE.match(line or "")
            if m:
                flush()
                cur.append(line)
            else:
                # Continuation lines: blank OR indented beyond the base item indent.
                # The "+ 2 spaces" is a common markdown continuation convention.
                if cur and (line.strip() == "" or line.startswith(base_indent + "  ")):
                    cur.append(line)
                else:
                    break
            i += 1

        flush()

        total_items = len(items)
        if total_items <= self.max_list_items:
            return [ln for it in items for ln in it], i, None

        keep_tail = max(0, min(self.keep_last_list_items, total_items))
        keep_head = max(0, self.max_list_items - keep_tail)

        # Ensure we don't exceed total_items and avoid negative omitted
        keep_head = min(keep_head, total_items)
        keep_tail = min(keep_tail, max(0, total_items - keep_head))

        head = items[:keep_head]
        tail = items[-keep_tail:] if keep_tail else []
        omitted = max(0, total_items - (len(head) + len(tail)))

        marker = f"{base_indent}[List truncated: omitted {omitted} item(s)]"

        out: List[str] = []
        for it in head:
            out.extend(it)
        out.append(marker)
        out.append("")
        for it in tail:
            out.extend(it)

        stats = {
            "start_line": start,
            "items_before": total_items,
            "items_after": len(head) + len(tail),
            "omitted": omitted,
            "kept_head": len(head),
            "kept_tail": len(tail),
        }
        return out, i, stats
