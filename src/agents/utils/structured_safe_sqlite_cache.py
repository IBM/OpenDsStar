"""
StructuredSafeSQLiteCache - JSON-based SQLite cache for LangChain.

This module provides a SQLite cache that stores generations as JSON (not pickle),
while preserving chat structure needed for structured output parsing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
from typing import Any, Optional, Sequence

from langchain_core.caches import BaseCache
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

logger = logging.getLogger(__name__)


class StructuredSafeSQLiteCache(BaseCache):
    """
    SQLite cache that stores generations as JSON (not pickle), while preserving
    chat structure needed for structured output parsing.

    Why this cache?
    - LangChain's default SQLiteCache can rely on pickle for generation objects.
    - Pickled payloads are brittle across versions and can break structured flows.
    - This cache stores a JSON-safe representation and reconstructs ChatGeneration.

    Storage strategy:
    - For ChatGeneration: store AIMessage fields + generation_info
    - For plain Generation: store text + generation_info
    """

    def __init__(self, database_path: str, sqlite_timeout_sec: float = 30.0):
        self.database_path = database_path
        self.sqlite_timeout_sec = sqlite_timeout_sec
        self._lock = threading.Lock()
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path, timeout=self.sqlite_timeout_sec)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _create_table(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lc_cache (
                        prompt_hash TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _hash_prompt(prompt: str, llm_string: str) -> str:
        combined = f"{prompt}:{llm_string}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def _json_dumps_safe(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _serialize_generation(gen: Generation) -> dict[str, Any]:
        """
        Serialize Generation or ChatGeneration to JSON-safe dict.
        """
        if isinstance(gen, ChatGeneration):
            msg = gen.message
            # AIMessage.content can be str or list[dict|str]
            payload = {
                "type": "chat_generation",
                "message": {
                    "content": msg.content,
                    "additional_kwargs": getattr(msg, "additional_kwargs", {}) or {},
                    "response_metadata": getattr(msg, "response_metadata", {}) or {},
                    "usage_metadata": getattr(msg, "usage_metadata", None),
                    "id": getattr(msg, "id", None),
                    "name": getattr(msg, "name", None),
                },
                "generation_info": gen.generation_info or {},
            }
            return payload

        # Plain text generation fallback
        return {
            "type": "generation",
            "text": getattr(gen, "text", str(gen)),
            "generation_info": getattr(gen, "generation_info", {}) or {},
        }

    @staticmethod
    def _deserialize_generation(payload: dict[str, Any]) -> Generation:
        ptype = payload.get("type")

        if ptype == "chat_generation":
            msg_data = payload.get("message", {})
            message = AIMessage(
                content=msg_data.get("content", ""),
                additional_kwargs=msg_data.get("additional_kwargs", {}) or {},
                response_metadata=msg_data.get("response_metadata", {}) or {},
                usage_metadata=msg_data.get("usage_metadata", None),
                id=msg_data.get("id", None),
                name=msg_data.get("name", None),
            )
            return ChatGeneration(
                message=message,
                generation_info=payload.get("generation_info", {}) or {},
            )

        # Default fallback
        return Generation(
            text=payload.get("text", ""),
            generation_info=payload.get("generation_info", {}) or {},
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Generation]]:
        prompt_hash = self._hash_prompt(prompt, llm_string)

        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT payload_json FROM lc_cache WHERE prompt_hash = ?",
                    (prompt_hash,),
                )
                row = cursor.fetchone()
            finally:
                conn.close()

        if not row:
            return None

        try:
            container = json.loads(row[0])
            generations_payload = container.get("generations", [])
            generations = [self._deserialize_generation(g) for g in generations_payload]
            if not generations:
                return None
            logger.info("LiteLLM cache hit for prompt_hash: %s", prompt_hash[:16])
            return generations
        except Exception as exc:
            logger.warning("Failed to deserialize cached generation: %s", exc)
            return None

    def update(
        self, prompt: str, llm_string: str, return_val: Sequence[Generation]
    ) -> None:
        if not return_val:
            return

        prompt_hash = self._hash_prompt(prompt, llm_string)

        generations_payload = [self._serialize_generation(g) for g in return_val]
        container = {"generations": generations_payload}
        payload_json = self._json_dumps_safe(container)

        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO lc_cache (prompt_hash, payload_json)
                    VALUES (?, ?)
                    ON CONFLICT(prompt_hash) DO UPDATE SET payload_json = excluded.payload_json
                    """,
                    (prompt_hash, payload_json),
                )
                conn.commit()
            finally:
                conn.close()

    def clear(self, **kwargs: Any) -> None:
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM lc_cache")
                conn.commit()
            finally:
                conn.close()

    # -------- Async wrappers --------
    async def alookup(
        self, prompt: str, llm_string: str
    ) -> Optional[Sequence[Generation]]:
        return await asyncio.to_thread(self.lookup, prompt, llm_string)

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: Sequence[Generation]
    ) -> None:
        await asyncio.to_thread(self.update, prompt, llm_string, return_val)

    async def aclear(self, **kwargs: Any) -> None:
        await asyncio.to_thread(self.clear, **kwargs)
