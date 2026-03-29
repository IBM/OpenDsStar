import gc
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

import torch
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores.milvus import Milvus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MilvusConfig:
    db_uri: str
    collection_name: str
    embedding_model: str


class MilvusManager:
    """
    Owns:
      - Opening Milvus (LangChain wrapper)
      - Best-effort access to underlying pymilvus Collection
      - "Insert only if missing" via key lookup (scalar field preferred, JSON metadata fallback)

    Assumptions:
      - Each document has metadata['source_key'] (canonical dedupe key)
      - Collection may or may not have scalar fields (e.g., source_key) or JSON field (metadata/meta)
    """

    def __init__(self, config: MilvusConfig):
        self.config = config
        self._embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)

    # ----------------------------
    # Open / schema helpers
    # ----------------------------

    def open(self) -> Milvus:
        return Milvus(
            embedding_function=self._embeddings,
            connection_args={"uri": self.config.db_uri},
            collection_name=self.config.collection_name,
            auto_id=True,
        )

    @staticmethod
    def _escape_expr_string(s: str) -> str:
        # json.dumps returns a quoted string with proper escaping
        return json.dumps(s)

    @staticmethod
    def _get_underlying_collection(vector_db: Milvus) -> Any:
        for attr in ("col", "_collection", "collection"):
            col = getattr(vector_db, attr, None)
            if col is not None:
                return col
        return None

    @staticmethod
    def _collection_field_names(col: Any) -> List[str]:
        try:
            schema = getattr(col, "schema", None)
            if schema is None:
                return []
            fields = getattr(schema, "fields", None)
            if not fields:
                return []
            return [f.name for f in fields if getattr(f, "name", None)]
        except Exception:
            return []

    # ----------------------------
    # Existence checks
    # ----------------------------

    def _query_existing_keys(
        self,
        col: Any,
        *,
        keys: List[str],
        scalar_field_preference: List[str],
    ) -> Tuple[Set[str], Optional[str]]:
        """
        Returns: (existing_keys, mode_used)
          mode_used:
            - "scalar:<field>"
            - "json:<metadata_field>.<json_key>"
            - None  (unsupported/failed)
        """
        if not keys:
            return set(), None

        field_names = set(self._collection_field_names(col))

        # 1) scalar field: source_key (or other)
        for field in scalar_field_preference:
            if field in field_names:
                existing: Set[str] = set()
                batch_size = 200
                for i in range(0, len(keys), batch_size):
                    batch = keys[i : i + batch_size]
                    quoted = ",".join(self._escape_expr_string(k) for k in batch)
                    expr = f"{field} in [{quoted}]"
                    rows = col.query(expr=expr, output_fields=[field])
                    for r in rows or []:
                        v = r.get(field)
                        if isinstance(v, str):
                            existing.add(v)
                return existing, f"scalar:{field}"

        # 2) JSON field (metadata/meta), with sub-key (source_key or file_path/filename)
        json_fields = [f for f in ("metadata", "meta") if f in field_names]
        if json_fields:
            meta_field = json_fields[0]
            json_key_preference = ["source_key", "file_path", "filename"]

            # Collect across all batches; don't early-return on partial results
            existing: Set[str] = set()
            batch_size = 50
            for json_key in json_key_preference:
                for i in range(0, len(keys), batch_size):
                    batch = keys[i : i + batch_size]
                    parts = [
                        f'{meta_field}["{json_key}"] == {self._escape_expr_string(k)}'
                        for k in batch
                    ]
                    expr = " or ".join(parts)
                    try:
                        rows = col.query(expr=expr, output_fields=[meta_field])
                    except Exception:
                        continue

                    for r in rows or []:
                        meta = r.get(meta_field)
                        if isinstance(meta, dict):
                            v = meta.get(json_key)
                            if isinstance(v, str):
                                existing.add(v)

                if existing:
                    return existing, f"json:{meta_field}.{json_key}"

        return set(), None

    # ----------------------------
    # Public operations
    # ----------------------------

    def filter_missing(
        self,
        vector_db: Milvus,
        docs: List[LangchainDocument],
        *,
        key_field: str = "source_key",
        scalar_field_preference: Optional[List[str]] = None,
    ) -> Tuple[List[LangchainDocument], int, str]:
        """
        Keeps only docs whose key is missing in Milvus.
        Returns: (filtered_docs, skipped_count, mode_used)
        """
        if not docs:
            return [], 0, "none"

        scalar_field_preference = scalar_field_preference or [key_field]

        def doc_key(d: LangchainDocument) -> str:
            meta = d.metadata or {}
            k = meta.get(key_field) or ""
            return k if isinstance(k, str) else ""

        keys = [doc_key(d) for d in docs]
        non_empty_keys = sorted({k for k in keys if k})

        col = self._get_underlying_collection(vector_db)
        if col is None:
            logger.warning(
                "Milvus underlying collection not accessible; cannot dedupe. Inserting all."
            )
            return docs, 0, "fallback:no_collection"

        existing, mode = self._query_existing_keys(
            col,
            keys=non_empty_keys,
            scalar_field_preference=scalar_field_preference,
        )
        if mode is None:
            logger.warning("Milvus key existence query unsupported; inserting all.")
            return docs, 0, "fallback:no_filter_support"

        filtered: List[LangchainDocument] = []
        skipped = 0
        for d in docs:
            k = doc_key(d)
            if k and k in existing:
                skipped += 1
            else:
                filtered.append(d)

        return filtered, skipped, mode

    def add_documents_if_missing(
        self,
        vector_db: Milvus,
        docs: List[LangchainDocument],
        *,
        progress_label: str = "Process",
        key_field: str = "source_key",
        scalar_field_preference: Optional[List[str]] = None,
        batch_size: int = 2,
    ) -> None:
        if not docs:
            logger.info("%s: milvus | no docs to add", progress_label)
            return

        filtered, skipped, mode = self.filter_missing(
            vector_db,
            docs,
            key_field=key_field,
            scalar_field_preference=scalar_field_preference,
        )

        logger.info(
            "%s: milvus prefilter | candidates=%d will_add=%d skipped_existing=%d mode=%s",
            progress_label,
            len(docs),
            len(filtered),
            skipped,
            mode,
        )

        if not filtered:
            logger.info("%s: milvus | nothing new to add", progress_label)
            return

        # Add documents in batches to avoid memory issues with large embeddings
        total_docs = len(filtered)
        total_batches = (total_docs + batch_size - 1) // batch_size

        for i in range(0, total_docs, batch_size):
            batch = filtered[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(
                "%s: milvus | adding batch %d/%d (size=%d)",
                progress_label,
                batch_num,
                total_batches,
                len(batch),
            )
            vector_db.add_documents(batch)

            # Clear GPU memory after each batch to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

            # Small delay to allow memory to be fully released
            time.sleep(0.5)

        logger.info(
            "%s: milvus | add_documents ok | n=%d (in %d batches)",
            progress_label,
            total_docs,
            total_batches,
        )
