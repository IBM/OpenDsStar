"""Resolve the Milvus connection URI, with platform-aware fallback."""

import os
import sys


def resolve_milvus_uri(local_db_path: str) -> str:
    """Return the Milvus URI to use for connections.

    Priority:
      1. ``OPENDSSTAR_MILVUS_URI`` env var (remote / Docker server)
      2. *local_db_path* — only when ``milvus-lite`` is importable
      3. Raise with a helpful message (Windows or missing package)
    """
    env_uri = os.environ.get("OPENDSSTAR_MILVUS_URI", "").strip()
    if env_uri:
        return env_uri

    try:
        import milvus_lite  # noqa: F401

        return local_db_path
    except ImportError:
        platform = "Windows" if sys.platform == "win32" else "this platform"
        raise RuntimeError(
            f"milvus-lite is not available on {platform}. "
            "Set the OPENDSSTAR_MILVUS_URI environment variable to a remote "
            "Milvus server (e.g. http://localhost:19530)."
        )
