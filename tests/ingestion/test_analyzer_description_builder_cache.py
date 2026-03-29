"""
Test caching behavior for AnalyzerDescriptionBuilder.

Covers:
- Cache hit on second run (same file, same config)
- Cache miss when file content changes (mtime changes)
- Cache disabled mode (never caches)
- Only successful results are cached
- Different LLM model produces different cache directory
"""

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.analyzer import AnalyzerDescriptionBuilder
from ingestion.docling_cache import AnalyzerDescriptionCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """Create a small CSV file for testing."""
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    p = test_dir / "data.csv"
    p.write_text("id,name,value\n1,alpha,10\n2,beta,20\n", encoding="utf-8")
    return p


@pytest.fixture
def successful_analyzer_result() -> dict:
    """A result dict that looks like a successful analyzer run."""
    return {
        "answer": "File has 3 columns: id, name, value. 2 rows of data.",
        "logs": "Columns: id, name, value\nShape: (2, 3)",
        "outputs": {},
        "code": "df = pd.read_csv('data.csv')\nprint(df.columns.tolist())",
        "trajectory": [{"node": "entry"}, {"node": "n_code"}, {"node": "n_finalizer"}],
        "debug_tries": 0,
        "max_debug_tries": 3,
        "execution_error": None,
        "fatal_error": "",
        "success": True,
        "input_tokens": 100,
        "output_tokens": 50,
        "num_llm_calls": 1,
    }


@pytest.fixture
def failed_analyzer_result() -> dict:
    """A result dict that looks like a failed analyzer run."""
    return {
        "answer": "Analysis failed: FileNotFoundError",
        "logs": "",
        "outputs": {},
        "code": "df = pd.read_csv('missing.csv')",
        "trajectory": [{"node": "entry"}, {"node": "n_code"}],
        "debug_tries": 3,
        "max_debug_tries": 3,
        "execution_error": "FileNotFoundError: missing.csv",
        "fatal_error": "",
        "success": False,
        "input_tokens": 100,
        "output_tokens": 50,
        "num_llm_calls": 3,
    }


# ---------------------------------------------------------------------------
# AnalyzerDescriptionCache unit tests
# ---------------------------------------------------------------------------


class TestAnalyzerDescriptionCache:
    def test_get_returns_none_when_empty(self, tmp_path: Path, csv_file: Path):
        cache = AnalyzerDescriptionCache(
            cache_base_dir=tmp_path / "cache",
            llm_model="test-model",
            enabled=True,
        )
        assert cache.get(csv_file) is None

    def test_put_and_get_roundtrip(self, tmp_path: Path, csv_file: Path):
        cache = AnalyzerDescriptionCache(
            cache_base_dir=tmp_path / "cache",
            llm_model="test-model",
            enabled=True,
        )
        result = {"success": True, "logs": "hello"}
        cache.put(csv_file, result)

        retrieved = cache.get(csv_file)
        assert retrieved is not None
        assert retrieved["success"] is True
        assert retrieved["logs"] == "hello"

    def test_disabled_cache_never_stores(self, tmp_path: Path, csv_file: Path):
        cache = AnalyzerDescriptionCache(
            cache_base_dir=tmp_path / "cache",
            llm_model="test-model",
            enabled=False,
        )
        cache.put(csv_file, {"success": True, "logs": "data"})
        assert cache.get(csv_file) is None

    def test_cache_miss_when_file_mtime_changes(self, tmp_path: Path):
        cache = AnalyzerDescriptionCache(
            cache_base_dir=tmp_path / "cache",
            llm_model="test-model",
            enabled=True,
        )
        test_dir = tmp_path / "files"
        test_dir.mkdir()
        f = test_dir / "data.csv"
        f.write_text("a,b\n1,2\n")

        cache.put(f, {"success": True, "logs": "v1"})
        assert cache.get(f) is not None

        # Modify the file (change mtime)
        time.sleep(0.05)  # Ensure mtime changes
        f.write_text("a,b,c\n1,2,3\n")

        assert cache.get(f) is None, "Cache should miss after file modification"

    def test_different_model_uses_different_cache_dir(self, tmp_path: Path):
        cache_a = AnalyzerDescriptionCache(
            cache_base_dir=tmp_path / "cache",
            llm_model="model-a",
            enabled=True,
        )
        cache_b = AnalyzerDescriptionCache(
            cache_base_dir=tmp_path / "cache",
            llm_model="model-b",
            enabled=True,
        )
        assert cache_a.cache_path != cache_b.cache_path

    def test_different_timeout_uses_different_cache_dir(self, tmp_path: Path):
        cache_a = AnalyzerDescriptionCache(
            cache_base_dir=tmp_path / "cache",
            llm_model="model",
            code_timeout=30,
            enabled=True,
        )
        cache_b = AnalyzerDescriptionCache(
            cache_base_dir=tmp_path / "cache",
            llm_model="model",
            code_timeout=60,
            enabled=True,
        )
        assert cache_a.cache_path != cache_b.cache_path

    def test_clear(self, tmp_path: Path, csv_file: Path):
        cache = AnalyzerDescriptionCache(
            cache_base_dir=tmp_path / "cache",
            llm_model="test-model",
            enabled=True,
        )
        cache.put(csv_file, {"success": True, "logs": "data"})
        assert cache.get(csv_file) is not None

        cache.clear()
        assert cache.get(csv_file) is None


# ---------------------------------------------------------------------------
# AnalyzerDescriptionBuilder integration tests (with mocked analyzer graph)
# ---------------------------------------------------------------------------


class TestAnalyzerDescriptionBuilderCache:
    def _make_builder(
        self, cache_dir: Path, db_uri: str, enable_caching: bool = True
    ) -> AnalyzerDescriptionBuilder:
        """Create a builder with a mocked LLM."""
        mock_llm = MagicMock()
        mock_llm.model = "mock-analyzer-model"

        with patch("ingestion.analyzer.ChatLiteLLM", return_value=mock_llm):
            builder = AnalyzerDescriptionBuilder(
                llm=mock_llm,
                code_timeout=30,
                max_debug_tries=3,
                embedding_model="ibm-granite/granite-embedding-english-r2",
                db_uri=db_uri,
                cache_dir=cache_dir,
                enable_caching=enable_caching,
            )
        return builder

    def test_second_run_uses_cache(
        self,
        tmp_path: Path,
        csv_file: Path,
        successful_analyzer_result: dict,
        caplog,
    ):
        """Second run on the same file should hit cache and skip the analyzer graph."""
        cache_dir = tmp_path / "cache"
        db_uri = str(tmp_path / "milvus.db")

        builder = self._make_builder(cache_dir, db_uri, enable_caching=True)
        builder.description_cache.clear()

        successful_analyzer_result["file_path"] = str(csv_file.absolute())

        with (
            patch.object(builder.analyzer_graph, "invoke") as mock_invoke,
            patch(
                "ingestion.analyzer.prepare_result_from_graph_state_analyzer_agent",
                return_value=dict(successful_analyzer_result),
            ),
            patch("ingestion.analyzer.Milvus") as mock_milvus_cls,
            patch("ingestion.analyzer.HuggingFaceEmbeddings"),
        ):
            mock_milvus_cls.return_value = MagicMock()
            mock_invoke.return_value = MagicMock()

            # First run: should call analyzer graph
            caplog.set_level(logging.INFO)
            caplog.clear()
            builder._process_files([csv_file], {})
            assert mock_invoke.call_count == 1
            assert "cached=0" in caplog.text and "analyzed=1" in caplog.text

            # Second run: should use cache, not call analyzer graph
            mock_invoke.reset_mock()
            caplog.clear()
            builder._process_files([csv_file], {})
            assert mock_invoke.call_count == 0
            assert "cached=1" in caplog.text and "analyzed=0" in caplog.text

    def test_failed_results_not_cached(
        self,
        tmp_path: Path,
        csv_file: Path,
        failed_analyzer_result: dict,
        caplog,
    ):
        """Failed analysis results should not be cached."""
        cache_dir = tmp_path / "cache"
        db_uri = str(tmp_path / "milvus.db")

        builder = self._make_builder(cache_dir, db_uri, enable_caching=True)
        builder.description_cache.clear()

        failed_analyzer_result["file_path"] = str(csv_file.absolute())

        with (
            patch.object(builder.analyzer_graph, "invoke") as mock_invoke,
            patch(
                "ingestion.analyzer.prepare_result_from_graph_state_analyzer_agent",
                return_value=dict(failed_analyzer_result),
            ),
            patch("ingestion.analyzer.Milvus") as mock_milvus_cls,
            patch("ingestion.analyzer.HuggingFaceEmbeddings"),
        ):
            mock_milvus_cls.return_value = MagicMock()
            mock_invoke.return_value = MagicMock()

            caplog.set_level(logging.INFO)

            # First run
            builder._process_files([csv_file], {})
            assert mock_invoke.call_count == 1

            # Second run: should call analyzer again (not cached)
            mock_invoke.reset_mock()
            builder._process_files([csv_file], {})
            assert mock_invoke.call_count == 1

    def test_cache_disabled_always_runs_analyzer(
        self,
        tmp_path: Path,
        csv_file: Path,
        successful_analyzer_result: dict,
        caplog,
    ):
        """With caching disabled, every run should invoke the analyzer graph."""
        db_uri = str(tmp_path / "milvus.db")

        builder = self._make_builder(tmp_path / "cache", db_uri, enable_caching=False)

        successful_analyzer_result["file_path"] = str(csv_file.absolute())

        with (
            patch.object(builder.analyzer_graph, "invoke") as mock_invoke,
            patch(
                "ingestion.analyzer.prepare_result_from_graph_state_analyzer_agent",
                return_value=dict(successful_analyzer_result),
            ),
            patch("ingestion.analyzer.Milvus") as mock_milvus_cls,
            patch("ingestion.analyzer.HuggingFaceEmbeddings"),
        ):
            mock_milvus_cls.return_value = MagicMock()
            mock_invoke.return_value = MagicMock()

            caplog.set_level(logging.INFO)

            builder._process_files([csv_file], {})
            assert mock_invoke.call_count == 1

            builder._process_files([csv_file], {})
            assert mock_invoke.call_count == 2

    def test_cache_miss_when_file_changes(
        self,
        tmp_path: Path,
        successful_analyzer_result: dict,
        caplog,
    ):
        """Modifying a file should cause a cache miss."""
        cache_dir = tmp_path / "cache"
        db_uri = str(tmp_path / "milvus.db")

        test_dir = tmp_path / "files"
        test_dir.mkdir()
        csv_file = test_dir / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        builder = self._make_builder(cache_dir, db_uri, enable_caching=True)
        builder.description_cache.clear()

        successful_analyzer_result["file_path"] = str(csv_file.absolute())

        with (
            patch.object(builder.analyzer_graph, "invoke") as mock_invoke,
            patch(
                "ingestion.analyzer.prepare_result_from_graph_state_analyzer_agent",
                return_value=dict(successful_analyzer_result),
            ),
            patch("ingestion.analyzer.Milvus") as mock_milvus_cls,
            patch("ingestion.analyzer.HuggingFaceEmbeddings"),
        ):
            mock_milvus_cls.return_value = MagicMock()
            mock_invoke.return_value = MagicMock()

            caplog.set_level(logging.INFO)

            # First run
            builder._process_files([csv_file], {})
            assert mock_invoke.call_count == 1

            # Second run (same file) -> cache hit
            mock_invoke.reset_mock()
            builder._process_files([csv_file], {})
            assert mock_invoke.call_count == 0

            # Modify file
            time.sleep(0.05)
            csv_file.write_text("a,b,c\n1,2,3\n")

            # Third run (modified file) -> cache miss
            mock_invoke.reset_mock()
            caplog.clear()
            builder._process_files([csv_file], {})
            assert mock_invoke.call_count == 1
            assert "cached=0" in caplog.text
