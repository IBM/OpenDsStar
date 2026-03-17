"""Test script for data reader caching functionality."""

import logging
import time
from pathlib import Path

import pytest

from src.experiments.benchmarks.hotpotqa.data_reader import HotpotQADataReader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_cache():
    """Test that data reader caching works correctly."""

    # Clean up any existing cache
    cache_dir = Path("./cache/data_readers")
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
        logger.info("Cleaned up existing cache")

    # Test 1: First load (should load from source)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: First load (should load from source)")
    logger.info("=" * 60)

    reader1 = HotpotQADataReader(
        split="test", question_limit=5, document_factor=2, seed=42, use_cache=True
    )

    start_time = time.time()
    reader1.read_data()
    first_load_time = time.time() - start_time

    corpus1 = reader1.get_data()
    benchmark1 = reader1.get_benchmark()

    logger.info(f"First load completed in {first_load_time:.2f} seconds")
    logger.info(f"Corpus size: {len(corpus1)}")
    logger.info(f"Benchmark size: {len(benchmark1)}")

    # Test 2: Second load (should load from cache)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Second load (should load from cache)")
    logger.info("=" * 60)

    reader2 = HotpotQADataReader(
        split="test", question_limit=5, document_factor=2, seed=42, use_cache=True
    )

    start_time = time.time()
    reader2.read_data()
    second_load_time = time.time() - start_time

    corpus2 = reader2.get_data()
    benchmark2 = reader2.get_benchmark()

    logger.info(f"Second load completed in {second_load_time:.2f} seconds")
    logger.info(f"Corpus size: {len(corpus2)}")
    logger.info(f"Benchmark size: {len(benchmark2)}")

    # Test 3: Verify cache speedup
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Verify cache speedup")
    logger.info("=" * 60)

    speedup = (
        first_load_time / second_load_time if second_load_time > 0 else float("inf")
    )
    logger.info(f"Cache speedup: {speedup:.2f}x faster")

    if speedup > 2:
        logger.info("✓ Cache is working! Second load was significantly faster.")
    else:
        logger.warning("⚠ Cache may not be working optimally. Speedup is less than 2x.")

    # Test 4: Verify data consistency
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Verify data consistency")
    logger.info("=" * 60)

    if len(corpus1) == len(corpus2) and len(benchmark1) == len(benchmark2):
        logger.info("✓ Data sizes match between cached and non-cached loads")
    else:
        logger.error("✗ Data sizes don't match!")

    # Test 5: Load with different parameters (should not use cache)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Different parameters (should not use cache)")
    logger.info("=" * 60)

    reader3 = HotpotQADataReader(
        split="test",
        question_limit=3,  # Different limit
        document_factor=2,
        seed=42,
        use_cache=True,
    )

    start_time = time.time()
    reader3.read_data()
    third_load_time = time.time() - start_time

    corpus3 = reader3.get_data()
    benchmark3 = reader3.get_benchmark()

    logger.info(f"Third load completed in {third_load_time:.2f} seconds")
    logger.info(f"Corpus size: {len(corpus3)}")
    logger.info(f"Benchmark size: {len(benchmark3)}")

    if len(benchmark3) != len(benchmark1):
        logger.info("✓ Different parameters resulted in different data")
    else:
        logger.warning("⚠ Different parameters resulted in same data size")

    # Test 6: Verify cache directory structure
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Verify cache directory structure")
    logger.info("=" * 60)

    if cache_dir.exists():
        cache_entries = list(cache_dir.iterdir())
        logger.info(f"Cache directory contains {len(cache_entries)} entries:")
        for entry in cache_entries:
            logger.info(f"  - {entry.name}")
        logger.info("✓ Cache directory created successfully")
    else:
        logger.error("✗ Cache directory not found!")

    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_cache()
