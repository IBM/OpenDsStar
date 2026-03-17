#!/usr/bin/env python3
"""
Script to print all document descriptions from the file description cache.

Usage:
    python print_file_descriptions_cache.py
"""

from pathlib import Path

import diskcache as dc

# Cache directory path
CACHE_DIR = Path(
    "/Users/yoavkantor/projects/OpenDsStar/src/experiments/benchmarks/databench/cache/file_descriptions_GCP_gemini-2_5-flash"
)


def print_all_descriptions():
    """Print all document descriptions from the cache."""

    if not CACHE_DIR.exists():
        print(f"❌ Cache directory not found: {CACHE_DIR}")
        return

    print(f"📁 Reading cache from: {CACHE_DIR}")
    print("=" * 80)
    print()

    # Open the diskcache
    cache = dc.Cache(str(CACHE_DIR))

    try:
        # Get all keys
        keys = list(cache.iterkeys())

        if not keys:
            print("⚠️  Cache is empty - no descriptions found")
            return

        print(f"Found {len(keys)} cached descriptions\n")
        print("=" * 80)

        # Iterate through all cached items
        for i, key in enumerate(keys, 1):
            value = cache.get(key)

            # Extract document name from key (format: docname_hash)
            doc_name = key.rsplit("_", 1)[0].replace("_", " ")

            print(f"\n[{i}/{len(keys)}] Document: {doc_name}")
            print(f"Cache Key: {key}")
            print("-" * 80)

            if value:
                # Print the description
                print(f"Description:\n{value}")
            else:
                print("⚠️  No description found (None value)")

            print("=" * 80)

        print(f"\n✅ Printed {len(keys)} descriptions")

    finally:
        cache.close()


if __name__ == "__main__":
    print_all_descriptions()
