#!/usr/bin/env python3
"""
Utility tool to load and run experiments from params JSON files.

This tool loads any experiment type from a Recreatable format params file
and runs it with the saved configuration.

Usage:
    python -m experiments.utils.run_experiment_by_params_file <path_to_params_file>

Or from project root:
    python src/experiments/utils/run_experiment_by_params_file.py <path_to_params_file>
"""

import sys
from pathlib import Path


def load_and_run(params_file: str):
    """
    Load and run experiment from params file.

    Automatically detects the experiment type and loads it.
    Works with any experiment that inherits from BaseExperiment.

    Args:
        params_file: Path to the params JSON file

    Returns:
        Tuple of (outputs, results) from the experiment
    """
    import json

    from ..base.base_experiment import BaseExperiment

    print("=" * 80)
    print("LOADING EXPERIMENT FROM PARAMS FILE")
    print("=" * 80)

    # Check file format and fix __main__ references if needed
    from pathlib import Path

    params_path = Path(params_file)

    with open(params_path) as f:
        data = json.load(f)

    if "type" not in data or "args" not in data:
        print("\n✗ Error: This file is not in the new Recreatable format.")
        print("Expected format:")
        print('  {"type": "module.path.ExperimentClass", "args": {...}}')
        sys.exit(1)

    print(f"\nParams file: {params_file}")
    print(f"Experiment type: {data['type']}")
    print(f"Parameters: {data['args']}")

    # Fix incorrect module paths (happens when experiment is run as script or with wrong imports)
    needs_fix = False
    original_type = data["type"]

    # Fix __main__ references by searching for the class
    if data["type"].startswith("__main__."):
        class_name = data["type"].split(".")[-1]
        print(f"\n⚠ Fixing __main__ reference for {class_name}...")
        needs_fix = True

        # Search for the class in the experiments package
        import importlib
        import inspect
        import pkgutil

        from ..base.base_experiment import BaseExperiment

        found_class = None
        # Search in experiments.benchmarks
        experiments_pkg = importlib.import_module("experiments.benchmarks")

        for importer, modname, ispkg in pkgutil.walk_packages(
            path=experiments_pkg.__path__,
            prefix="experiments.benchmarks.",
            onerror=lambda x: None,
        ):
            try:
                module = importlib.import_module(modname)
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    # Verify it's a BaseExperiment subclass
                    if (
                        inspect.isclass(cls)
                        and issubclass(cls, BaseExperiment)
                        and cls is not BaseExperiment
                    ):
                        found_class = f"{modname}.{class_name}"
                        break
            except Exception:
                continue

        if found_class:
            data["type"] = found_class
        else:
            print(f"✗ Could not find experiment class: {class_name}")
            print("Make sure it's a BaseExperiment subclass in experiments.benchmarks")
            sys.exit(1)

    # Fix src.experiments prefix (should be just experiments)
    elif data["type"].startswith("src.experiments."):
        print("\n⚠ Fixing 'src.experiments' prefix...")
        data["type"] = data["type"].replace("src.experiments.", "experiments.")
        needs_fix = True

    # Save the fixed version back to file if needed
    if needs_fix:
        with open(params_path, "w") as f:
            json.dump(data, f, indent=2)
        print("✓ Updated params file:")
        print(f"  From: {original_type}")
        print(f"  To:   {data['type']}")

    # Load the experiment using BaseExperiment.load_instance
    # This automatically resolves the correct experiment class
    print("\n" + "=" * 80)
    print("LOADING EXPERIMENT")
    print("=" * 80)

    experiment = BaseExperiment.load_instance(params_file)

    print(f"\n✓ Loaded: {experiment.__class__.__name__}")
    print(f"✓ Module: {experiment.__class__.__module__}")
    print(f"✓ Config: {experiment.get_config()}")

    # Run the experiment
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT")
    print("=" * 80 + "\n")

    outputs, results = experiment.experiment_main()
    return outputs, results


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print(
            "Usage: python -m experiments.utils.run_experiment_by_params_file <path_to_params_file>"
        )
        print("\nOr from project root:")
        print(
            "  python src/experiments/utils/run_experiment_by_params_file.py <path_to_params_file>"
        )
        print("\nExample:")
        print(
            "  python -m experiments.utils.run_experiment_by_params_file src/experiments/benchmarks/hotpotqa/output/result_ds_star_wx_llama_maverick_hotpotqa_3_20260129_164021_params.json"
        )
        sys.exit(1)

    params_file = sys.argv[1]

    if not Path(params_file).exists():
        print(f"Error: File not found: {params_file}")
        sys.exit(1)

    try:
        outputs, results = load_and_run(params_file)
        print("\n✓ Success!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
