"""Test script to verify parameter loading logic."""

import json
from pathlib import Path

# Find an existing params file
params_file = Path(
    "src/experiments/benchmarks/hotpotqa/output/result_ds_star_wx_mistral_medium_e2e_3_20260215_085412_params.json"
)

# Skip test if file doesn't exist
if not params_file.exists():
    print(f"⚠️  Params file not found: {params_file}")
    print("This test requires an existing params file from a previous experiment run.")
    exit(0)

with open(params_file, "r") as f:
    data = json.load(f)

print(f"Loading experiment parameters from: {params_file}")

# Extract actual parameters from the nested structure
if "parameters" in data:
    params = data["parameters"]
    print(f"Experiment: {data.get('experiment_class', 'Unknown')}")
    print(f"Original run_id: {data.get('run_id', 'Unknown')}")
    print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
else:
    # Fallback for old format without nesting
    params = data

print("\nExtracted Parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")

print("\n✅ Parameter extraction successful!")
print("These parameters can now be passed to run_hotpotqa_experiment()")
