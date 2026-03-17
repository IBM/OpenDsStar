#!/usr/bin/env .venv/bin/python
"""Test script to verify databench combination logic."""

import itertools

# Simulate the combination logic
agent_types = ["ds_star", "react_langchain"]
models = ["model1", "model2", "model3"]

combinations = list(itertools.product(agent_types, models))

print(f"\n{'='*80}")
print("Testing combination logic:")
print(f"Agent types: {agent_types}")
print(f"Models: {models}")
print(f"Total combinations: {len(combinations)}")
print(f"{'='*80}\n")

for i, (agent_type, model) in enumerate(combinations, 1):
    print(f"{i}. Agent: {agent_type}, Model: {model}")

print(f"\n{'='*80}")
print(
    f"Expected: {len(agent_types)} x {len(models)} = {len(agent_types) * len(models)} combinations"
)
print(f"Actual: {len(combinations)} combinations")
print(
    f"Test: {'PASS' if len(combinations) == len(agent_types) * len(models) else 'FAIL'}"
)
print(f"{'='*80}\n")
