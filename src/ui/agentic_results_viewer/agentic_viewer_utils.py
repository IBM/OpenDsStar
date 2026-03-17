import ast
import datetime
import json
import os
import re
from typing import Any, Dict

import pandas as pd
import streamlit as st

# --- 1. Data Loading ---


def _load_single_file(file_path):
    """Internal helper: Loads a single JSON or JSONL file and extracts items."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            try:
                # Attempt 1: Standard JSON
                parsed = json.loads(content)

                # Handle new _output.json format with "items" field
                if isinstance(parsed, dict) and "items" in parsed:
                    # Extract items and flatten the structure
                    for item in parsed["items"]:
                        flattened = _flatten_item(item)
                        data.append(flattened)
                    return data

                # Fallback: old format
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                # Attempt 2: JSON Lines
                f.seek(0)
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except Exception:
                            continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    return data


def _flatten_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the new _output.json item structure into a flat dict for the viewer.

    New structure:
    {
      "question_id": "...",
      "output": {
        "question_id": "...",
        "answer": "...",
        "metadata": {
          "question": "...",
          "trajectory": [...],
          "input_tokens": ...,
          "output_tokens": ...,
          ...
        }
      },
      "evaluations": [
        {
          "question_id": "...",
          "score": 0.9,
          "passed": true,
          "details": {...}
        }
      ]
    }
    """
    flattened = {}

    # Extract question_id
    flattened["q_id"] = item.get("question_id", "unknown")

    # Extract output fields
    output = item.get("output", {})
    if output:
        flattened["answer"] = output.get("answer", "")

        # Extract metadata fields
        metadata = output.get("metadata", {})
        flattened["question"] = metadata.get("question", "")
        flattened["trajectory"] = metadata.get("trajectory", [])
        flattened["input_tokens"] = metadata.get("input_tokens", 0)
        flattened["output_tokens"] = metadata.get("output_tokens", 0)

        # Extract ground truths if available
        ground_truth = metadata.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            flattened["ground_truths"] = ground_truth.get("answers", [])
        else:
            flattened["ground_truths"] = []

        # Extract error analysis if available
        flattened["error_analysis_results"] = metadata.get("error_analysis_results")
    else:
        # No output available
        flattened["answer"] = ""
        flattened["question"] = ""
        flattened["trajectory"] = []
        flattened["input_tokens"] = 0
        flattened["output_tokens"] = 0
        flattened["ground_truths"] = []
        flattened["error_analysis_results"] = None

    # Extract evaluation score (use first evaluator's score)
    evaluations = item.get("evaluations", [])
    if evaluations and len(evaluations) > 0:
        first_eval = evaluations[0]
        flattened["score"] = first_eval.get("score", 0.0)
        flattened["metrics"] = {
            "rag": {
                "external_rag": {
                    "answer_correctness": {
                        "llama_3_3_70b_instruct_watsonx_judge": first_eval.get(
                            "score", 0.0
                        )
                    }
                }
            }
        }
    else:
        flattened["score"] = 0.0
        flattened["metrics"] = {}

    return flattened


@st.cache_data(show_spinner=True)
def load_data_from_path(path_input):
    """
    Master loader: Handles either a single file OR a directory recursion.
    - Single File: Loads it directly (ignoring name).
    - Directory: Recursively finds 'PatternX.json'.
    """
    all_data = []
    path_input = path_input.strip().strip('"').strip("'")

    if not os.path.exists(path_input):
        return [], 0

    files_to_process = []

    # Case A: Specific File (Load directly)
    if os.path.isfile(path_input):
        files_to_process.append(path_input)
    # Case B: Directory (Regex filter for _output.json files)
    elif os.path.isdir(path_input):
        # Match files ending with _output.json (experiment output files)
        pattern_regex = re.compile(r".*_output\.json$", re.IGNORECASE)
        for root, dirs, files in os.walk(path_input):
            for file in files:
                if pattern_regex.match(file):
                    files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        return [], 0

    for fpath in files_to_process:
        file_data = _load_single_file(fpath)

        # Calculate relative source path for display
        if os.path.isdir(path_input):
            display_source = os.path.relpath(fpath, start=path_input)
        else:
            display_source = os.path.basename(fpath)

        for record in file_data:
            record["_source_file"] = display_source

        all_data.extend(file_data)

    return all_data, len(files_to_process)


# --- 2. Data Processing ---

KNOWN_SCORE_KEYS = [
    "metrics.rag.external_rag.answer_correctness.llama_3_3_70b_instruct_watsonx_judge",
    "score",
]


def extract_score(row):
    """Extracts score from various known locations/formats."""
    for key in KNOWN_SCORE_KEYS:
        val = row.get(key)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass

    if "metrics" in row and isinstance(row["metrics"], dict):
        try:
            return float(
                row["metrics"]["rag"]["external_rag"]["answer_correctness"][
                    "llama_3_3_70b_instruct_watsonx_judge"
                ]
            )
        except Exception:
            pass
    return 0.0


def process_dataframe(data_list):
    """Converts list of dicts to a standardized DataFrame with helper columns."""
    if not data_list:
        return pd.DataFrame()
    df = pd.DataFrame(data_list)
    df["_ui_score"] = df.apply(extract_score, axis=1)
    df["_ui_steps_count"] = df["trajectory"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    if "_source_file" not in df.columns:
        df["_source_file"] = "Single File"
    if "q_id" in df.columns:
        df["q_id"] = df["q_id"].astype(str)

    return df


# --- 3. UI Renderers ---


def render_ds_star_step(i, step, key_prefix=""):
    """Renderer for structured Dictionary steps (DS-Star)."""
    node_name = step.get("node", "unknown_node")
    raw_time = step.get("time", 0)
    try:
        readable_time = datetime.datetime.fromtimestamp(raw_time).strftime("%H:%M:%S")
    except Exception:
        readable_time = str(raw_time)

    icon = "🔄"
    if "plan" in node_name:
        icon = "🧠"
    elif "code" in node_name:
        icon = "💻"
    elif "execute" in node_name:
        icon = "⚙️"
    elif "verify" in node_name:
        status = step.get("sufficient")
        icon = " ✅" if status is True else " ❌" if status is False else "✅"
    elif "finalizer" in node_name:
        icon = "🏁"

    with st.expander(f"Step {i + 1}: {icon} {node_name} ({readable_time})"):
        last_step = step.get("last_step")
        if last_step:
            st.caption("Context from Previous Step:")
            ls_tabs = st.tabs(["Plan", "Code", "Logs", "Outputs", "Verifier"])
            with ls_tabs[0]:
                st.info(last_step.get("plan") or "No plan")
            with ls_tabs[1]:
                st.code(last_step.get("code") or "# No code", language="python")

            # UNIQUE KEY GENERATION: key_prefix + loop_index
            with ls_tabs[2]:
                st.text_area(
                    "Logs",
                    last_step.get("logs") or "",
                    height=150,
                    key=f"{key_prefix}logs_{i}",
                )

            with ls_tabs[3]:
                st.json(last_step.get("outputs") or {})
            with ls_tabs[4]:
                st.write(f"**Sufficient:** {last_step.get('verifier_sufficient')}")
                st.write(last_step.get("verifier_explanation"))
        st.divider()
        st.markdown("**Details:**")
        st.json({k: v for k, v in step.items() if k != "last_step"})


def render_string_step(i, step_str, key_prefix=""):
    """Renderer for String steps (CodeAct/ReAct)."""
    match = re.match(r"^trajectory\s+(.*?)\s+-\s+(.*)$", step_str, re.DOTALL)
    if match:
        actor_raw = match.group(1).strip()
        content = match.group(2).strip()
        if "User" in actor_raw:
            actor = "User 👤"
            icon = "👤"
        elif "AI" in actor_raw:
            actor = "AI 🤖"
            icon = "🤖"
        elif "Tool" in actor_raw:
            actor = "Tool 🛠️"
            icon = "🛠️"
        else:
            actor = actor_raw
            icon = "📝"
    else:
        actor = "Unknown"
        content = step_str
        icon = "📝"

    # Try parsing content
    formatted = None
    is_json = False
    try:
        formatted = json.loads(content)
        is_json = True
    except Exception:
        try:
            if content.strip().startswith("{") or content.strip().startswith("["):
                formatted = ast.literal_eval(content)
                is_json = True
            else:
                is_json = False
        except Exception:
            is_json = False

    with st.expander(f"Step {i + 1}: {icon} {actor}"):
        if is_json and formatted is not None:
            st.json(formatted)
        elif "```" in content:
            st.markdown(content)
        else:
            st.info(content)


def render_trajectory(trajectory, key_prefix=""):
    """Dispatcher for trajectory rendering."""
    if not trajectory:
        st.write("No trajectory data.")
        return

    for i, step in enumerate(trajectory):
        if isinstance(step, dict):
            render_ds_star_step(i, step, key_prefix)
        elif isinstance(step, str):
            render_string_step(i, step, key_prefix)
        else:
            with st.expander(f"Step {i + 1}: Unknown Format"):
                st.write(step)


def render_error_analysis(error_analysis):
    """Renders the Error Analysis block with metrics."""
    if error_analysis and isinstance(error_analysis, dict):
        st.divider()
        st.subheader("Error Analysis")
        c1, c2, c3 = st.columns(3)

        run_ok = error_analysis.get("was_run_successful")
        ans_ok = error_analysis.get("is_answer_correct")
        can_imp = error_analysis.get("can_trajectory_be_improved")

        c1.metric(
            "Run Success",
            "Yes" if run_ok else "No",
            delta="✅" if run_ok else "❌",
            delta_color="normal" if run_ok else "inverse",
        )
        c2.metric(
            "Ans Correct",
            "Yes" if ans_ok else "No",
            delta="✅" if ans_ok else "❌",
            delta_color="normal" if ans_ok else "inverse",
        )
        c3.metric(
            "Improve?",
            "Yes" if can_imp else "No",
            delta="⚠️" if can_imp else "✅",
            delta_color="inverse" if can_imp else "normal",
        )

        with st.expander("Analysis Details", expanded=True):
            fields = [
                ("was_run_successful_explanation", "Run Success Explanation"),
                ("is_answer_correct_explanation", "Answer Correctness Explanation"),
                (
                    "can_trajectory_be_improved_explanation",
                    "Trajectory Improvement Details",
                ),
                ("execution_errors_descriptions", "Execution Errors"),
                ("anything_else_the_agent_developer_should_know", "Developer Notes"),
            ]
            for key, label in fields:
                val = error_analysis.get(key)
                if val and str(val).lower() != "none":
                    st.markdown(f"**{label}:**")
                    st.info(val)

            # Show extras
            known = {k for k, _ in fields} | {
                "was_run_successful",
                "is_answer_correct",
                "can_trajectory_be_improved",
            }
            extras = {k: v for k, v in error_analysis.items() if k not in known}
            if extras:
                st.markdown("**Additional Info:**")
                st.json(extras)
