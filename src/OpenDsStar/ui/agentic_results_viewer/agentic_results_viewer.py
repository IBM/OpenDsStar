import streamlit as st

from OpenDsStar.ui.agentic_results_viewer import agentic_viewer_utils

st.set_page_config(page_title="Output Viewer", layout="wide")

st.sidebar.title("Configuration")
input_path = st.sidebar.text_input(
    "Path (File or Directory):", placeholder="path/to/results/"
)

# 1. Load Data
data = []
if input_path:
    data, file_count = agentic_viewer_utils.load_data_from_path(input_path)
    if file_count > 0:
        st.sidebar.success(f"Loaded {len(data)} records from {file_count} file(s).")
    else:
        st.sidebar.error("No valid data found.")

if not data:
    st.info("👈 Please enter a path in the sidebar.")
    st.stop()

# 2. Process
df = agentic_viewer_utils.process_dataframe(data)

# 3. Filters
st.sidebar.divider()
st.sidebar.header("Filters")
if st.sidebar.button("Clear Filters"):
    st.session_state["score_slider"] = (0.0, 1.0)
    st.session_state["file_multiselect"] = []
    st.rerun()

min_score, max_score = st.sidebar.slider(
    "Score Range", 0.0, 1.0, (0.0, 1.0), 0.05, key="score_slider"
)

all_files = sorted(df["_source_file"].unique().tolist())
has_multiple_files = len(all_files) > 1
selected_files = []
if has_multiple_files:
    selected_files = st.sidebar.multiselect(
        "Source File", all_files, default=[], key="file_multiselect"
    )

# 4. Apply Filters
mask = (df["_ui_score"] >= min_score) & (df["_ui_score"] <= max_score)
if selected_files:
    mask = mask & (df["_source_file"].isin(selected_files))
filtered_df = df[mask]
st.sidebar.markdown(f"**Matches:** {len(filtered_df)} / {len(df)}")

# 5. UI Layout
if filtered_df.empty:
    st.warning("No questions match filters.")
else:
    tab_agg, tab_detail = st.tabs(["📊 Aggregation", "🔍 Detailed Inspection"])

    # -- Tab 1: Aggregation --
    with tab_agg:
        st.header("Overview")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Count", len(filtered_df))
        m2.metric("Avg Score", f"{filtered_df['_ui_score'].mean():.2f}")
        m3.metric("Avg Steps", f"{filtered_df['_ui_steps_count'].mean():.1f}")
        m4.metric("Avg In Tok", f"{filtered_df['input_tokens'].mean():.0f}")
        m5.metric("Avg Out Tok", f"{filtered_df['output_tokens'].mean():.0f}")
        st.divider()

        display_cols = [
            "q_id",
            "_source_file",
            "_ui_score",
            "question",
            "_ui_steps_count",
            "input_tokens",
            "output_tokens",
        ]
        if not has_multiple_files and "_source_file" in display_cols:
            display_cols.remove("_source_file")

        # Safe selection of columns
        valid_cols = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[valid_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "_ui_score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=1, format="%.2f"
                ),
                "_source_file": st.column_config.TextColumn("Source", width="medium"),
            },
        )

    # -- Tab 2: Inspection --
    with tab_detail:
        # Build selector options
        options = {
            row[
                "q_id"
            ]: f"[{row['q_id']}] ({row['_ui_score']:.2f}) {row.get('question', '')[:50]}..."
            for _, row in filtered_df.iterrows()
        }

        c_sel, _ = st.columns([2, 1])
        with c_sel:
            qid = st.selectbox(
                "Select Question:",
                options=options.keys(),
                format_func=lambda x: options[x],
            )

        item = df[df["q_id"] == qid].iloc[0].to_dict()

        st.title(f"QID: {item.get('q_id')} ({item.get('_ui_score', 0):.2f})")
        st.caption(f"Source: {item.get('_source_file')}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input Tokens", item.get("input_tokens", 0))
        c2.metric("Output Tokens", item.get("output_tokens", 0))
        c3.metric("Score", f"{item.get('_ui_score', 0):.2f}")
        c4.metric("Steps", len(item.get("trajectory", [])))
        st.divider()

        c1, c2 = st.columns([1, 1])
        with c1:
            st.info(f"**Question:** {item.get('question')}")
            with st.expander("Ground Truths"):
                for gt in item.get("ground_truths", []):
                    st.warning(gt)
        with c2:
            st.success(f"**Answer:** {item.get('answer')}")

        agentic_viewer_utils.render_error_analysis(item.get("error_analysis_results"))

        st.header("Trajectory")
        agentic_viewer_utils.render_trajectory(
            item.get("trajectory", []), key_prefix="single_"
        )
