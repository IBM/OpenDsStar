import pandas as pd
import streamlit as st

from OpenDsStar.ui.agentic_results_viewer import agentic_viewer_utils

st.set_page_config(page_title="Output Comparator", layout="wide")

st.sidebar.title("Comparison Sources")
path_a = st.sidebar.text_input(
    "Path Source A (Baseline):", placeholder="path/to/results_A"
)
path_b = st.sidebar.text_input(
    "Path Source B (Comparison):", placeholder="path/to/results_B"
)

# 1. Load Data
df_a, df_b = pd.DataFrame(), pd.DataFrame()
if path_a:
    data_a, c_a = agentic_viewer_utils.load_data_from_path(path_a)
    if c_a:
        df_a = agentic_viewer_utils.process_dataframe(data_a)
        st.sidebar.success(f"A: {len(df_a)} items")
if path_b:
    data_b, c_b = agentic_viewer_utils.load_data_from_path(path_b)
    if c_b:
        df_b = agentic_viewer_utils.process_dataframe(data_b)
        st.sidebar.success(f"B: {len(df_b)} items")

if df_a.empty and df_b.empty:
    st.info("👈 Please enter comparison paths in the sidebar.")
    st.stop()

# 2. Logic (Venn)
ids_a = set(df_a["q_id"]) if not df_a.empty else set()
ids_b = set(df_b["q_id"]) if not df_b.empty else set()
common = list(ids_a & ids_b)

st.sidebar.divider()
st.sidebar.markdown(f"**Common Questions:** {len(common)}")
st.sidebar.markdown(f"**Unique to A:** {len(ids_a - ids_b)}")
st.sidebar.markdown(f"**Unique to B:** {len(ids_b - ids_a)}")

# 3. Filters
st.sidebar.header("Comparison Filters")
if st.sidebar.button("Reset Filters"):
    for k in ["s_a", "s_b", "delta"]:
        st.session_state[k] = (0.0, 1.0) if k != "delta" else (-1.0, 1.0)
    st.rerun()

min_a, max_a = st.sidebar.slider("Score Range A", 0.0, 1.0, (0.0, 1.0), 0.05, key="s_a")
min_b, max_b = st.sidebar.slider("Score Range B", 0.0, 1.0, (0.0, 1.0), 0.05, key="s_b")
d_min, d_max = st.sidebar.slider(
    "Delta (A-B)", -1.0, 1.0, (-1.0, 1.0), 0.1, key="delta"
)

# 4. Compare DF Construction
tab_over, tab_insp = st.tabs(["📊 Comparative Dashboard", "🔍 Side-by-Side Inspection"])

with tab_over:
    if not common:
        st.warning("No common questions found to compare.")
    else:
        # Prepare subset
        da = df_a[df_a["q_id"].isin(common)].set_index("q_id")[
            ["_ui_score", "input_tokens", "_ui_steps_count", "question"]
        ]
        db = df_b[df_b["q_id"].isin(common)].set_index("q_id")[
            ["_ui_score", "input_tokens", "_ui_steps_count"]
        ]

        comp = da.join(db, lsuffix="_a", rsuffix="_b")
        comp["delta"] = comp["_ui_score_a"] - comp["_ui_score_b"]

        # Apply Filters
        mask = (
            (comp["_ui_score_a"].between(min_a, max_a))
            & (comp["_ui_score_b"].between(min_b, max_b))
            & (comp["delta"].between(d_min, d_max))
        )
        res = comp[mask]

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions Matched", len(res))
        c2.metric(
            "Avg Score A vs B",
            f"{res['_ui_score_a'].mean():.2f} vs {res['_ui_score_b'].mean():.2f}",
            delta=f"{res['delta'].mean():.2f}",
        )
        c3.metric(
            "Avg Steps A vs B",
            f"{res['_ui_steps_count_a'].mean():.1f} vs {res['_ui_steps_count_b'].mean():.1f}",
        )
        c4.metric(
            "Avg Tokens A vs B",
            f"{res['input_tokens_a'].mean():.0f} vs {res['input_tokens_b'].mean():.0f}",
        )

        st.divider()
        st.subheader("Comparison List")

        display_df = res.reset_index()[
            ["q_id", "_ui_score_a", "_ui_score_b", "delta", "question"]
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "_ui_score_a": st.column_config.ProgressColumn(
                    "Score A", min_value=0, max_value=1, format="%.2f"
                ),
                "_ui_score_b": st.column_config.ProgressColumn(
                    "Score B", min_value=0, max_value=1, format="%.2f"
                ),
                "delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                "question": st.column_config.TextColumn("Question", width="large"),
            },
        )

with tab_insp:
    # Selector Options: Prioritize filtered common list, fallback to any available
    target_ids = res.index.tolist() if "res" in locals() and not res.empty else common
    if not target_ids:
        target_ids = list(ids_a | ids_b)

    opts = {}
    for qid in target_ids:
        sa = df_a[df_a["q_id"] == qid]["_ui_score"].values[0] if qid in ids_a else 0
        sb = df_b[df_b["q_id"] == qid]["_ui_score"].values[0] if qid in ids_b else 0
        q_text = (
            df_a[df_a["q_id"] == qid]["question"].values[0]
            if qid in ids_a
            else "Unknown Question"
        )
        opts[qid] = f"[{qid}] A:{sa:.2f} | B:{sb:.2f} | {q_text[:50]}..."

    if opts:
        col_sel, _ = st.columns([2, 1])
        with col_sel:
            sel_qid = st.selectbox(
                "Select Question:", opts.keys(), format_func=lambda x: opts[x]
            )

        st.divider()
        st.header(f"QID: {sel_qid}")

        col_a, col_b = st.columns(2)

        def render_side(col, df_source, title, key_prefix):
            with col:
                st.subheader(title)
                if sel_qid in df_source["q_id"].values:
                    item = df_source[df_source["q_id"] == sel_qid].iloc[0].to_dict()

                    # Mini Metrics
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Score", f"{item['_ui_score']:.2f}")
                    mc2.metric("Tokens", item.get("input_tokens", 0))

                    st.success(f"**Answer:** {item.get('answer', 'No answer')}")
                    agentic_viewer_utils.render_error_analysis(
                        item.get("error_analysis_results")
                    )

                    st.markdown("---")
                    st.markdown("**Trajectory**")
                    agentic_viewer_utils.render_trajectory(
                        item.get("trajectory", []), key_prefix=key_prefix
                    )
                else:
                    st.warning("Question not found in this source.")

        render_side(col_a, df_a, "Source A (Baseline)", "A_")
        render_side(col_b, df_b, "Source B (Comparison)", "B_")
    else:
        st.warning("No questions available to inspect.")
