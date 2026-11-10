"""Render group comparisons from shared application state."""

import streamlit as st

from srcs.utils_features.common import get_state
from srcs.utils_visualisation.group_comparison_plots import (
    render_group_comparison_page,
    render_summary_grid,
)
from srcs.utils_visualisation.repeated_measures_plots import render_longitudinal_comparison_page

st.set_page_config(layout="wide")

st.title("Group Comparison")

state = get_state()
df = state["df"]
continuous_cols = state.get("continuous_cols", [])
group_col = state.get("group_col")
participants_col = state.get("participants_col")

analysis_type = st.selectbox(
    "Analysis Type",
    ["Independent Groups", "Paired / Repeated Measures", "Group × Time (Mixed Design)"],
    help="Choose the design that matches how observations are related.",
    key="group_comparison_analysis_type",
)

if analysis_type == "Independent Groups":
    if not group_col:
        st.warning("Please select a group column in the settings.")
        st.stop()
    summaries, cols_to_plot, grid_cols = render_group_comparison_page(
        df=df,
        continuous_cols=continuous_cols,
        group_col=group_col,
    )
elif analysis_type == "Paired / Repeated Measures":
    summaries, cols_to_plot, grid_cols = render_longitudinal_comparison_page(
        df=df,
        continuous_cols=continuous_cols,
        group_col=group_col,
        participants_col=participants_col,
        mixed=False,
    )
else:
    summaries, cols_to_plot, grid_cols = render_longitudinal_comparison_page(
        df=df,
        continuous_cols=continuous_cols,
        group_col=group_col,
        participants_col=participants_col,
        mixed=True,
    )

show_summary = st.checkbox(
    "Statistical summaries",
    value=False,
    key="Group_summary_checkbox",
)

if show_summary and summaries:
    render_summary_grid(
        summaries=summaries,
        cols_to_plot=cols_to_plot,
        ncols=grid_cols,
    )
