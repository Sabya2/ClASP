"""Render group comparisons from shared application state."""

import streamlit as st

from srcs.utils_features.common import get_state
from srcs.utils_visualisation.group_comparison_plots import (
    render_group_comparison_page,
    render_summary_grid,
)

st.set_page_config(layout="wide")

st.title("Group Comparison")

state = get_state()
df = state["df"]
continuous_cols = state.get("continuous_cols", [])
group_col = state.get("group_col")

if not group_col:
    st.warning("Please select a group column in the settings.")
    st.stop()

summaries, cols_to_plot, grid_cols = render_group_comparison_page(
    df=df,
    continuous_cols=continuous_cols,
    group_col=group_col,
)

show_summary = st.checkbox(
    "Summary of group comparisons",
    value=False,
    key="Group_summary_checkbox",
)

if show_summary and summaries:
    render_summary_grid(
        summaries=summaries,
        cols_to_plot=cols_to_plot,
        ncols=grid_cols,
    )
