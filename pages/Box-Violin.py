"""Render box and violin plots from shared application state."""

import streamlit as st

from srcs.utils_features.common import get_state
from srcs.utils_visualisation.distribution_plots import (
    render_box_violin_page,
    render_interactive_box_violin_plot,
)


# Read shared state
state = get_state()
df = state["df"]
group_col = state["group_col"]
continuous_cols = state["continuous_cols"]
participants_col = state["participants_col"]

st.markdown("### 1. Box and violin plots")

if df is not None and len(continuous_cols) >= 2:
    render_box_violin_page(
        df=df,
        continuous_cols=continuous_cols,
        group_col=group_col
    )
else:
    st.info("Upload data and select at least two continuous variables.")
st.markdown("---")
st.subheader("2. Interactive box/violin plot")

interactive_pair = st.checkbox(
    "Interactive box/violin plot",
    value=False,
    help="Toggle to enable interactive box/violin plot for continuous variables.",
)

if interactive_pair and df is not None and len(continuous_cols) >= 2:
    render_interactive_box_violin_plot(
        df=df,
        continuous_cols=continuous_cols,
        group_col=group_col,
        participants_col=participants_col,
    )
