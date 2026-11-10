"""Render standalone interactive scatter and box plots."""

import streamlit as st

from srcs.utils_features.common import get_state
from srcs.utils_visualisation.regression_analysis import (
    render_interactive_box_plot,
    render_interactive_linear_plot,
)


state = get_state()
df = state["df"]
group_col = state["group_col"]
continuous_cols = state["continuous_cols"]
categorical_cols = state["categorical_cols"]
participants_col = state["participants_col"]

st.markdown("## Interactive Plots")
st.caption(
    "Explore continuous relationships and distributions with hoverable, "
    "interactive figures. These plots are descriptive and do not change any "
    "analysis on another page."
)

with st.expander("How this page works", expanded=False):
    scatter_help, box_help = st.columns([1, 1], border=True, gap="medium")
    with scatter_help:
        st.markdown(
            """
            ### Linear scatter plot

            - Select continuous X and Y variables.
            - Hover over observations to inspect their values or participant ID.
            - Display an OLS trend line for the selected relationship.
            - Optionally color observations by a grouping variable.

            The trend summarizes an unadjusted bivariate relationship and does not
            establish causality.
            """
        )
    with box_help:
        st.markdown(
            """
            ### Box plot

            - Select one continuous variable.
            - Optionally compare its distribution across a categorical variable.
            - Show or hide individual observations.
            - Hover over points for additional information.

            The median is shown inside the box, while the box covers the middle 50%
            of observations. Group differences are descriptive unless supported by
            an appropriate statistical test.
            """
        )

plot_type = st.pills(
    "Choose an interactive plot",
    options=["Linear scatter plot", "Box plot"],
    default="Linear scatter plot",
    selection_mode="single",
    key="interactive_plots_page_type",
)

if df is None:
    st.info("Upload data to use the interactive plots.")
elif plot_type == "Linear scatter plot":
    if len(continuous_cols) < 2:
        st.info("Select at least two continuous variables for a scatter plot.")
    else:
        render_interactive_linear_plot(
            df=df,
            continuous_cols=continuous_cols,
            group_col=group_col,
            participants_col=participants_col,
        )
elif plot_type == "Box plot":
    render_interactive_box_plot(
        df=df,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        group_col=group_col,
        participants_col=participants_col,
    )
