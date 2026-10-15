"""Render distribution plots from shared application state."""

from srcs.utils_features.common import get_state

from srcs.utils_visualisation.distribution_plots import render_histogram_page


state = get_state()
df = state["df"]
group_col = state["group_col"]
continuous_cols = state["continuous_cols"]
render_histogram_page(
    df=df,
    continuous_cols=continuous_cols,
    group_col=group_col
)
