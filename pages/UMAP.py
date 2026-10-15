"""Render UMAP analysis from shared application state."""

from srcs.utils_features.common import get_state
from srcs.utils_visualisation.umap_plots import render_umap_page

state = get_state()
df = state["df"]
group_col = state["group_col"]
continuous_cols = state["continuous_cols"]
participants_col = state["participants_col"]

render_umap_page(
    df=df,
    continuous_cols=continuous_cols,
    group_col=group_col,
    participants_col=participants_col,
)
