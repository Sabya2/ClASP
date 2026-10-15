"""Render descriptive statistics from shared application state."""

from srcs.utils_features.common import get_state
from srcs.utils_features.data_description import render_data_description_page


state = get_state()
df = state["df"]
group_col = state["group_col"]
continuous_cols = state["continuous_cols"]
categorical_cols = state["categorical_cols"]

render_data_description_page(
    df=df,
    group_col=group_col,
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
)
