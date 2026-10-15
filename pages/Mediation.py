"""Render mediation analysis from shared application state."""

import streamlit as st
from srcs.utils_features.common import get_state
from srcs.utils_visualisation.mediation_plots import render_mediation_page

st.title("Mediation Analysis")
state = get_state()
df = state["df"]
continuous_cols = state.get("continuous_cols", [])
categorical_cols = state.get("categorical_cols", [])
group_col = state.get("group_col")

render_mediation_page(df, continuous_cols, categorical_cols, group_col)
