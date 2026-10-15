"""Render molecular or vascular temporal analysis."""

import streamlit as st

from srcs.utils_features.common import get_state
from srcs.utils_visualisation.temporal_molecular import render_molecular_temporal_dynamics
from srcs.utils_visualisation.temporal_vascular import render_vascular_temporal_dynamics


state = get_state()
df = state["df"]
group_col = state["group_col"]
continuous_cols = state["continuous_cols"]
participants_col = state["participants_col"]


if "temporal_dynamics_type" not in st.session_state:
    st.session_state["temporal_dynamics_type"] = None

temporal_dynamics_type = st.radio(
    "Temporal Dynamics",
    options=["Molecular dynamics", "Vascular dynamics"],
    index=None,
    horizontal=True,
    key="temporal_dynamics_type"
)


if df is not None and temporal_dynamics_type == "Molecular dynamics":
    if len(continuous_cols) >= 2:
        with st.expander("How the Temporal Dynamics for molecule page works", expanded=False):
            st.markdown("Select one or more variable families (e.g., `Glucose_T1`, `Klotho_T2`, …) to plot median trajectories over time.")
            st.markdown("All columns in a family must share the same base name and use sequential time suffixes like `_T1`, `_T2`, `_T3`, … (or `_Tp1`, `_tp1`, etc.).")
            st.markdown("Choose error bars as SD, 95% CI, or SE to show variability around the median at each timepoint.")
            st.markdown("Optionally baseline-normalize so each participant's first timepoint equals 100 and later values are expressed as a percentage of baseline.")
        render_molecular_temporal_dynamics(
            df=df,
            group_col=group_col,
            participants_col=participants_col,
        )
    else:
        st.info("Upload data and select at least two continuous variables.")

elif df is not None and temporal_dynamics_type == "Vascular dynamics":
    with st.expander("How the Vascular Dynamics page works", expanded=False):
        st.markdown("Select one or more vascular age columns (e.g., `PWV`, `VascularAge`) and one or more biomarker/fitness columns to compare side-by-side in grouped boxplots.")
        st.markdown("Choose Combined mode to plot each age marker with all selected biomarkers in one figure, or Separate mode to create one plot per age–biomarker pair.")
        st.markdown("Adjust horizontal offset, box width, and axis spacing to improve readability when multiple y-axes are overlaid.")

    render_vascular_temporal_dynamics(
        df=df,
        group_col=group_col,
    )

elif df is not None and temporal_dynamics_type is None:
    st.info("Select the temporal pattern to observe")

else:
    st.info("Upload data")
