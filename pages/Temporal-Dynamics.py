"""Render within-family and cross-family temporal analyses."""

import streamlit as st

from srcs.utils_features.common import get_state
from srcs.utils_visualisation.temporal_within import render_molecular_temporal_dynamics
from srcs.utils_visualisation.temporal_cross import render_vascular_temporal_dynamics


state = get_state()
df = state["df"]
group_col = state["group_col"]
participants_col = state["participants_col"]

st.title("Temporal Dynamics")
st.caption(
    "Explore repeated measurements over time within a variable family or "
    "compare trajectories across multiple families."
)

with st.expander("How Temporal Dynamics works", expanded=False):
    st.markdown(
        "The page automatically groups columns whose names have the same "
        "structure and differ at a numbered measurement position. At least "
        "two distinct numbered columns are required to form a family."
    )
    st.markdown(
        "Supported examples include `Glucose_T1`, `Spiro_3min_HR`, "
        "`OCTA hour 4`, and `Spiro_Level1_[W]`. Spaces, underscores, and "
        "hyphens are treated as equivalent separators when matching names."
    )
    st.markdown(
        "Open **Detected repeated-measure families** inside either analysis "
        "view to inspect the detected family, source columns, and timepoints."
    )

    within_help, cross_help = st.tabs(["Within-Family","Cross-Family",])
    with within_help:
        st.markdown(
            "Use this view to examine each selected family in its own graph. "
            "Every point is the median value at a detected timepoint."
        )
        st.markdown(
            "Choose **Overall median** for one cohort trajectory, or select an "
            "existing low-cardinality group column to draw separate group "
            "trajectories."
        )
        st.markdown(
            "Select a 95% participant-bootstrap median CI, IQR, or descriptive SD; optionally normalize each "
            "participant so their first available timepoint equals 100; then "
            "choose how many family graphs appear per row."
        )
        st.markdown(
            "The optional long-form and summary tables expose the transformed "
            "values and statistics used in the graphs. Baseline normalization "
            "requires a participant ID column."
        )

    with cross_help:
        st.markdown(
            "Use this view to compare any number of selected families. Families "
            "use consistent colors and markers, while optional participant "
            "groups use different line styles."
        )
        st.markdown(
            "**Separate** layout creates successive graphs containing two or "
            "three families per graph. **Combined** layout overlays every "
            "selected family in one graph."
        )
        st.markdown(
            "Baseline normalization defaults to Yes when a participant ID is "
            "available. It sets each participant's first value within each "
            "family to 100% before calculating medians and bootstrap confidence intervals."
        )
        st.markdown(
            "Grouping defaults to the overall median. Choose **Use another "
            "group** to reveal a group-column selector. Lines show medians and "
            "shaded regions show 95% percentile bootstrap confidence intervals for medians. "
            "These plots are descriptive and are not repeated-measures hypothesis tests."
        )

temporal_dynamics_type = st.segmented_control(
    "Analysis view",
    options=["Within-Family dynamics", "Cross-Family dynamics"],
    default=None,
    width="stretch",
    selection_mode="single",
    key="temporal_dynamics_type_v2",
    help=(
        "Within-Family shows families in graph batches. Cross-Family compares "
        "multiple families in the same graph or graph sequence."
    ),
)

if df is not None and temporal_dynamics_type == "Within-Family dynamics":
    render_molecular_temporal_dynamics(
        df=df,
        group_col=group_col,
        participants_col=participants_col,
    )

elif df is not None and temporal_dynamics_type == "Cross-Family dynamics":
    render_vascular_temporal_dynamics(
        df=df,
        group_col=group_col,
        participants_col=participants_col,
    )

elif df is not None and temporal_dynamics_type is None:
    st.info("Choose Within-Family or Cross-Family dynamics to begin.")

else:
    st.info("Upload a dataset to begin temporal analysis.")
