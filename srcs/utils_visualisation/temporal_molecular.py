"""Molecular time-series visualization."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def extract_base_name(col):
    """Remove a supported timepoint suffix from a column name."""
    return re.sub(r'(_T\d+|_Tp\d+|_tp\d+)$', '', col)


def extract_timepoint(col):
    """Extract a numeric timepoint from a column name."""
    match = re.search(r'(?:_T|_Tp|_tp)(\d+)', col)
    return int(match.group(1)) if match else None


def render_molecular_temporal_dynamics(df, group_col, participants_col=None):
    """Render longitudinal molecular trajectories."""
    st.header("Temporal Dynamics")

    if df is None:
        st.warning("Upload data first.")
        return

    temporal_cols = [
        col for col in df.columns
        if re.search(r'_T\d+|_Tp\d+|_tp\d+', col)
    ]

    if not temporal_cols:
        st.warning("No temporal columns detected.")
        return

    grouped_vars = {}
    for col in temporal_cols:
        base = extract_base_name(col)
        grouped_vars.setdefault(base, []).append(col)

    available_groups = sorted(grouped_vars.keys())

    col1, col2, col3, col4 = st.columns([2.4, 0.8, 0.8, 1], gap="small")

    with col1:
        selected_groups = st.multiselect(
            "Select one or more variable families",
            options=available_groups,
            default=available_groups[: min(4, len(available_groups))]
        )
        if not selected_groups:
            st.info("Please select at least one variable family.")
            return

    with col2:
        error_type = st.radio(
            "Error bars",
            options=["SD", "CI (95%)", "SE"],
            horizontal=False
        )

    with col3:
        normalise = st.radio(
            "Baseline Normalise?",
            options=["Yes", "No"],
            horizontal=False
        )

    with col4:
        ncols = st.slider("Plots per row", min_value=1, max_value=4, value=2)

    long_frames = []

    for var_name in selected_groups:
        selected_cols = sorted(
            grouped_vars[var_name],
            key=extract_timepoint
        )

        id_vars = [group_col]
        if participants_col:
            id_vars.append(participants_col)

        tmp = df.melt(
            id_vars=id_vars,
            value_vars=selected_cols,
            var_name="Timepoint",
            value_name="Value"
        )

        tmp["Time"] = tmp["Timepoint"].apply(extract_timepoint)
        tmp["Variable"] = var_name
        tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce")
        tmp["Original_Value"] = tmp["Value"]
        tmp = tmp.dropna(subset=["Value"]).copy()

        if normalise == "Yes":
            if not participants_col:
                st.warning("Baseline normalization requires a participant ID column.")
                return

            tmp = tmp.sort_values([participants_col, "Time"])
            tmp["Baseline"] = tmp.groupby(participants_col)["Value"].transform("first")
            tmp["Baseline"] = tmp["Baseline"].replace(0, np.nan)
            tmp["Value"] = (tmp["Value"] / tmp["Baseline"]) * 100
            tmp = tmp.dropna(subset=["Value"]).copy()

        long_frames.append(tmp)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df.dropna(subset=["Value"]).copy()

    agg_df = (
        long_df
        .groupby(["Variable", group_col, "Time"], dropna=False)
        .agg(
            median=("Value", "median"),
            std=("Value", "std"),
            n=("Value", "count"),
            q1=("Value", lambda x: x.quantile(0.25)),
            q3=("Value", lambda x: x.quantile(0.75))
        )
        .reset_index()
    )

    agg_df["std"] = agg_df["std"].fillna(0)
    agg_df["sem"] = agg_df["std"] / np.sqrt(agg_df["n"])
    agg_df["ci"] = 1.96 * agg_df["sem"]
    agg_df["iqr_lower"] = agg_df["median"] - agg_df["q1"]
    agg_df["iqr_upper"] = agg_df["q3"] - agg_df["median"]

    for row_start in range(0, len(selected_groups), ncols):
        cols = st.columns(ncols)
        row_groups = selected_groups[row_start: row_start + ncols]

        for col_container, var_name in zip(cols, row_groups):
            with col_container:
                sub_var = agg_df[agg_df["Variable"] == var_name].copy()

                fig, ax = plt.subplots(figsize=(5, 4))

                groups = list(sub_var[group_col].unique())
                offset = 0.1

                for i, (grp, sub) in enumerate(sub_var.groupby(group_col)):
                    sub = sub.sort_values("Time")

                    shift = (i - len(groups) / 2) * offset
                    x_vals = sub["Time"] + shift

                    ax.plot(x_vals, sub["median"], marker="o", label=str(grp))

                    if error_type == "SD":
                        err = sub["std"]
                        ax.errorbar(
                            x_vals, sub["median"], yerr=err,
                            capsize=4, fmt="none", ecolor="black",
                            elinewidth=0.8, capthick=0.8
                        )
                    elif error_type == "CI (95%)":
                        err = sub["ci"]
                        ax.errorbar(
                            x_vals, sub["median"], yerr=err,
                            capsize=4, fmt="none", ecolor="black",
                            elinewidth=0.8, capthick=0.8
                        )
                    elif error_type == "SE":
                        err = sub["sem"]
                        ax.errorbar(
                            x_vals, sub["median"], yerr=err,
                            capsize=4, fmt="none", ecolor="black",
                            elinewidth=0.8, capthick=0.8
                        )

                ax.set_xticks(sorted(sub_var["Time"].dropna().unique()))
                ax.set_title(var_name)
                ax.set_xlabel("Time")

                if normalise == "Yes":
                    ax.set_ylabel("Normalized value (baseline = 100)")
                else:
                    ax.set_ylabel("Median Value")

                ax.grid(True, alpha=0.3)
                ax.legend(title=group_col, fontsize=8)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("Long-form data")

    show_long_df = st.checkbox("Show long-form dataframe")
    if show_long_df:
        display_cols = ["Variable", group_col, "Time"]
        if participants_col:
            display_cols.insert(2, participants_col)
        display_cols.extend(["Original_Value"])
        if "Baseline" in long_df.columns:
            display_cols.append("Baseline")
        display_cols.append("Value")

        st.dataframe(
            long_df[display_cols].sort_values(
                [c for c in ["Variable", group_col, participants_col, "Time"] if c is not None]
            ),
            width="stretch",
            height=400,
            hide_index=True
        )

    show_summary = st.checkbox("Show summary table")
    if show_summary:
        summary_df = (
            long_df
            .groupby(["Variable", group_col, "Time"], dropna=False)
            .agg(
                median_value=("Value", "median"),
                q1=("Value", lambda x: x.quantile(0.25)),
                q3=("Value", lambda x: x.quantile(0.75)),
                n=("Value", "count")
            )
            .reset_index()
            .sort_values(["Variable", group_col, "Time"])
        )

        st.dataframe(
            summary_df,
            width="stretch",
            height=300,
            hide_index=True
        )
