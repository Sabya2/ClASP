"""Molecular time-series visualization."""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from srcs.utils_features.common import get_repeated_measure_families
from srcs.utils_visualisation.temporal_stats import summarize_median_trajectory


def render_molecular_temporal_dynamics(df, group_col, participants_col=None):
    """Render longitudinal within-family trajectories."""
    st.subheader("Within-Family Dynamics")

    if df is None:
        st.warning("Upload data first.")
        return

    if not group_col or group_col not in df.columns:
        st.warning("Select a valid grouping column first.")
        return

    if participants_col and participants_col not in df.columns:
        participants_col = None

    temporal_cols, temporal_families, temporal_summary = get_repeated_measure_families(
        df,
        minimum_timepoints=2,
    )
    if not temporal_families:
        st.warning("No repeated-measure column families were detected.")
        return

    available_groups = sorted(
        temporal_families,
        key=lambda key: (temporal_families[key]["family"].lower(), key[1]),
    )
    family_name_counts = Counter(
        family_data["family"] for family_data in temporal_families.values()
    )
    family_labels = {
        key: (
            family_data["family"]
            if family_name_counts[family_data["family"]] == 1
            else f'{family_data["family"]} — {family_data["template"]}'
        )
        for key, family_data in temporal_families.items()
    }

    with st.expander("Detected repeated-measure families", expanded=False):
        st.dataframe(temporal_summary, width="stretch", hide_index=True)

    grouping_options = [
        "Overall median across participants",
        "Use an existing group column",
    ]
    default_grouping_index = int(
        not participants_col or group_col != participants_col
    )
    grouping_mode = st.radio(
        "How should the trajectories be grouped?",
        options=grouping_options,
        index=default_grouping_index,
        horizontal=True,
        key="temporal_within_grouping_mode",
    )

    analysis_group_col = None
    if grouping_mode == "Use an existing group column":
        temporal_col_set = set(temporal_cols)
        existing_group_candidates = [
            column
            for column in df.columns
            if column != participants_col
            and column not in temporal_col_set
            and 1 < df[column].nunique(dropna=True) <= 20
        ]
        if not existing_group_candidates:
            st.warning(
                "No suitable existing group columns were found. Choose the overall median option."
            )
            return

        default_group_index = (
            existing_group_candidates.index(group_col)
            if group_col in existing_group_candidates
            else 0
        )
        analysis_group_col = st.selectbox(
            "Existing group column",
            options=existing_group_candidates,
            index=default_group_index,
            key="temporal_within_existing_group_col",
            help="The selected column defines the separate trajectories in each plot.",
        )

    aggregate_all_participants = analysis_group_col is None

    col1, col2, col3, col4 = st.columns([2.4, 0.8, 0.8, 1], gap="small")

    with col1:
        selected_groups = st.multiselect(
            "Select one or more variable families",
            options=available_groups,
            default=available_groups[: min(4, len(available_groups))],
            format_func=lambda key: family_labels[key],
        )
        if not selected_groups:
            st.info("Please select at least one variable family.")
            return

    with col2:
        error_type = st.radio(
            "Median interval / spread",
            options=["95% bootstrap CI", "IQR", "SD (descriptive spread)"],
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

    for family_key in selected_groups:
        family_data = temporal_families[family_key]
        var_name = family_labels[family_key]
        selected_cols = family_data["columns"]
        time_lookup = {
            item["column"]: item["time"] for item in family_data["items"]
        }

        # pandas.melt fails when the same identifier is supplied twice. This
        # can happen when Pat_ID (or another ID) is selected as both fields.
        id_vars = list(dict.fromkeys(
            ([analysis_group_col] if analysis_group_col else [])
            + ([participants_col] if participants_col else [])
        ))
        selected_cols = [
            column for column in selected_cols
            if column in df.columns and column not in id_vars
        ]
        if not selected_cols:
            continue

        tmp = df.melt(
            id_vars=id_vars,
            value_vars=selected_cols,
            var_name="Timepoint",
            value_name="Value"
        )

        tmp["Time"] = tmp["Timepoint"].map(time_lookup)
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

    if not long_frames:
        st.warning("No valid repeated-measure value columns remain after selection.")
        return

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df.dropna(subset=["Value"]).copy()

    # Overall mode uses a single plotting group, so each point represents the
    # median across all available participants at that timepoint.
    plot_group_col = "__plot_group__"
    if aggregate_all_participants:
        long_df[plot_group_col] = "All participants"
        st.caption(
            "The plots show the median across participants at each timepoint."
        )
    else:
        long_df[plot_group_col] = long_df[analysis_group_col]

    agg_df = summarize_median_trajectory(
        long_df,
        ["Variable", plot_group_col, "Time"],
        participant_col=participants_col,
    )
    st.caption(
        "Points are descriptive medians, not longitudinal hypothesis tests. Missing values are omitted "
        "per timepoint. Bootstrap intervals resample participant-level values; n=1 and constant cells "
        "have zero-width intervals. Baseline normalization uses each participant's first observed value."
    )

    for row_start in range(0, len(selected_groups), ncols):
        cols = st.columns(ncols)
        row_groups = selected_groups[row_start: row_start + ncols]

        for col_container, family_key in zip(cols, row_groups):
            with col_container:
                var_name = family_labels[family_key]
                sub_var = agg_df[agg_df["Variable"] == var_name].copy()

                fig, ax = plt.subplots(figsize=(6.2, 4.6))

                groups = list(sub_var[plot_group_col].unique())
                offset = 0 if len(groups) == 1 else min(0.12, 0.4 / len(groups))
                colors = plt.get_cmap("tab10")

                for i, (grp, sub) in enumerate(sub_var.groupby(plot_group_col)):
                    sub = sub.sort_values("Time")

                    shift = (i - (len(groups) - 1) / 2) * offset
                    x_vals = sub["Time"] + shift
                    color = colors(i % 10)

                    ax.plot(
                        x_vals,
                        sub["median"],
                        color=color,
                        marker="o",
                        markersize=6,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        linewidth=2.2,
                        label=str(grp),
                        zorder=3,
                    )

                    if error_type == "95% bootstrap CI":
                        err = np.vstack((sub["median"] - sub["ci_low"], sub["ci_high"] - sub["median"]))
                    elif error_type == "IQR":
                        err = np.vstack((sub["median"] - sub["q1"], sub["q3"] - sub["median"]))
                    else:
                        err = sub["std"].fillna(0)

                    ax.errorbar(
                        x_vals,
                        sub["median"],
                        yerr=err,
                        capsize=4,
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.2,
                        capthick=1.2,
                        alpha=0.8,
                        zorder=2,
                    )

                ax.set_xticks(sorted(sub_var["Time"].dropna().unique()))
                title = var_name
                if aggregate_all_participants:
                    title = f"{var_name}\nMedian across participants"
                ax.set_title(title, fontsize=11, fontweight="semibold", pad=10)
                ax.set_xlabel("Timepoint")

                if normalise == "Yes":
                    ax.set_ylabel("Normalized value (baseline = 100)")
                else:
                    ax.set_ylabel("Median value")

                ax.grid(True, axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.65)
                ax.set_axisbelow(True)
                ax.margins(x=0.08)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if len(groups) > 1:
                    ax.legend(
                        title=analysis_group_col,
                        fontsize=8,
                        title_fontsize=8,
                        frameon=False,
                        ncols=2 if len(groups) > 4 else 1,
                    )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("Long-form data")

    show_long_df = st.checkbox("Show long-form dataframe")
    if show_long_df:
        display_cols = ["Variable"]
        if participants_col:
            display_cols.append(participants_col)
        if analysis_group_col and analysis_group_col != participants_col:
            display_cols.append(analysis_group_col)
        display_cols.append("Time")
        display_cols.extend(["Original_Value"])
        if "Baseline" in long_df.columns:
            display_cols.append("Baseline")
        display_cols.append("Value")

        st.dataframe(
            long_df[display_cols].sort_values(
                list(dict.fromkeys(
                    c for c in [
                        "Variable", analysis_group_col, participants_col, "Time"
                    ]
                    if c is not None
                ))
            ),
            width="stretch",
            height=400,
            hide_index=True
        )

    show_summary = st.checkbox("Show summary table")
    if show_summary:
        summary_df = agg_df.rename(columns={plot_group_col: "Population" if aggregate_all_participants else analysis_group_col})
        if aggregate_all_participants:
            summary_df["Population"] = "All participants"

        st.dataframe(
            summary_df,
            width="stretch",
            height=300,
            hide_index=True
        )
        st.download_button(
            "Download temporal median summary (.csv)",
            summary_df.to_csv(index=False).encode("utf-8"),
            "temporal_median_bootstrap_summary.csv",
            "text/csv",
            key="temporal_within_summary_download",
        )
