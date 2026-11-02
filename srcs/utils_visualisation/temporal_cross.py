"""Cross-family temporal dynamics visualization."""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from srcs.utils_features.common import get_repeated_measure_families


def _family_labels(temporal_families):
    """Return unique, readable labels for detected temporal families."""
    name_counts = Counter(
        family_data["family"] for family_data in temporal_families.values()
    )
    return {
        key: (
            family_data["family"]
            if name_counts[family_data["family"]] == 1
            else f'{family_data["family"]} — {family_data["template"]}'
        )
        for key, family_data in temporal_families.items()
    }


def _existing_group_candidates(df, temporal_cols, participants_col):
    """Return low-cardinality non-temporal columns suitable for grouping."""
    temporal_col_set = set(temporal_cols)
    return [
        column
        for column in df.columns
        if column != participants_col
        and column not in temporal_col_set
        and 1 < df[column].nunique(dropna=True) <= 20
    ]


def render_vascular_temporal_dynamics(
    df,
    group_col,
    participants_col=None,
):
    """Compare trajectories across any number of repeated-measure families."""
    st.subheader("Cross-Family Dynamics")

    if df is None:
        st.warning("Upload data first.")
        return

    if participants_col and participants_col not in df.columns:
        participants_col = None

    temporal_cols, temporal_families, temporal_summary = (
        get_repeated_measure_families(df, minimum_timepoints=2)
    )
    if not temporal_families:
        st.warning("No repeated-measure column families were detected.")
        return

    family_labels = _family_labels(temporal_families)
    available_families = sorted(
        temporal_families,
        key=lambda key: (family_labels[key].lower(), key[1]),
    )

    with st.expander("Detected repeated-measure families", expanded=False):
        st.dataframe(temporal_summary, width="stretch", hide_index=True)

    selected_families = st.multiselect(
        "Select families",
        options=available_families,
        default=available_families[: min(3, len(available_families))],
        format_func=lambda key: family_labels[key],
        key="temporal_cross_selected_families",
        help="There is no limit on the number of families you can select.",
    )
    if not selected_families:
        st.info("Select at least one repeated-measure family.")
        return

    grouping_options = ["Overall median", "Use another group"]
    analysis_group_col = None
    with st.container(border=True):
        layout_col, baseline_col, grouping_col = st.columns([1, 1, 3], gap="small")
        with layout_col:
            plot_mode = st.radio(
                "Plot layout",
                options=["Separate", "Combined"],
                index=0,
                horizontal=True,
                key="temporal_cross_plot_mode_v3",
                help=(
                    "Separate creates multiple graphs. Combined places every "
                    "selected family in one graph."
                ),
            )
            families_per_plot = None
            if plot_mode == "Separate":
                families_per_plot = st.radio(
                    "Families per graph",
                    options=[2, 3],
                    index=0,
                    horizontal=True,
                    key="temporal_cross_separate_families_per_plot",
                    help="Create successive graphs containing two or three families.",
                )
            else:
                st.caption("All selected families will be plotted together.")
        
        with baseline_col:
            normalise = st.radio(
                "Baseline normalization",
                options=["Yes", "No"],
                index=0 if participants_col else 1,
                horizontal=False,
                key="temporal_cross_baseline_normalise_v2",
                help=(
                    "Yes sets each participant's first available value within "
                    "each family to 100%."
                ),
            )

            if normalise == "Yes" and not participants_col:
                st.warning("Select a participant ID to use baseline normalization.")
                return

        with grouping_col:
            grouping_mode = st.radio(
                "Grouping",
                options=grouping_options,
                index=0,
                horizontal=True,
                key="temporal_cross_grouping_mode_v2",
            )

            if grouping_mode == "Use another group":
                group_candidates = _existing_group_candidates(
                    df,
                    temporal_cols,
                    participants_col,
                )
                if not group_candidates:
                    st.warning(
                        "No suitable existing group columns were found. Choose "
                        "the overall median option."
                    )
                    return

                default_group_index = (
                    group_candidates.index(group_col)
                    if group_col in group_candidates
                    else 0
                )
                analysis_group_col = st.selectbox(
                    "Group column",
                    options=group_candidates,
                    index=default_group_index,
                    key="temporal_cross_existing_group_col",
                    help=(
                        "The selected column defines the line styles in the plot."
                    ),
                    width="stretch",
                )

    aggregate_all_participants = analysis_group_col is None
    long_frames = []

    for family_key in selected_families:
        family_data = temporal_families[family_key]
        selected_columns = [
            column for column in family_data["columns"]
            if column in df.columns
        ]
        time_lookup = {
            item["column"]: item["time"] for item in family_data["items"]
        }
        id_vars = list(dict.fromkeys(
            ([analysis_group_col] if analysis_group_col else [])
            + ([participants_col] if participants_col else [])
        ))
        selected_columns = [
            column for column in selected_columns if column not in id_vars
        ]
        if not selected_columns:
            continue

        family_long = df.melt(
            id_vars=id_vars,
            value_vars=selected_columns,
            var_name="Timepoint",
            value_name="Value",
        )
        family_long["Time"] = family_long["Timepoint"].map(time_lookup)
        family_long["Family"] = family_labels[family_key]
        family_long["Value"] = pd.to_numeric(
            family_long["Value"], errors="coerce"
        )
        family_long = family_long.dropna(subset=["Time", "Value"]).copy()
        family_long["Original_Value"] = family_long["Value"]

        if normalise == "Yes":
            family_long = family_long.sort_values([participants_col, "Time"])
            family_long["Baseline"] = (
                family_long
                .groupby(participants_col)["Value"]
                .transform("first")
                .replace(0, np.nan)
            )
            family_long["Value"] = (
                family_long["Value"] / family_long["Baseline"]
            ) * 100
            family_long = family_long.dropna(subset=["Value"]).copy()

        if not family_long.empty:
            long_frames.append(family_long)

    if not long_frames:
        st.warning("The selected families do not contain numeric temporal values.")
        return

    long_df = pd.concat(long_frames, ignore_index=True)
    plot_group_col = "__plot_group__"
    if aggregate_all_participants:
        long_df[plot_group_col] = "All participants"
        st.caption(
            "Each family line shows the median across participants at every timepoint."
        )
    else:
        long_df[plot_group_col] = long_df[analysis_group_col]

    agg_df = (
        long_df
        .groupby(["Family", plot_group_col, "Time"], dropna=False)
        .agg(
            median=("Value", "median"),
            std=("Value", "std"),
            n=("Value", "count"),
        )
        .reset_index()
    )
    agg_df["std"] = agg_df["std"].fillna(0)
    agg_df["sem"] = agg_df["std"] / np.sqrt(agg_df["n"].clip(lower=1))
    agg_df["ci"] = 1.96 * agg_df["sem"]

    available_plot_labels = set(agg_df["Family"])
    selected_labels = [
        family_labels[key]
        for key in selected_families
        if family_labels[key] in available_plot_labels
    ]
    group_values = sorted(
        agg_df[plot_group_col].dropna().unique(),
        key=str,
    )
    family_colors = {
        family: plt.get_cmap("tab20")(index % 20)
        for index, family in enumerate(selected_labels)
    }
    marker_options = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    family_markers = {
        family: marker_options[index % len(marker_options)]
        for index, family in enumerate(selected_labels)
    }
    line_styles = ["-", "--", ":", "-."]
    group_styles = {
        group: line_styles[index % len(line_styles)]
        for index, group in enumerate(group_values)
    }

    batch_size = (
        len(selected_labels)
        if plot_mode == "Combined"
        else families_per_plot
    )
    plot_batches = [
        selected_labels[index:index + batch_size]
        for index in range(0, len(selected_labels), batch_size)
    ]

    for plot_number, plot_families in enumerate(plot_batches, start=1):
        plot_df = agg_df[agg_df["Family"].isin(plot_families)]
        fig, ax = plt.subplots(figsize=(9.2, 5.4))

        for family_label in plot_families:
            family_df = plot_df[plot_df["Family"] == family_label]

            for group_value, group_df in family_df.groupby(plot_group_col):
                group_df = group_df.sort_values("Time")
                x_values = group_df["Time"].to_numpy(dtype=float)
                median_values = group_df["median"].to_numpy(dtype=float)
                ci_values = group_df["ci"].to_numpy(dtype=float)
                color = family_colors[family_label]
                legend_label = (
                    family_label
                    if len(group_values) == 1
                    else f"{family_label} — {group_value}"
                )

                ax.plot(
                    x_values,
                    median_values,
                    color=color,
                    marker=family_markers[family_label],
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    linewidth=2.2,
                    linestyle=group_styles[group_value],
                    label=legend_label,
                    zorder=3,
                )
                ax.fill_between(
                    x_values,
                    median_values - ci_values,
                    median_values + ci_values,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                    zorder=2,
                )

        ax.set_xticks(sorted(plot_df["Time"].dropna().unique()))
        if plot_mode == "Combined":
            plot_title = "Combined cross-family temporal dynamics"
        else:
            plot_title = "Cross-family temporal dynamics"
            if len(plot_batches) > 1:
                plot_title += f" ({plot_number}/{len(plot_batches)})"
        ax.set_title(
            plot_title,
            fontsize=12,
            fontweight="semibold",
            pad=12,
        )
        ax.set_xlabel("Timepoint")
        ax.set_ylabel(
            "Median normalized value (baseline = 100%)"
            if normalise == "Yes"
            else "Median value"
        )
        if normalise == "Yes":
            ax.axhline(
                100,
                color="#6B7280",
                linewidth=1,
                linestyle="--",
                alpha=0.7,
                zorder=1,
            )
        ax.grid(True, axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.65)
        ax.set_axisbelow(True)
        ax.margins(x=0.06)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            title=(
                "Family"
                if len(group_values) == 1
                else f"Family — {analysis_group_col}"
            ),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
        )

        fig.tight_layout(rect=(0, 0, 0.80, 1))
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with st.expander("Show temporal summary", expanded=False):
        summary = agg_df.rename(columns={
            plot_group_col: (
                "Population" if aggregate_all_participants else analysis_group_col
            )
        })
        st.dataframe(
            summary[[
                "Family",
                "Population" if aggregate_all_participants else analysis_group_col,
                "Time",
                "median",
                "ci",
                "n",
            ]],
            width="stretch",
            hide_index=True,
        )
