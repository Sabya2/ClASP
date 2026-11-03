"""Group-comparison statistics and visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from srcs.utils_visualisation.common import (
    _candidate_group_columns,
    _grid_shape,
    get_figure_download,
    p_to_label,
    p_to_star,
    run_group_test,
    set_paper_plot_style,
)


METHOD_MODE_GUIDANCE = {
    "Automatic": (
        "Assumptions are checked separately for each outcome. If every group has "
        "3–5,000 non-constant observations and Shapiro–Wilk p ≥ 0.05, CLASP uses "
        "the parametric route; otherwise it uses the non-parametric route."
    ),
    "Parametric": (
        "Forces tests of group means. Two groups use Student's or Welch's independent "
        "t-test; three or more groups use one-way or Welch's ANOVA. Levene's test "
        "selects the equal-variance or Welch version."
    ),
    "Non-parametric": (
        "Forces rank-based tests. Two groups use Mann–Whitney U; three or more groups "
        "use Kruskal–Wallis followed by Dunn's pairwise comparisons."
    ),
}


def _format_assumption_pvalue(value):
    """Format an assumption-check p-value for explanatory text."""
    if value is None or not np.isfinite(value):
        return "not available"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def build_group_method_explanation(result):
    """Describe how the statistical method was requested and selected."""
    policy = result.get("selection_policy", "auto")
    selected = result.get("selected_method", "unknown")
    policy_labels = {
        "auto": "Automatic",
        "parametric": "Parametric (explicit override)",
        "nonparametric": "Non-parametric (explicit override)",
    }
    lines = [
        f"**Method selection:** {policy_labels.get(policy, policy)}; the resulting "
        f"route for this outcome was **{selected}**."
    ]

    normality = result.get("normality_p") or {}
    if policy == "auto":
        normality_text = ", ".join(
            f"{group}: {_format_assumption_pvalue(p_value)}"
            for group, p_value in normality.items()
        ) or "not available"
        lines.append(
            "Automatic selection used group-wise Shapiro–Wilk checks at α = 0.05 "
            f"(p-values: {normality_text}). All eligible groups must have p ≥ 0.05 "
            "for the parametric route; unavailable checks route to non-parametric tests."
        )
    elif policy == "parametric":
        lines.append(
            "The parametric route was requested explicitly, so normality checks did not "
            "change the test family. Group distributions and outliers should still be reviewed."
        )
    else:
        lines.append(
            "The non-parametric route was requested explicitly. These tests compare ranks "
            "and distributional location; they should not automatically be described as tests "
            "of means or medians."
        )

    if selected == "parametric":
        levene_p = result.get("levene_p", np.nan)
        if np.isfinite(levene_p) and levene_p >= 0.05:
            variance_choice = "equal variances were retained"
        else:
            variance_choice = "Welch unequal-variance inference was selected"
        lines.append(
            f"Levene's variance test p = {_format_assumption_pvalue(levene_p)}; "
            f"{variance_choice} for the overall test. Pairwise t-tests perform their own "
            "two-group variance check."
        )
    else:
        lines.append(
            "Levene's variance test is not used to select rank-based tests. With three or "
            "more groups, Dunn's comparisons form one pairwise family for this outcome."
        )

    lines.append(
        "All tests on this page are for independent groups. Missing or infinite values are "
        "removed for the current outcome, and any selected transformation is applied before "
        "testing. Automatic selection is a convenience rather than proof that assumptions hold."
    )
    return "\n\n".join(lines)

def _prepare_group_data(df, group_col, y_col, selected_groups=None):
    sub = df[[group_col, y_col]].copy()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    if selected_groups is not None:
        sub = sub[sub[group_col].isin(selected_groups)]

    return sub


def _apply_scaling(df_plot, y_col, scaling="none"):
    df_plot = df_plot.copy()
    y_plot_col = y_col
    y_label_plot = y_col
    scaling_note = "raw"

    s = pd.to_numeric(df_plot[y_col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    if scaling == "none":
        df_plot[y_plot_col] = s
        scaling_note = "raw"

    elif scaling == "log10":
        valid = s > 0
        df_plot = df_plot.loc[valid].copy()
        s = pd.to_numeric(df_plot[y_col], errors="coerce")
        y_plot_col = f"{y_col}_log10"
        df_plot[y_plot_col] = np.log10(s)
        y_label_plot = f"log10({y_col})"
        scaling_note = "log10-transformed"

    elif scaling == "min-max":
        s_min = s.min()
        s_max = s.max()
        y_plot_col = f"{y_col}_minmax"
        if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
            df_plot[y_plot_col] = np.nan
        else:
            df_plot[y_plot_col] = (s - s_min) / (s_max - s_min)
        y_label_plot = f"{y_col} (min-max scaled)"
        scaling_note = "min-max scaled"

    elif scaling == "z-score":
        s_mean = s.mean()
        s_std = s.std()
        y_plot_col = f"{y_col}_zscore"
        if pd.isna(s_std) or s_std == 0:
            df_plot[y_plot_col] = np.nan
        else:
            df_plot[y_plot_col] = (s - s_mean) / s_std
        y_label_plot = f"{y_col} (z-score)"
        scaling_note = "z-score scaled"

    else:
        raise ValueError(f"Unsupported scaling option: {scaling}")

    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=[y_plot_col])

    return df_plot, y_plot_col, y_label_plot, scaling_note




def _build_overall_text(result, scaling_note="raw"):
    if result is None:
        return "No valid test result"

    overall_p = result.get("overall_p", np.nan)
    overall_test = result.get("overall_test", "Overall test")

    if pd.isna(overall_p):
        return f"{overall_test} [{scaling_note}] p = n/a"

    return f"{overall_test} [{scaling_note}] p {p_to_label(overall_p)} ({p_to_star(overall_p)})"

def _add_pairwise_annotations(ax, result, groups_order, y_min, y_max, max_brackets=10):
    pairwise = result.get("pairwise", [])
    if not pairwise:
        return

    sig_pairs = []
    for res in pairwise:
        p_val = res.get("p_corrected", res.get("p", np.nan))
        if pd.isna(p_val) or p_val >= 0.05:
            continue
        if res["group1"] not in groups_order or res["group2"] not in groups_order:
            continue
        sig_pairs.append(res)

    if not sig_pairs:
        return

    sig_pairs = sig_pairs[:max_brackets]

    y_range = (y_max - y_min) if y_max != y_min else 1.0

    # much bigger spacing between annotation rows
    base_offset = 0.28 * y_range
    step = 0.10 * y_range
    h = 0.025 * y_range

    for idx, res in enumerate(sig_pairs):
        g1 = res["group1"]
        g2 = res["group2"]

        i1 = groups_order.index(g1)
        i2 = groups_order.index(g2)

        if i1 > i2:
            i1, i2 = i2, i1
            g1, g2 = g2, g1

        p_val = res.get("p_corrected", res.get("p", np.nan))
        stars = p_to_star(p_val) if not pd.isna(p_val) else ""

        y_pos = y_max + base_offset + idx * step

        ax.plot(
            [i1, i1, i2, i2],
            [y_pos, y_pos + h, y_pos + h, y_pos],
            lw=1.2,
            c="black",
        )

        # pair label
        pair_label = f"{stars}"
        ax.text(
            (i1 + i2) / 2,
            y_pos + h + 0.012 * y_range,
            pair_label,
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="medium",
            color="black",
        )

    top_extra = base_offset + len(sig_pairs) * step + 0.10 * y_range
    ax.set_ylim(y_min - 0.08 * y_range, y_max + top_extra)



def draw_group_comparison_on_ax(
    ax,
    df,
    group_col,
    y_col,
    plot_kind="box",
    selected_groups=None,
    y_label=None,
    showfliers=False,
    strip=True,
    two_groups_only=False,
    force_parametric=False,
    method="auto",
    show_annotations=True,
    scaling="none",
):
    """Draw one group comparison and its statistical annotations."""
    set_paper_plot_style()

    df_plot = _prepare_group_data(df, group_col, y_col, selected_groups=selected_groups)
    if df_plot.empty:
        ax.axis("off")
        return None

    df_plot, y_plot_col, y_label_plot, scaling_note = _apply_scaling(
        df_plot=df_plot,
        y_col=y_col,
        scaling=scaling,
    )
    if df_plot.empty:
        ax.axis("off")
        return None

    result = run_group_test(
        df=df_plot,
        group_col=group_col,
        y_col=y_plot_col,
        two_groups_only=two_groups_only,
        force_parametric=force_parametric,
        method=method,
    )
    if result is None:
        ax.axis("off")
        return None

    result["scaling_note"] = scaling_note
    result["original_y_col"] = y_col
    result["plot_y_col"] = y_plot_col

    if selected_groups is not None and len(selected_groups) > 0:
        groups_order = [g for g in selected_groups if g in df_plot[group_col].unique().tolist()]
    else:
        groups_order = [g for g in result["groups"] if g in df_plot[group_col].unique().tolist()]

    if not groups_order:
        ax.axis("off")
        return None

    if plot_kind == "box":
        sns.boxplot(
            data=df_plot,
            x=group_col,
            y=y_plot_col,
            order=groups_order,
            hue=group_col,
            hue_order=groups_order,
            showfliers=showfliers,
            width=0.5,
            ax=ax,
        )

    else:
        sns.violinplot(
                data=df_plot,
                x=group_col,
                y=y_plot_col,
                order=groups_order,
                hue=group_col,
                hue_order=groups_order,
                inner="box",
                dodge=False,
                legend=False,
                ax=ax,
            )


    counts = df_plot[group_col].value_counts()
    x_labels = [f"{g}\n(n={counts.get(g, 0)})" for g in groups_order]
    ax.set_xticks(range(len(groups_order)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel(y_label if y_label else y_label_plot)

    overall_text = _build_overall_text(result, scaling_note=scaling_note)
    ax.set_title(f"{y_col}\n{overall_text}", fontsize=9)

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    # to add the significance line
    if show_annotations and len(groups_order) >= 2:
        y_min = df_plot[y_plot_col].min()
        y_max = df_plot[y_plot_col].max()
        _add_pairwise_annotations(ax, result, groups_order, y_min, y_max)

    return result



def format_group_summary(result, group_col, y_col):
    """Convert a group-test result into a display-ready summary."""
    if result is None:
        return f"**{group_col} – {y_col}**: Insufficient data for group comparison."

    total_n = sum(len(v) for v in result["groups_data"].values())
    correction_method = result.get("correction_method", "none")
    selected_groups = result.get("groups", [])
    pairwise = result.get("pairwise", [])
    overall_test = result.get("overall_test", "n/a")
    overall_p = result.get("overall_p", np.nan)
    use_dunn = result.get("use_dunn_test", False)
    scaling_note = result.get("scaling_note", "raw")
    sample_sizes = ", ".join(f"{group}: n={len(values)}" for group, values in result["groups_data"].items())

    txt = (
        f"**{group_col} – {y_col}** (independent groups; total N = {total_n}, "
        f"scale = {scaling_note})\n\nAnalysed sample sizes: {sample_sizes}.\n"
    )
    if result.get("data_warning"):
        txt += f"\n{result['data_warning']}\n"

    txt += "\n### Statistical method used\n\n"
    txt += build_group_method_explanation(result) + "\n\n"

    if pd.isna(overall_p):
        txt += f"Overall test {overall_test} failed (no valid p-value).\n"
    elif overall_p < 0.05:
        txt += f"Overall test {overall_test} was significant on the {scaling_note} data (p {p_to_label(overall_p)}{p_to_star(overall_p)}).\n"
    else:
        txt += f"Overall test {overall_test} was not significant on the {scaling_note} data (p = {overall_p:.3f}).\n"

    if use_dunn:
        if correction_method and correction_method != "none":
            txt += (
                f"\nBecause the omnibus test was **Kruskal–Wallis**, post-hoc pairwise comparisons "
                f"were performed using **Dunn's test** with **{correction_method} correction** "
                f"to control for multiple testing.\n"
            )
        else:
            txt += (
                f"\nBecause the omnibus test was **Kruskal–Wallis**, post-hoc pairwise comparisons "
                f"were performed using **Dunn's test**.\n"
            )
    else:
        if correction_method and correction_method != "none" and len(selected_groups) > 2:
            txt += (
                f"\nPairwise p-values were adjusted using the **{correction_method} correction** "
                f"to control false positives from multiple comparisons.\n"
            )
        elif len(selected_groups) <= 2:
            txt += "\nNo pairwise correction was needed because only 2 groups were compared.\n"

    if len(pairwise) == 0:
        txt += "\nNo valid pairwise comparisons were available.\n"
        return txt

    txt += f"\n**All pairwise comparisons on {scaling_note} data:**\n"

    for r in pairwise:
        p_raw = r.get("p", np.nan)
        p_val = r.get("p_corrected", p_raw)
        d_val = r.get("cohens_d", np.nan)
        d_txt = "n/a" if pd.isna(d_val) else f"{d_val:.2f}"
        sig_marker = " **✓**" if (not pd.isna(p_val) and p_val < 0.05) else ""

        if use_dunn:
            if pd.isna(p_val):
                txt += (
                    f"- {r['group1']} vs {r['group2']}: Dunn's test, adjusted p = n/a, "
                    f"Cohen's d = {d_txt}\n"
                )
            else:
                txt += (
                    f"- {r['group1']} vs {r['group2']}: Dunn's test, "
                    f"adjusted p = {p_val:.3f} {p_to_star(p_val)}, "
                    f"Cohen's d = {d_txt}{sig_marker}\n"
                )
        else:
            if pd.isna(p_val):
                txt += (
                    f"- {r['group1']} vs {r['group2']}: {r['test']} p = n/a, "
                    f"Cohen's d = {d_txt}\n"
                )
            else:
                label_prefix = "adjusted p" if "p_corrected" in r else "p"
                txt += (
                    f"- {r['group1']} vs {r['group2']}: {r['test']}, "
                    f"{label_prefix} = {p_val:.3f} {p_to_star(p_val)}, "
                    f"Cohen's d = {d_txt}{sig_marker}\n"
                )

    return txt



def render_group_comparison_page(df, continuous_cols, group_col):
    """Render group-comparison controls and plots."""
    set_paper_plot_style()

    if df is None:
        st.warning("Upload data first.")
        return

    if not continuous_cols:
        st.warning("Select at least one continuous column in the sidebar.")
        return

    if not group_col:
        st.warning("Select a valid group column.")
        return

    with st.expander("How this page works", expanded=False):
        st.markdown(
            """
            This page compares continuous variables across selected groups using boxplots and statistical group tests.

            **What you can do here**
            - Select one or more continuous variables to visualize.
            - Choose how many plots appear per row using **Grid columns**.
            - Apply a transformation to the y-axis values:
              - **none**: uses the raw values
              - **log10**: applies log base 10 to positive values only
              - **min-max**: rescales values to the 0–1 range
              - **z-score**: standardizes values relative to the sample mean and standard deviation
            - Optionally switch to another grouping variable for this page only.
            - Restrict the analysis to selected group levels.

            **How the statistics work**
            - For each selected variable, the app prepares valid numeric data and removes missing or infinite values.
            - Tests compare **independent groups**; this page does not perform paired or repeated-measures comparisons.
            - **Automatic:** selection is repeated for each outcome. Every group must have 3–5,000 non-constant observations and a Shapiro–Wilk p-value ≥ 0.05 to use the parametric route. If a check is unavailable or p < 0.05, the non-parametric route is used.
            - **Parametric:** compares group means. For two groups, CLASP uses Student's independent t-test when Levene's test supports equal variance and Welch's t-test otherwise. For three or more groups it uses ordinary one-way ANOVA or Welch's ANOVA by the same variance rule.
            - **Non-parametric:** compares ranks/distributional location. Two groups use Mann–Whitney U. Three or more groups use Kruskal–Wallis, followed by Dunn's pairwise comparisons.
            - For three or more groups, pairwise p-values are Bonferroni-adjusted within the comparisons for the current outcome. This does not correct across all selected outcome variables.
            - Parametric overrides do not guarantee that assumptions are satisfied. Non-parametric significance should not automatically be interpreted as a difference in medians unless distribution shapes are comparable.
            - If pairwise comparisons are available, significant group differences are drawn above the plot as brackets with significance stars.
            - The plot title shows the omnibus test and p-value for the current variable and transformation.

            **How to read the plots**
            - Each box summarizes the distribution within one group.
            - The x-axis label includes the group name and sample size, shown as `n=...`.
            - Pairwise stars above the plot indicate statistically significant differences for selected comparisons.
            - The summary section below the plots gives a text interpretation of the same statistical results.

            **Notes**
            - `log10` scaling keeps only positive values.
            - If a variable has no usable data after filtering or transformation, that plot is skipped.
            - Summary text reports the transformation used so interpretation stays aligned with the displayed plot.
                        """
            )

    plot_group_col = group_col

    top_col1, top_col2 = st.columns([3, 2], gap="small")

    with top_col1:
        cols_to_plot = st.multiselect(
            "Select variables",
            options=continuous_cols,
            default=continuous_cols[: min(6, len(continuous_cols))],
            key="group_comp_cols_multiselect",
        )

    with top_col2:

        side_col1, side_col2 = st.columns([2, 3], gap="small")
        with side_col1:
            plot_kind = st.radio(
                    "Plot type",
                    ["box", "violin"],
                    horizontal=False,
                    key="dist_plot_kind",
                )
        with side_col2:
            scaling = st.selectbox(
                        "Y scaling / transformation",
                        options=["none", "log10", "min-max", "z-score"],
                        index=0,
                        key="group_comp_scaling",
                    )

        grid_cols = st.slider(
            "Grid columns",
            min_value=1,
            max_value=4,
            value=2,
            key="group_comp_grid_cols",
        )
        method_label = st.selectbox(
            "Statistical method",
            ["Automatic", "Parametric", "Non-parametric"],
            help="Summary display format does not control this choice.",
            key="group_comp_method",
        )
        method = {
            "Automatic": "auto", "Parametric": "parametric",
            "Non-parametric": "nonparametric",
        }[method_label]
        st.caption(METHOD_MODE_GUIDANCE[method_label])



    available_group_cols = _candidate_group_columns(df, group_col)

    filter_col1, filter_col2 = st.columns([1, 2], gap="small")

    with filter_col1:
        use_alt_group = st.checkbox(
            "Use a different group variable on this page",
            value=False,
            key="group_comp_use_alt_group",
        )

        if use_alt_group:
            plot_group_col = st.selectbox(
                "Select another group variable",
                options=available_group_cols,
                index=available_group_cols.index(group_col) if group_col in available_group_cols else 0,
                key="group_comp_group_selectbox",
            )

    with filter_col2:
        st.markdown(f"Group values for **{plot_group_col}**")
        groups_to_show = df[plot_group_col].dropna().unique().tolist()
        default_groups = groups_to_show[: min(6, len(groups_to_show))]

        selected_groups = st.multiselect(
            "Select the groups to compare",
            options=groups_to_show,
            default=default_groups,
            key=f"group_comp_group_values_multiselect_{plot_group_col}",
        )

    if not cols_to_plot:
        st.info("Select at least one variable.")
        return

    if not selected_groups:
        st.info("Select at least one group.")
        return

    plot_df = df[df[plot_group_col].isin(selected_groups)].copy()
    if plot_df.empty:
        st.info("No data available for the selected groups.")
        return

    nrows, _ = _grid_shape(len(cols_to_plot), ncols=grid_cols)
    fig, axes = plt.subplots(nrows, grid_cols, figsize=(6.6 * grid_cols, 5.2 * nrows))
    axes = np.array(axes).reshape(-1)

    summaries = {}

    st.subheader("Group comparison plots")

    for row_start in range(0, len(cols_to_plot), grid_cols):
        row_vars = cols_to_plot[row_start: row_start + grid_cols]
        cols_layout = st.columns(grid_cols)

        for col_container, col in zip(cols_layout, row_vars):
            with col_container:
                fig, ax = plt.subplots(figsize=(6.0, 4.8))

                result = draw_group_comparison_on_ax(
                    ax=ax,
                    df=plot_df,
                    group_col=plot_group_col,
                    y_col=col,
                    plot_kind = plot_kind,
                    selected_groups=selected_groups,
                    y_label=col,
                    showfliers=True,
                    show_annotations=True,
                    scaling=scaling,
                    method=method,
                )

                summaries[col] = format_group_summary(result, plot_group_col, col)

                plt.tight_layout()
                st.pyplot(fig)

                buf = get_figure_download(fig)
                st.download_button(
                    label=f"Download {col}",
                    data=buf,
                    file_name=f"group_comparison_{col}.png",
                    mime="image/png",
                    key=f"group_comp_download_{col}",
                )

    return summaries, cols_to_plot, grid_cols



def render_summary_grid(summaries, cols_to_plot, ncols=2):
    """Render selected group summaries in a column grid."""
    st.markdown("### Statistical summaries")

    nrows, _ = _grid_shape(len(cols_to_plot), ncols=ncols)

    for row_idx in range(nrows):
        row_cols = st.columns(ncols, gap="small")
        for col_idx in range(ncols):
            item_idx = row_idx * ncols + col_idx

            with row_cols[col_idx]:
                if item_idx < len(cols_to_plot):
                    col_name = cols_to_plot[item_idx]
                    with st.container(border=True):
                        st.markdown(f"#### {col_name}")
                        st.markdown(summaries[col_name])
