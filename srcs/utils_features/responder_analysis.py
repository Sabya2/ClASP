
"""
Responder Analysis Module - Core calculation and plotting functions

This module contains the SWC/TE calculation logic and matplotlib-based visualization
for individual responder analysis.
"""


import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from statistics import NormalDist
from typing import Optional, Tuple, List
from scipy import stats
import seaborn as sns
import streamlit as st



def icc_31_from_two_timepoints(df, t1_col, t2_col,
):
    """
    Compute ICC(3,1) for test–retest reliability from two timepoints (t1, t2)
    using a two-way mixed-effects, absolute-agreement, single-measurement model.

    Formula (k = 2 measurements per subject):
        ICC(3,1) = (MS_subjects - MS_error) / (MS_subjects + MS_error)

    Returns:
        icc_value: float
        details: dict with variance components and mean squares
    """
    data = df[[t1_col, t2_col]].dropna(subset=[t1_col, t2_col]).copy()

    n = len(data)          # number of subjects
    k = 2                  # number of measurements (t1, t2)

    y1 = data[t1_col].to_numpy()
    y2 = data[t2_col].to_numpy()

    # Subject means and grand mean
    subject_means = (y1 + y2) / 2.0
    grand_mean = subject_means.mean()
    # print(subject_means)


    # Sum of squares for subjects (between-subject)
    SS_subjects = k * np.sum((subject_means - grand_mean) ** 2)

    # Total sum of squares
    SS_total = np.sum((y1 - grand_mean) ** 2) + np.sum((y2 - grand_mean) ** 2)

    # Sum of squares for error (within-subject)
    SS_error = np.sum((y1 - subject_means) ** 2) + np.sum((y2 - subject_means) ** 2)

    # Degrees of freedom
    df_subjects = n - 1
    df_error = (n - 1) * (k - 1)  # for k=2, this is n-1

    # Mean squares
    MS_subjects = SS_subjects / df_subjects
    MS_error = SS_error / df_error if df_error > 0 else 0.0

    # ICC(3,1)
    if MS_subjects + MS_error == 0:
        icc_value = 0.0
    else:
        icc_value = (MS_subjects - MS_error) / (MS_subjects + MS_error)

    details = {
        "n_subjects": n,
        "k_measurements": k,
        "SS_subjects": SS_subjects,
        "SS_error": SS_error,
        "SS_total": SS_total,
        "df_subjects": df_subjects,
        "df_error": df_error,
        "MS_subjects": MS_subjects,
        "MS_error": MS_error,
        "var_subject": MS_subjects,
        "var_error": MS_error,
    }

    return icc_value, details



def check_t1_t2_stability(df, t1_col, t2_col,):
    """
    Check if t1 and t2 are stable (no systematic change) and display results in Streamlit.
    This validates using them for TE estimation.

    Returns:
        dict: Stability metrics and verdict
    """


    # Keep only paired observations
    mask = df[t1_col].notna() & df[t2_col].notna()
    t1_paired = df.loc[mask, t1_col]
    t2_paired = df.loc[mask, t2_col]
    t1_name = f'Test: {t1_col}'
    t2_name = f'ReTest: {t2_col}'

    if len(t1_paired) == 0:
        st.error("No paired observations found for reliability check.")
        return None

    # Create expandable section
    with st.expander("🔍 Reliability Check: t1 vs t2 Stability", expanded=False):
        st.markdown(f"**Checking stability:** {t1_name} vs {t2_name} with {len(t1_paired)} paired observations")

        # 1. Paired t-test
        t_stat, t_p = stats.ttest_rel(t1_paired, t2_paired)
        mean_diff = (t2_paired - t1_paired).mean()
        sd_diff = (t2_paired - t1_paired).std()

        col1, col2, col3= st.columns(3)
        with col1:
            st.markdown("**Paired t-test**")
            st.metric("Mean difference (t2 - t1)", f"{mean_diff:.4f}")
            st.metric("t-statistic", f"{t_stat:.3f}")
            st.metric("p-value", f"{t_p:.4f}")

            if t_p > 0.05:
                st.success("✓ No systematic change (p > 0.05) → GOOD for TE estimation")
            else:
                st.warning("✗ Systematic change detected (p < 0.05) → WARNING: may not be stable")

            st.divider()
            # 6. K-S test
            st.markdown("**6. K-S test (same distribution)**")
            ks_stat, ks_p = stats.ks_2samp(t1_paired, t2_paired)
            st.metric("K-S statistic", f"{ks_stat:.3f}")
            st.metric("p-value", f"{ks_p:.3f}")

            if ks_p > 0.05:
               st.success("✓ Same distribution → consistent with stability")
            else:
                st.warning("⚠ Different distributions → some drift or change")

            st.divider()

        with col2:
            # 2. Effect size
            cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0
            st.markdown("**Effect size (Cohen's d)**")
            st.metric("Cohen's d", f"{cohens_d:.3f}")

            if abs(cohens_d) < 0.2:
                st.success("✓ Very small effect (|d| < 0.2) → negligible change")
            elif abs(cohens_d) < 0.5:
                st.warning("⚠ Small-to-moderate effect (|d| = 0.2-0.5) → some change")
            else:
                st.error("✗ Moderate-to-large effect (|d| > 0.5) → substantial change")
            st.divider()


        # 3. Correlation
            st.markdown("**Pearson correlation**")
            r, r_p = stats.pearsonr(t1_paired, t2_paired)
            st.metric("Correlation (r)", f"{r:.4f}")
            st.metric("p-value", f"{r_p:.3e}")

            if r > 0.95:
                st.success("✓ Very high correlation (r > 0.95) → excellent stability")
            elif r > 0.8:
                st.warning("⚠ Good correlation (r = 0.8-0.95) → acceptable stability")
            else:
                st.error("✗ Moderate/low correlation (r < 0.8) → poor stability")
            st.divider()


            st.markdown("**ICC**")
            icc_value, icc_details = icc_31_from_two_timepoints(df, t1_col, t2_col)
            st.metric("ICC value", f"{icc_value:.4f}")

            # Interpretation (Koo & Li, 2016)
            if icc_value < 0.50:
                icc_interp = "Poor reliability"
            elif icc_value < 0.75:
                icc_interp = "Moderate reliability"
            elif icc_value < 0.90:
                icc_interp = "Good reliability"
            else:
                icc_interp = "Excellent reliability"

            st.markdown(f"**Interpretation:** {icc_interp}")
            st.divider()

        with col3:
            # 4. CV
            cv_per_pair = (np.abs(t2_paired - t1_paired) / ((t1_paired + t2_paired) / 2)) * 100
            mean_cv = cv_per_pair.mean()
            max_cv = cv_per_pair.max()

            st.markdown("**Coefficient of Variation (CV)**")
            st.metric("Mean CV", f"{mean_cv:.2f}%")
            st.metric("Max CV", f"{max_cv:.2f}%")

            if mean_cv < 5:
                st.success("✓ Low CV (< 5%) → good stability")
            elif mean_cv < 10:
                st.warning("⚠ Moderate CV (5-10%) → acceptable")
            else:
                st.error("✗ High CV (> 10%) → poor stability")

            st.divider()

            # 5. Bland-Altman

            st.markdown("**Bland-Altman analysis**")
            diff = t2_paired - t1_paired
            mean_val = (t1_paired + t2_paired) / 2
            loa_lower = mean_diff - 1.96 * sd_diff
            loa_upper = mean_diff + 1.96 * sd_diff

            st.metric("Mean difference", f"{mean_diff:.4f}")
            st.metric("SD of differences", f"{sd_diff:.4f}")
            st.metric("95% limits of agreement", f"[{loa_lower:.2f}, {loa_upper:.2f}]")

            st.divider()



        # 7. Visualizations
        with st.expander("**Visualizations**", expanded=False):

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Scatter plot
            axes[0, 0].scatter(t1_paired, t2_paired, alpha=0.6, edgecolors='black')
            axes[0, 0].plot([t1_paired.min(), t1_paired.max()],
                            [t1_paired.min(), t1_paired.max()], 'r--', linewidth=2, label='y=x')
            axes[0, 0].set_xlabel(t1_name)
            axes[0, 0].set_ylabel(t2_name)
            axes[0, 0].set_title(f"Scatter plot (r = {r:.3f})")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Bland-Altman
            axes[0, 1].scatter(mean_val, diff, alpha=0.6, edgecolors='black')
            axes[0, 1].axhline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean diff: {mean_diff:.3f}')
            axes[0, 1].axhline(mean_diff + 1.96*sd_diff, color='gray', linestyle=':', label=f'+1.96 SD')
            axes[0, 1].axhline(mean_diff - 1.96*sd_diff, color='gray', linestyle=':', label=f'-1.96 SD')
            axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
            axes[0, 1].set_xlabel(f"Mean of {t1_name} and {t2_name}")
            axes[0, 1].set_ylabel(f"Difference ({t2_name} - {t1_name})")
            axes[0, 1].set_title("Bland-Altman plot")
            axes[0, 1].legend(fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)

            # Histograms
            axes[1, 0].hist(t1_paired, alpha=0.5, label=t1_name, bins=15, edgecolor='black')
            axes[1, 0].hist(t2_paired, alpha=0.5, label=t2_name, bins=15, edgecolor='black')
            axes[1, 0].set_xlabel("Value")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title(f"Distributions (K-S p = {ks_p:.3f})")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # ECDF
            sns.ecdfplot(t1_paired, label=t1_name, ax=axes[1, 1])
            sns.ecdfplot(t2_paired, label=t2_name, ax=axes[1, 1])
            axes[1, 1].set_xlabel("Value")
            axes[1, 1].set_ylabel("ECDF")
            axes[1, 1].set_title("Empirical CDFs")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        # 8. Verdict
        st.divider()
        st.markdown("**8. Overall Conclusion**")

        stable_criteria = 0
        unstable_criteria = 0

        if t_p > 0.05:
            stable_criteria += 1
        else:
            unstable_criteria += 1

        if abs(cohens_d) < 0.2:
            stable_criteria += 1
        elif abs(cohens_d) >= 0.5:
            unstable_criteria += 1

        if r > 0.95:
            stable_criteria += 1
        elif r < 0.8:
            unstable_criteria += 1

        if mean_cv < 5:
            stable_criteria += 1
        elif mean_cv >= 10:
            unstable_criteria += 1

        verdict = 'stable' if stable_criteria >= 3 else 'unstable' if unstable_criteria >= 3 else 'inconclusive'

        if verdict == 'stable':
            st.success(f"✅ CONCLUSION: t1 and t2 are STABLE → suitable for TE estimation (passed {stable_criteria}/4 criteria)")
            st.info(f"TE = SD(t2 - t1) = {sd_diff:.4f}")
        elif verdict == 'unstable':
            st.error(f"❌ CONCLUSION: t1 and t2 are NOT STABLE → NOT suitable for TE estimation (failed {unstable_criteria}/4 criteria)")
            st.warning("There may be systematic drift, learning effect, or biological change.")
        else:
            st.warning(f"⚠️ CONCLUSION: INCONCLUSIVE → use with caution (mixed results: {stable_criteria} pass, {unstable_criteria} fail)")
            st.info("Some metrics suggest stability, others don't. Proceed but acknowledge uncertainty.")

    # Return metrics
    return {
        'n': len(t1_paired),
        'mean_diff': mean_diff,
        'sd_diff': sd_diff,
        't_p': t_p,
        'cohens_d': cohens_d,
        'icc_value':icc_value,
        'correlation': r,
        'mean_cv': mean_cv,
        'ks_p': ks_p,
        'verdict': verdict
    }







def plot_individual_responders_matplotlib(
    df_individual: pd.DataFrame,
    id_col: str,
    change_col: str = "relative_change_proportion",
    lower_col: str = "ci_lower_proportion",
    upper_col: str = "ci_upper_proportion",
    swc_col: str = "swc_relative_proportion",
    classification_col: str = "ci_classification",
    group_col: str = "analysis_group",
    group_order: Optional[List[str]] = None,
    ci_label: str = "95% CI",
    title: Optional[str] = None,
    sort_within_group: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot individual relative changes with error bars and SWC thresholds using matplotlib.

    Parameters
    ----------
    df_individual : pd.DataFrame
        Individual-level results with change scores and classifications
    id_col : str
        Column with participant identifiers
    change_col : str
        Column with change scores
    lower_col : str
        Column with CI lower bounds
    upper_col : str
        Column with CI upper bounds
    swc_col : str
        Column with SWC threshold values
    classification_col : str
        Column with responder classifications
    group_col : str
        Column with grouping variable (default: "analysis_group")
        Use "All participants" for single-group plots
    group_order : list, optional
        Custom order for groups
    ci_label : str
        Label for confidence interval in legend
    title : str, optional
        Plot title (auto-generated if None)
    sort_within_group : bool
        Sort participants within groups by classification

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    """
    required_cols = {id_col, change_col, lower_col, upper_col, swc_col, classification_col, group_col}

    missing_cols = required_cols - set(df_individual.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    data = df_individual.dropna(subset=required_cols).copy()

    colors = {
        "reliable_meaningful_responder": "#1B9E77",
        "potential_responder": "#E6AB02",
        "non_responder": "#B22222",
    }

    class_order = {
        "non_responder": 0,
        "potential_responder": 1,
        "reliable_meaningful_responder": 2,
    }

    data["bar_color"] = data[classification_col].map(colors).fillna("#808080")
    data["_class_order"] = data[classification_col].map(class_order).fillna(99)

    # Set up grouping
    plotting_groups = (
        group_order
        if group_order is not None
        else data[group_col].dropna().unique().tolist()
    )
    plotting_groups = [g for g in plotting_groups if g in data[group_col].unique()]
    sort_cols = [group_col, "_class_order", change_col]

    if sort_within_group:
        data = data.sort_values(sort_cols)

    data = data.drop(columns="_class_order")

    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_positions = []
    x_labels = []
    group_centers = {}
    group_boundaries = []
    current_x = 0
    group_gap = 2

    for group in plotting_groups:
        group_data = data[data[group_col] == group].copy()
        if group_data.empty:
            continue

        x = np.arange(current_x, current_x + len(group_data))
        y = group_data[change_col].to_numpy()
        lower = group_data[lower_col].to_numpy()
        upper = group_data[upper_col].to_numpy()

        yerr = np.vstack([np.maximum(y - lower, 0), np.maximum(upper - y, 0)])

        ax.bar(x, y, color=group_data["bar_color"], edgecolor="black", linewidth=0.5, alpha=0.9, zorder=1)
        ax.errorbar(x, y, yerr=yerr, fmt='o', ecolor='black', elinewidth=1, capsize=7, marker='none', zorder=3)

        swc_values = group_data[swc_col].dropna().unique()
        if len(swc_values) != 1:
            raise ValueError("Each plotting group must have exactly one SWC value.")

        ax.hlines(swc_values[0], x.min() - 0.45, x.max() + 0.45, color="#2C3E50", linestyle="--", linewidth=2, zorder=4)

        x_positions.extend(x)
        x_labels.extend(group_data[id_col].astype(str))
        group_centers[group] = x.mean()
        group_boundaries.append(x.max() + 0.5)
        current_x = x.max() + group_gap + 1

    ax.axhline(0, color="black", linewidth=0.9, zorder=1)

    # Add group separators (skip if only one group)
    if len(group_boundaries) > 1:
        for boundary in group_boundaries[:-1]:
            ax.axvline(boundary, color="grey", linestyle=":", linewidth=0.8, alpha=0.8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Relative change (proportion)")

    if title is None:
        threshold_label = "overall" if len(plotting_groups) == 1 else "group-specific"
        title = f"Individual relative change with {ci_label} and {threshold_label} SWC thresholds"

    ax.set_title(title)

    # Add group labels below x-axis (skip if only one group)
    if len(group_centers) > 1:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        for group, center in group_centers.items():
            ax.text(center, y_min - 0.10 * y_range, str(group), ha="center", va="top", fontsize=11, fontweight="bold")
        ax.set_ylim(y_min - 0.2 * y_range, y_max)

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=label.replace("_", " ").title())
        for label, color in colors.items()
    ]
    legend_handles += [
        plt.Line2D([0], [0], color="black", marker="_", markersize=2, linewidth=1, label=ci_label),
        plt.Line2D([0], [0], color="#2C3E50", linestyle="--", linewidth=2, label="Threshold"),
    ]

    ax.legend(handles=legend_handles, loc="upper left", frameon=False)
    ax.margins(x=0.01)
    plt.tight_layout()

    return fig, ax





def calculate_swc_te(
    df: pd.DataFrame,
    id_col: str,
    pre_col: str,
    post_col: str,
    t1_col: str,
    t2_col: str,
    plot: bool = False,
    group_col: Optional[str] = None,
    threshold_type: str = None,
    swc_multiplier: float = 0.2,
    ci_level: float = 0.5,
    higher_is_better: bool = True,
    use_relative: bool = False,
    error_metric: str = "TE",  # "TE" or "SEM"
    icc_value: float | None = None,  # optional: if provided, SEM = SD_baseline * sqrt(1 - ICC),
    group_order: Optional[List[str]] = None,
    sort_within_group: bool = True,
    # figsize_per_participant: float = 0.32,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Calculate SWC and TE in raw or relative units.

    Raw change: post - pre
    Relative change: (post - pre) / pre

    Raw SWC: swc_multiplier × SD(pre)
    Relative SWC: swc_multiplier × SD(pre) / mean(pre)  [expressed as proportion]

    TE: SD(t2 - t1)  [for change scores, no sqrt(2) division]

    Interval: change ± z × TE

    ci_level=0.5 gives approximately 50% CI (z ≈ 0.674)
    ci_level=0.6827 gives approximately change ± TE (z ≈ 1)

    Returns:
        summary_df: Summary statistics per group
        individual_df: Individual-level results with classifications
        fig, ax: Matplotlib figure and axes (if plot=True)
    """
    required_cols = {id_col, pre_col, post_col, t1_col, t2_col}
    if group_col is not None:
        required_cols.add(group_col)

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1.")

    data = df.loc[:, list(required_cols)].copy()

    if group_col is None:
        group_used = "_analysis_group"
        data[group_used] = "All participants"
    else:
        group_used = group_col

    z_value = NormalDist().inv_cdf(1 - (1 - ci_level) / 2)
    interval_label = (
        "Change ± TE"
        if np.isclose(z_value, 1.0, atol=0.02)
        else f"{ci_level:.1%} CI"
    )

    summary_rows = []
    individual_results = []

    for group_name, group_df in data.groupby(group_used, dropna=False):
        # print(group_name)
        group_df = group_df.copy()
        intervention_df = group_df.dropna(subset=[id_col, pre_col, post_col]).copy()

        if intervention_df.empty:
            print(f"Skipping '{group_name}': no valid pre/post data.")
            continue

        if intervention_df[id_col].duplicated().any():
            raise ValueError(f"Duplicate participant IDs found in '{group_name}'.")

        # Individual change
        pre_sd = intervention_df[pre_col].std()
        pre_mean = intervention_df[pre_col].mean()
        intervention_df["change"] = intervention_df[post_col] - intervention_df[pre_col]
        if use_relative:
            intervention_df["change"] = (intervention_df["change"]) / intervention_df[pre_col]
        else:
            intervention_df["change"] = intervention_df["change"]

        if threshold_type=='SWC-Based':
            if use_relative:
                swc = swc_multiplier * pre_sd / pre_mean  # relative SWC as proportion
            else:
                swc = swc_multiplier * pre_sd  # raw SWC
        else:
            swc = 'Nan'

        if not higher_is_better:
            intervention_df["change"] *= -1

        pre_mean_val = intervention_df[pre_col].mean()

        # TE from retest
        retest_df = group_df.dropna(subset=[id_col, t1_col, t2_col]).copy()

        if retest_df.empty:
            print(f"Skipping '{group_name}': no valid t1/t2 data.")
            continue

        if use_relative:
            retest_df["retest_difference"] = (retest_df[t2_col] - retest_df[t1_col]) / retest_df[t1_col]
        else:
            retest_df["retest_difference"] = retest_df[t2_col] - retest_df[t1_col]

        sd_diff= retest_df["retest_difference"].std()


        # TE and SEM
        te_change = sd_diff  # SD of difference scores
        icc_value, icc_details = icc_31_from_two_timepoints(df, t1_col, t2_col)
        if icc_value is not None:
            sem_change = pre_sd * np.sqrt(1 - icc_value)
            if use_relative:
                sem_change = sem_change / pre_mean
        else:
            # Difference-score based SEM: SEM = SD_diff / sqrt(2)
            sem_change = sd_diff / np.sqrt(2)

        # Choose error metric for CI
        if error_metric == "TE":
            error_value = te_change
        elif error_metric == "SEM":
            error_value = sem_change
        else:
            raise ValueError("error_metric must be 'TE' or 'SEM'.")

        # Interval around change
        interval_margin = z_value * error_value
        intervention_df["analysis_group"] = group_name
        intervention_df["te_change"] = te_change
        intervention_df["sem_change"] = sem_change
        intervention_df["error_metric_used"] = error_metric
        intervention_df["interval_margin"] = interval_margin
        intervention_df["interval_lower"] = intervention_df["change"] - interval_margin
        intervention_df["interval_upper"] = intervention_df["change"] + interval_margin
        intervention_df["ci_level"] = ci_level
        intervention_df["z_value"] = z_value
        intervention_df["interval_label"] = interval_label

        # Classification
        if threshold_type == 'SWC-Based':

            intervention_df["swc"] = swc
            threshold = intervention_df["swc"]
        elif threshold_type == 'Zero-Based':
            intervention_df['zero_threshold'] = 0
            intervention_df["swc"] = 0
            threshold =  0
        observed_responder = intervention_df["change"] > threshold
        reliable_responder = intervention_df["interval_lower"] > threshold
        intervention_df["classification"] = np.select(
            [reliable_responder, observed_responder & ~reliable_responder],
            ["reliable_meaningful_responder", "potential_responder"],
            default="non_responder",
        )


        # Summary
        summary_rows.append({
            "analysis_group": group_name,
            "n_intervention": len(intervention_df),
            "n_retest": len(retest_df),
            "pre_mean": pre_mean_val,
            "pre_sd": pre_sd,
            "swc_multiplier": swc_multiplier,
            "swc": swc,
            'threshold':threshold,
            "retest_difference_mean": retest_df["retest_difference"].mean(),
            "retest_difference_sd": retest_df["retest_difference"].std(),
            "te_change": te_change,
            "sem_change": sem_change,
            "error_metric_used": error_metric,
            "ci_level": ci_level,
            # "z_value": z_value,
            "interval_label": interval_label,
            "analysis_type": "relative" if use_relative else "raw",
        })
        individual_results.append(intervention_df)

    summary_df = pd.DataFrame(summary_rows)
    individual_df = pd.concat(individual_results, ignore_index=True) if individual_results else pd.DataFrame()

    fig, ax = None, None
    if plot and not individual_df.empty:
        y_label = "Relative change (post - pre) / pre" if use_relative else "Raw change (post - pre)"
        threshold_label = "overall" if group_col is None else "group-specific"
        plot_title = f"Individual {'relative' if use_relative else 'raw'} change with {interval_label} and {threshold_type} thresholds"

        fig, ax = plot_individual_responders_matplotlib(
            df_individual=individual_df,
            id_col=id_col,
            group_col="analysis_group",
            change_col="change",
            lower_col="interval_lower",
            upper_col="interval_upper",
            swc_col="swc",
            classification_col="classification",
            group_order=group_order,
            ci_label=interval_label,
            title=plot_title,
            sort_within_group=sort_within_group,
        )

    return summary_df, individual_df, fig, ax






def fig_to_png_buffer(fig: plt.Figure) -> io.BytesIO:
    """Convert matplotlib figure to PNG buffer for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf