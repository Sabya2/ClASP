"""Streamlit controls and plots for longitudinal group comparisons."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from srcs.utils_visualisation.common import get_figure_download, set_paper_plot_style
from srcs.utils_visualisation.repeated_measures_stats import (
    RepeatedMeasuresValidationError,
    run_linear_mixed_model,
    run_mixed_anova,
    run_repeated_measures,
    wide_to_long,
)


ANALYSIS_HELP = """
### Choosing a design

- **Independent Groups** compares unrelated participants and retains CLASP's t-test/ANOVA or rank-based workflow.
- **Paired / Repeated Measures** compares the same participants at two or more selected times. Two times use a paired t-test or Wilcoxon signed-rank test; three or more use repeated-measures ANOVA or Friedman, followed by Holm-adjusted paired comparisons.
- **Group × Time (Mixed Design)** asks whether longitudinal change differs between independent groups. Complete records can be analysed with split-plot ANOVA. A random-intercept linear mixed-effects model can retain incomplete records, but reports treatment-coded coefficients rather than omnibus ANOVA effects.

### Data and assumptions

Long data need explicit participant, time, and outcome columns. Wide data need an ID and explicitly selected columns—one column per time point. Row order is never used to infer pairs. Duplicate participant/time records are rejected rather than averaged, and each participant in a mixed design must have one consistent group.

Paired tests exclude incomplete pairs. Repeated-measures and mixed ANOVA use participants complete at every selected time. Automatic two-time selection applies Shapiro–Wilk to paired differences. For three or more times, automatic selection requires all pairwise difference checks to support the parametric route; this is a convenience check, not proof of multivariate normality. Repeated-measures ANOVA reports Mauchly's approximate sphericity test and uses Greenhouse–Geisser-adjusted degrees of freedom. Mixed models assume a correctly specified random-intercept model and missingness compatible with likelihood inference.

Transformations are fitted once across all selected observations for the outcome, never separately by group or time. `log10` excludes non-positive values; constant outcomes cannot be min-max or z-score transformed. Exclusions are reported. Post-hoc p-values are Holm-adjusted within each outcome, not across selected outcomes.

Main effects average across the other factor. The **Group × Time** interaction tests whether trajectories differ between groups; it is not, by itself, evidence of a causal treatment effect. Small samples, outliers, non-normal differences, covariance misspecification, and convergence warnings require cautious interpretation.
"""


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "outcome"


def _format_p(value):
    if not np.isfinite(value):
        return "n/a"
    return "<0.001" if value < .001 else f"{value:.3f}"


def format_repeated_summary(analysis):
    result = analysis["result"]
    counts = analysis["counts"]
    lines = [
        f"**{analysis['outcome']}** ({analysis['design']}; scale = {counts['transformation']})",
        f"Complete participants: **{analysis['n_included']}**; excluded incomplete participants: "
        f"**{analysis['n_excluded_incomplete']}**. Usable observations: {counts['observations']}.",
    ]
    if counts["invalid_transformation_values"]:
        lines.append(f"Transformation excluded {counts['invalid_transformation_values']} otherwise numeric observation(s).")
    summaries = "; ".join(
        f"{time}: mean {row['mean']:.3g} (SD {row['sd']:.3g}), median {row['median']:.3g} (IQR {row['iqr']:.3g}), n={row['n']}"
        for time, row in analysis["summaries"].items()
    )
    lines.append(f"Time-point summaries: {summaries}.")
    if len(analysis["times"]) == 2:
        ci = result["change_ci_95"]
        effect = (f"Cohen's dz = {result['cohens_dz']:.3g}" if result["test"] == "Paired t-test"
                  else f"matched rank-biserial r = {result['rank_biserial']:.3g}")
        lines.append(
            f"**{result['test']}**: statistic = {result['statistic']:.3g}, p = {_format_p(result['p'])}; "
            f"mean paired change = {result['mean_change']:.3g}, 95% CI [{ci[0]:.3g}, {ci[1]:.3g}]; {effect}."
        )
        lines.append(f"Shapiro–Wilk on paired differences: p = {_format_p(result['normality_p'])}.")
    else:
        if result["test"] == "Repeated-measures ANOVA":
            lines.append(
                f"**Repeated-measures ANOVA**: F({result['df1_corrected']:.2f}, "
                f"{result['df2_corrected']:.2f}) = {result['statistic']:.3g}, p = {_format_p(result['p'])}, "
                f"partial η² = {result['partial_eta_squared']:.3g}. Greenhouse–Geisser ε = "
                f"{result['gg_epsilon']:.3g}; Mauchly p = {_format_p(result['mauchly_p'])}."
            )
        else:
            lines.append(f"**Friedman test**: χ² = {result['statistic']:.3g}, p = {_format_p(result['p'])}.")
        normality = ", ".join(
            f"{contrast}: {_format_p(p_value)}"
            for contrast, p_value in result.get("difference_normality_p", {}).items()
        )
        if normality:
            lines.append(f"Shapiro–Wilk checks on pairwise differences: {normality}.")
        if analysis["pairwise"]:
            lines.append("Holm-adjusted paired post-hoc comparisons:")
            for row in analysis["pairwise"]:
                lines.append(
                    f"- {row['time1']} vs {row['time2']}: {row['test']}, statistic = "
                    f"{row['statistic']:.3g}, adjusted p = {_format_p(row.get('p_corrected', row['p']))}."
                )
    return "\n\n".join(lines)


def format_mixed_summary(analysis):
    counts = analysis["counts"]
    if analysis["design"] == "linear mixed-effects model":
        lines = [
            f"**{analysis['outcome']}** (random-intercept linear mixed-effects model; scale = {counts['transformation']})",
            f"{analysis['n_observations']} observations from {analysis['n_participants']} participants. "
            f"Converged: **{'yes' if analysis['converged'] else 'no'}**.",
            analysis["inference"],
            "Fixed-effect coefficients (relative to reference levels):",
        ]
        for row in analysis["terms"]:
            lines.append(
                f"- {row['term']}: estimate {row['estimate']:.3g}, SE {row['se']:.3g}, "
                f"95% CI [{row['ci_low']:.3g}, {row['ci_high']:.3g}], p = {_format_p(row['p'])}."
            )
        if analysis["warnings"]:
            lines.append("Model warnings: " + " | ".join(analysis["warnings"]))
        lines.append("Coefficient tests are not omnibus ANOVA tests; interaction coefficients compare specific reference-level contrasts.")
        return "\n\n".join(lines)

    sizes = ", ".join(f"{group}: n={size}" for group, size in analysis["group_sizes"].items())
    lines = [
        f"**{analysis['outcome']}** (mixed-design ANOVA; scale = {counts['transformation']})",
        f"Complete participants: **{analysis['n_included']}** ({sizes}); excluded incomplete participants: "
        f"**{analysis['n_excluded_incomplete']}**. Greenhouse–Geisser ε = "
        f"{analysis['sphericity']['gg_epsilon']:.3g}; Mauchly p = {_format_p(analysis['sphericity']['mauchly_p'])}.",
        f"Assumption diagnostics: residual Shapiro–Wilk p = "
        f"{_format_p(analysis['assumptions']['residual_shapiro_p'])}; Levene p for participant means = "
        f"{_format_p(analysis['assumptions']['subject_mean_levene_p'])}.",
    ]
    for row in analysis["effects"]:
        lines.append(
            f"- **{row['effect']}**: F({row['df1']:.2f}, {row['df2']:.2f}) = {row['F']:.3g}, "
            f"p = {_format_p(row['p'])}, partial η² = {row['partial_eta_squared']:.3g}."
        )
    interaction = next(row for row in analysis["effects"] if row["effect"] == "Group × Time")
    if np.isfinite(interaction["p"]) and interaction["p"] < .05:
        lines.append("The Group × Time interaction was statistically significant, suggesting that change differed between the selected groups; this is not automatically a causal effect.")
        if analysis["posthoc"]:
            lines.append("Interaction follow-ups (Holm-adjusted as one family):")
            for row in analysis["posthoc"]:
                lines.append(
                    f"- {row['contrast']}: {row['test']}, statistic = {row['statistic']:.3g}, "
                    f"adjusted p = {_format_p(row.get('p_corrected', row['p']))}."
                )
    else:
        lines.append("The Group × Time interaction was not statistically significant; the analysis did not detect different changes between groups.")
    return "\n\n".join(lines)


def _trajectory_plot(analysis, participant_col, time_col, outcome_col, group_col=None):
    data = analysis["plot_data"]
    times = analysis["times"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    set_paper_plot_style()
    order = {value: index for index, value in enumerate(times)}
    data = data.assign(_time_order=data[time_col].map(order)).sort_values([participant_col, "_time_order"])
    if group_col:
        palette = dict(zip(data[group_col].drop_duplicates(), sns.color_palette(n_colors=data[group_col].nunique())))
        for _, participant in data.groupby(participant_col):
            group = participant[group_col].iloc[0]
            ax.plot(participant["_time_order"], participant[outcome_col], color=palette[group], alpha=.16, lw=.8)
        summary = data.groupby([group_col, time_col], observed=True)[outcome_col].agg(["mean", "sem"]).reset_index()
        for group, part in summary.groupby(group_col, observed=True):
            x = part[time_col].map(order).to_numpy()
            ax.errorbar(x, part["mean"], yerr=1.96 * part["sem"], marker="o", lw=2,
                        capsize=3, label=str(group), color=palette[group])
        ax.legend(title=group_col, frameon=False)
        ax.set_title(f"{outcome_col}: group trajectories and mean ± 95% CI")
    else:
        for _, participant in data.groupby(participant_col):
            ax.plot(participant["_time_order"], participant[outcome_col], color="#777777", alpha=.28, lw=.8, marker="o", ms=2)
        summary = data.groupby(time_col, observed=True)[outcome_col].agg(["mean", "sem"]).reindex(times)
        ax.errorbar(range(len(times)), summary["mean"], yerr=1.96 * summary["sem"], color="#1f77b4",
                    marker="o", lw=2.3, capsize=3, label="Mean ± 95% CI")
        ax.legend(frameon=False)
        ax.set_title(f"{outcome_col}: participant trajectories")
    ax.set_xticks(range(len(times)), [str(value) for value in times])
    ax.set_xlabel(time_col)
    ax.set_ylabel(f"{outcome_col} ({analysis['counts']['transformation']})")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def render_longitudinal_comparison_page(df, continuous_cols, group_col, participants_col, mixed=False):
    """Render either the repeated-measures or Group × Time workflow."""
    if df is None:
        st.warning("Upload data first.")
        return {}, [], 1
    with st.expander("How paired, repeated, and mixed analyses work", expanded=False):
        st.markdown(ANALYSIS_HELP)

    columns = df.columns.tolist()
    if len(columns) < 3:
        st.warning("This analysis requires participant, time, and outcome columns.")
        return {}, [], 1
    format_choice = st.radio("Data format", ["Long", "Wide"], horizontal=True, key=f"rm_format_{mixed}")
    default_id = participants_col if participants_col in columns else columns[0]
    id_col = st.selectbox("Participant ID variable", columns, index=columns.index(default_id), key=f"rm_id_{mixed}")

    analysis_df = df
    if format_choice == "Long":
        time_options = [column for column in columns if column != id_col]
        preferred_time = next(
            (column for column in time_options if str(column).strip().lower() in {"time", "timepoint", "time_point", "visit"}),
            next((column for column in time_options if column != group_col), time_options[0]),
        )
        time_col = st.selectbox(
            "Time-point variable", time_options, index=time_options.index(preferred_time), key=f"rm_time_{mixed}"
        )
        outcome_options = [column for column in continuous_cols if column in columns and column not in {id_col, time_col}]
        if not outcome_options:
            outcome_options = [column for column in columns if column not in {id_col, time_col}]
        outcomes = st.multiselect("Continuous outcome variables", outcome_options,
                                  default=outcome_options[: min(3, len(outcome_options))], key=f"rm_outcomes_{mixed}")
    else:
        candidates = [column for column in continuous_cols if column in columns and column != id_col]
        repeated_columns = st.multiselect(
            "Repeated-measure columns (ordered as time points)", candidates,
            default=candidates[: min(2, len(candidates))], key=f"rm_wide_columns_{mixed}",
            help="Each selected column is one time point. The displayed column names are the explicit time labels.",
        )
        time_col = "time"
        outcomes = ["outcome"] if len(repeated_columns) >= 2 else []

    selected_group_col = None
    selected_groups = None
    mapped_measure_columns = set(outcomes + ([time_col] if format_choice == "Long" else repeated_columns))
    group_candidates = [column for column in columns if column != id_col and column not in mapped_measure_columns]
    if mixed and not group_candidates:
        st.warning("Select mappings that leave a distinct between-subject group column.")
        return {}, outcomes, 1
    if mixed:
        default_group = group_col if group_col in group_candidates else group_candidates[0]
        selected_group_col = st.selectbox("Between-subject grouping variable", group_candidates,
                                          index=group_candidates.index(default_group), key="mixed_group_col")
    elif st.checkbox("Filter by a group", value=False, key="paired_group_filter"):
        if not group_candidates:
            st.warning("No distinct column is available for an optional group filter.")
            return {}, outcomes, 1
        default_group = group_col if group_col in group_candidates else group_candidates[0]
        selected_group_col = st.selectbox("Group filter variable", group_candidates,
                                          index=group_candidates.index(default_group), key="paired_group_col")
    if selected_group_col:
        group_values = df[selected_group_col].dropna().drop_duplicates().tolist()
        selected_groups = st.multiselect("Group values", group_values, default=group_values, key=f"rm_groups_{mixed}")

    if format_choice == "Wide" and outcomes:
        try:
            analysis_df = wide_to_long(df, id_col, repeated_columns, "outcome", selected_group_col)
        except RepeatedMeasuresValidationError as exc:
            st.error(str(exc))
            return {}, [], 1
    available_times = analysis_df[time_col].dropna().drop_duplicates().tolist() if time_col in analysis_df else []
    selected_times = st.multiselect("Time points", available_times, default=available_times, key=f"rm_times_{mixed}")
    transformation = st.selectbox("Transformation", ["none", "log10", "min-max", "z-score"], key=f"rm_transform_{mixed}")
    grid_cols = st.slider("Grid columns", 1, 4, 2, key=f"rm_grid_{mixed}")
    if mixed:
        model = st.selectbox("Statistical model", ["Mixed-design ANOVA", "Linear mixed-effects model"], key="mixed_model")
        method = None
    else:
        method_label = st.selectbox("Statistical method", ["Automatic", "Parametric", "Non-parametric"], key="rm_method")
        method = {"Automatic": "auto", "Parametric": "parametric", "Non-parametric": "nonparametric"}[method_label]

    if not outcomes or len(selected_times) < 2 or (selected_group_col and not selected_groups):
        st.info("Select at least one outcome, two time points, and any required groups.")
        return {}, outcomes, grid_cols

    summaries = {}
    st.subheader("Repeated-measures plots" if not mixed else "Group × Time interaction plots")
    for start in range(0, len(outcomes), grid_cols):
        layout = st.columns(grid_cols)
        for container, outcome in zip(layout, outcomes[start:start + grid_cols]):
            with container:
                try:
                    if mixed and model == "Mixed-design ANOVA":
                        analysis = run_mixed_anova(
                            analysis_df, id_col, selected_group_col, time_col, outcome,
                            selected_times, selected_groups, transformation,
                        )
                    elif mixed:
                        analysis = run_linear_mixed_model(
                            analysis_df, id_col, selected_group_col, time_col, outcome,
                            selected_times, selected_groups, transformation,
                        )
                    else:
                        analysis = run_repeated_measures(
                            analysis_df, id_col, time_col, outcome, selected_times,
                            method, transformation, selected_group_col, selected_groups,
                        )
                    summaries[outcome] = format_mixed_summary(analysis) if mixed else format_repeated_summary(analysis)
                    fig = _trajectory_plot(analysis, id_col, time_col, outcome, selected_group_col if mixed else None)
                    st.pyplot(fig)
                    st.download_button(
                        f"Download {outcome}", get_figure_download(fig),
                        f"{'mixed' if mixed else 'repeated'}_{_safe_name(outcome)}.png", "image/png",
                        key=f"rm_download_{mixed}_{outcome}",
                    )
                    plt.close(fig)
                except RepeatedMeasuresValidationError as exc:
                    summaries[outcome] = f"**{outcome}:** analysis not run. {exc}"
                    st.error(str(exc))
                except Exception as exc:
                    summaries[outcome] = f"**{outcome}:** numerical analysis failed. {exc}"
                    st.error(f"{outcome}: numerical analysis failed; no result was substituted. {exc}")
    return summaries, outcomes, grid_cols
