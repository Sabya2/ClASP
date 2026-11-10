"""Statistical helpers for paired, repeated-measures, and mixed designs.

The functions in this module do not depend on Streamlit.  They accept long-form
data and return dictionaries that the Group Comparison page can render or test.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests


class RepeatedMeasuresValidationError(ValueError):
    """Raised when the selected columns do not define a valid repeated design."""


def wide_to_long(df, participant_col, value_columns, outcome_name="outcome", group_col=None):
    """Convert explicitly selected repeated-measure columns to canonical long form."""
    value_columns = list(value_columns)
    if len(value_columns) < 2:
        raise RepeatedMeasuresValidationError("Select at least two repeated-measure columns.")
    required = [participant_col, *value_columns]
    if group_col:
        required.append(group_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RepeatedMeasuresValidationError(f"Columns not found: {', '.join(map(str, missing))}.")
    id_vars = [participant_col] + ([group_col] if group_col else [])
    return df[required].melt(
        id_vars=id_vars,
        value_vars=value_columns,
        var_name="time",
        value_name=outcome_name,
    )


def _transform(values, transformation):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if transformation == "none":
        transformed = values
    elif transformation == "log10":
        transformed = np.log10(values.where(values > 0))
    elif transformation == "min-max":
        span = values.max() - values.min()
        transformed = (values - values.min()) / span if pd.notna(span) and span > 0 else values * np.nan
    elif transformation == "z-score":
        standard_deviation = values.std()
        transformed = ((values - values.mean()) / standard_deviation
                       if pd.notna(standard_deviation) and standard_deviation > 0 else values * np.nan)
    else:
        raise ValueError(f"Unsupported transformation: {transformation}")
    return transformed.replace([np.inf, -np.inf], np.nan)


def prepare_long_data(
    df,
    participant_col,
    time_col,
    outcome_col,
    selected_times=None,
    group_col=None,
    selected_groups=None,
    transformation="none",
):
    """Validate and clean one long-form outcome without aggregating duplicates."""
    columns = [participant_col, time_col, outcome_col] + ([group_col] if group_col else [])
    if len(set(columns)) != len(columns):
        raise RepeatedMeasuresValidationError("Participant, time, outcome, and group columns must be distinct.")
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise RepeatedMeasuresValidationError(f"Columns not found: {', '.join(map(str, missing_columns))}.")

    data = df[columns].copy()
    if selected_times is not None:
        data = data[data[time_col].isin(selected_times)]
    if group_col and selected_groups is not None:
        data = data[data[group_col].isin(selected_groups)]
    data = data.dropna(subset=[participant_col, time_col])
    duplicate_mask = data.duplicated([participant_col, time_col], keep=False)
    if duplicate_mask.any():
        examples = data.loc[duplicate_mask, [participant_col, time_col]].drop_duplicates().head(5)
        example_text = ", ".join(f"({row[participant_col]}, {row[time_col]})" for _, row in examples.iterrows())
        raise RepeatedMeasuresValidationError(
            f"Duplicate participant/time observations were found ({int(duplicate_mask.sum())} rows; "
            f"examples: {example_text}). Resolve duplicates explicitly; CLASP will not average them."
        )
    if group_col:
        group_counts = data.dropna(subset=[group_col]).groupby(participant_col)[group_col].nunique()
        inconsistent = group_counts[group_counts > 1].index.tolist()
        if inconsistent:
            raise RepeatedMeasuresValidationError(
                f"{len(inconsistent)} participant(s) belong to more than one group; "
                f"examples: {', '.join(map(str, inconsistent[:5]))}."
            )

    participants_selected = int(data[participant_col].nunique())
    original_nonmissing = pd.to_numeric(data[outcome_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).notna()
    data[outcome_col] = _transform(data[outcome_col], transformation)
    invalid_transformation = int((original_nonmissing & data[outcome_col].isna()).sum())
    missing_outcomes = int(data[outcome_col].isna().sum())
    data = data.dropna(subset=[outcome_col] + ([group_col] if group_col else []))
    if data.empty:
        raise RepeatedMeasuresValidationError("No usable observations remain after filtering and transformation.")
    times = list(selected_times) if selected_times is not None else data[time_col].drop_duplicates().tolist()
    times = [time for time in times if time in set(data[time_col])]
    if len(times) < 2:
        raise RepeatedMeasuresValidationError("At least two time points with usable observations are required.")
    return data, times, {
        "observations": len(data),
        "participants_observed": int(data[participant_col].nunique()),
        "participants_selected": participants_selected,
        "missing_or_invalid_outcomes": missing_outcomes,
        "invalid_transformation_values": invalid_transformation,
        "transformation": transformation,
    }


def _complete_matrix(data, participant_col, time_col, outcome_col, times):
    matrix = data.pivot(index=participant_col, columns=time_col, values=outcome_col).reindex(columns=times)
    complete = matrix.dropna()
    return complete, int(len(matrix) - len(complete))


def _shapiro(values):
    values = np.asarray(values, dtype=float)
    if 3 <= len(values) <= 5000 and np.ptp(values) > 0:
        return float(stats.shapiro(values).pvalue)
    return np.nan


def _paired_effects(x, y):
    difference = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
    sd = np.std(difference, ddof=1) if len(difference) > 1 else np.nan
    dz = float(np.mean(difference) / sd) if np.isfinite(sd) and sd > 0 else np.nan
    nonzero = difference[difference != 0]
    if len(nonzero):
        ranks = stats.rankdata(np.abs(nonzero))
        rank_biserial = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    else:
        rank_biserial = np.nan
    return difference, dz, rank_biserial


def _paired_test(x, y, method, alpha=0.05):
    difference, dz, rank_biserial = _paired_effects(x, y)
    normality_p = _shapiro(difference)
    selected = method if method != "auto" else ("parametric" if np.isfinite(normality_p) and normality_p >= alpha else "nonparametric")
    if selected == "parametric" and np.ptp(difference) == 0:
        statistic, p_value = (0.0, 1.0) if difference[0] == 0 else (np.inf, 0.0)
        test = "Paired t-test"
    elif selected == "parametric":
        result = stats.ttest_rel(y, x)
        statistic, p_value, test = float(result.statistic), float(result.pvalue), "Paired t-test"
    else:
        if np.all(difference == 0):
            statistic, p_value = 0.0, 1.0
        else:
            result = stats.wilcoxon(y, x, alternative="two-sided")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        test = "Wilcoxon signed-rank test"
    mean_change = float(np.mean(difference))
    if len(difference) > 1 and np.std(difference, ddof=1) > 0:
        margin = stats.t.ppf(0.975, len(difference) - 1) * stats.sem(difference)
        confidence_interval = (float(mean_change - margin), float(mean_change + margin))
    else:
        confidence_interval = (np.nan, np.nan)
    return {
        "test": test, "statistic": statistic, "p": p_value,
        "normality_p": normality_p, "selected_method": selected,
        "mean_change": mean_change, "median_change": float(np.median(difference)),
        "change_ci_95": confidence_interval, "cohens_dz": dz,
        "rank_biserial": rank_biserial,
    }


def _sphericity(matrix):
    """Return Mauchly's approximate test and Greenhouse-Geisser epsilon."""
    n_subjects, n_times = matrix.shape
    if n_times <= 2:
        return {"mauchly_w": np.nan, "mauchly_p": np.nan, "gg_epsilon": 1.0}
    covariance = np.cov(np.asarray(matrix, dtype=float), rowvar=False, ddof=1)
    centering = np.eye(n_times) - np.ones((n_times, n_times)) / n_times
    eigenvalues = np.linalg.eigvalsh(centering @ covariance @ centering)
    eigenvalues = eigenvalues[eigenvalues > np.finfo(float).eps * max(1.0, np.max(np.abs(eigenvalues)))]
    p = n_times - 1
    if len(eigenvalues) != p or np.sum(eigenvalues ** 2) == 0:
        return {"mauchly_w": np.nan, "mauchly_p": np.nan, "gg_epsilon": 1.0 / p}
    epsilon = float(np.clip(np.sum(eigenvalues) ** 2 / (p * np.sum(eigenvalues ** 2)), 1 / p, 1.0))
    w = float(np.prod(eigenvalues) / (np.mean(eigenvalues) ** p))
    correction = (2 * p * p + p + 2) / (6 * p * max(n_subjects - 1, 1))
    chi_square = -max(n_subjects - 1, 1) * (1 - correction) * np.log(max(w, np.finfo(float).tiny))
    dof = p * (p + 1) / 2 - 1
    return {"mauchly_w": w, "mauchly_p": float(stats.chi2.sf(chi_square, dof)), "gg_epsilon": epsilon}


def _rm_anova(matrix):
    values = np.asarray(matrix, dtype=float)
    n_subjects, n_times = values.shape
    grand = values.mean()
    ss_time = n_subjects * np.sum((values.mean(axis=0) - grand) ** 2)
    ss_subject = n_times * np.sum((values.mean(axis=1) - grand) ** 2)
    ss_total = np.sum((values - grand) ** 2)
    ss_error = max(float(ss_total - ss_time - ss_subject), 0.0)
    df_time, df_error = n_times - 1, (n_subjects - 1) * (n_times - 1)
    if df_error <= 0 or ss_error <= 0:
        f_statistic = np.inf if ss_time > 0 else np.nan
        p_raw = 0.0 if ss_time > 0 else np.nan
    else:
        f_statistic = float((ss_time / df_time) / (ss_error / df_error))
        p_raw = float(stats.f.sf(f_statistic, df_time, df_error))
    sphericity = _sphericity(matrix)
    epsilon = sphericity["gg_epsilon"]
    p_corrected = float(stats.f.sf(f_statistic, df_time * epsilon, df_error * epsilon)) if np.isfinite(f_statistic) else p_raw
    return {
        "test": "Repeated-measures ANOVA", "statistic": f_statistic,
        "df1": float(df_time), "df2": float(df_error), "p": p_corrected,
        "p_uncorrected": p_raw, "df1_corrected": float(df_time * epsilon),
        "df2_corrected": float(df_error * epsilon),
        "partial_eta_squared": float(ss_time / (ss_time + ss_error)) if ss_time + ss_error > 0 else np.nan,
        **sphericity,
    }


def run_repeated_measures(
    df, participant_col, time_col, outcome_col, selected_times=None,
    method="auto", transformation="none", group_col=None, selected_groups=None,
    correction_method="holm", alpha=0.05,
):
    """Run a paired two-time test or a complete-case repeated-measures test."""
    if method not in {"auto", "parametric", "nonparametric"}:
        raise ValueError("method must be 'auto', 'parametric', or 'nonparametric'.")
    data, times, counts = prepare_long_data(
        df, participant_col, time_col, outcome_col, selected_times, group_col,
        selected_groups, transformation,
    )
    matrix, excluded = _complete_matrix(data, participant_col, time_col, outcome_col, times)
    excluded = counts["participants_selected"] - len(matrix)
    if len(matrix) < 2:
        raise RepeatedMeasuresValidationError("At least two participants with complete selected time points are required.")
    summaries = {
        time: {"n": int(matrix[time].count()), "mean": float(matrix[time].mean()),
               "sd": float(matrix[time].std()), "median": float(matrix[time].median()),
               "iqr": float(matrix[time].quantile(.75) - matrix[time].quantile(.25))}
        for time in times
    }
    if len(times) == 2:
        result = _paired_test(matrix[times[0]].to_numpy(), matrix[times[1]].to_numpy(), method, alpha)
        pairwise = []
    else:
        if np.ptp(matrix.to_numpy()) == 0:
            raise RepeatedMeasuresValidationError("The complete outcome values are constant; no repeated-measures test is defined.")
        anova = _rm_anova(matrix)
        try:
            friedman = stats.friedmanchisquare(*(matrix[time].to_numpy() for time in times))
            friedman_result = {"test": "Friedman test", "statistic": float(friedman.statistic), "p": float(friedman.pvalue)}
        except ValueError:
            friedman_result = {"test": "Friedman test", "statistic": np.nan, "p": np.nan}
        if method == "parametric":
            selected = "parametric"
        elif method == "nonparametric":
            selected = "nonparametric"
        else:
            # Multivariate normality is not established by marginal checks.  This
            # convenience rule checks every pairwise difference and is reported.
            difference_ps = [_shapiro(matrix[t2] - matrix[t1]) for t1, t2 in itertools.combinations(times, 2)]
            selected = "parametric" if all(np.isfinite(p) and p >= alpha for p in difference_ps) else "nonparametric"
        result = anova if selected == "parametric" else friedman_result
        result["selected_method"] = selected
        result["difference_normality_p"] = {
            f"{time1} vs {time2}": _shapiro(matrix[time2] - matrix[time1])
            for time1, time2 in itertools.combinations(times, 2)
        }
        pairwise = []
        for time1, time2 in itertools.combinations(times, 2):
            comparison = _paired_test(matrix[time1].to_numpy(), matrix[time2].to_numpy(), selected, alpha)
            comparison.update({"time1": time1, "time2": time2})
            pairwise.append(comparison)
        finite = [index for index, row in enumerate(pairwise) if np.isfinite(row["p"])]
        if len(finite) > 1:
            adjusted = multipletests([pairwise[index]["p"] for index in finite], method=correction_method)[1]
            for index, p_adjusted in zip(finite, adjusted):
                pairwise[index]["p_corrected"] = float(p_adjusted)
    return {
        "design": "paired/repeated measures", "outcome": outcome_col, "times": times,
        "result": result, "pairwise": pairwise, "summaries": summaries,
        "n_included": int(len(matrix)), "n_excluded_incomplete": excluded,
        "complete_data": matrix, "plot_data": data, "counts": counts,
        "correction_method": correction_method if len(pairwise) > 1 else None,
    }


def run_mixed_anova(
    df, participant_col, group_col, time_col, outcome_col, selected_times=None,
    selected_groups=None, transformation="none",
):
    """Run a complete-case split-plot ANOVA for one between-subject factor."""
    data, times, counts = prepare_long_data(
        df, participant_col, time_col, outcome_col, selected_times, group_col,
        selected_groups, transformation,
    )
    matrix, excluded = _complete_matrix(data, participant_col, time_col, outcome_col, times)
    excluded = counts["participants_selected"] - len(matrix)
    participant_groups = data.drop_duplicates(participant_col).set_index(participant_col)[group_col]
    participant_groups = participant_groups.reindex(matrix.index)
    groups = participant_groups.drop_duplicates().tolist()
    if len(groups) < 2:
        raise RepeatedMeasuresValidationError("At least two groups are required for a Group × Time analysis.")
    if len(matrix) <= len(groups):
        raise RepeatedMeasuresValidationError("Each design needs residual participants within groups.")
    values = matrix.to_numpy(dtype=float)
    n_subjects, n_times = values.shape
    grand = values.mean()
    subject_means = values.mean(axis=1)
    time_means = values.mean(axis=0)
    group_masks = [participant_groups.to_numpy() == group for group in groups]
    group_means = [values[mask].mean() for mask in group_masks]
    cell_means = [values[mask].mean(axis=0) for mask in group_masks]
    group_sizes = [int(mask.sum()) for mask in group_masks]
    if any(size < 2 for size in group_sizes):
        raise RepeatedMeasuresValidationError("Every group needs at least two complete participants.")

    ss_group = n_times * sum(size * (mean - grand) ** 2 for size, mean in zip(group_sizes, group_means))
    ss_between = n_times * np.sum((subject_means - grand) ** 2)
    ss_subject_group = max(float(ss_between - ss_group), 0.0)
    ss_time = n_subjects * np.sum((time_means - grand) ** 2)
    ss_interaction = sum(
        size * np.sum((cells - group_mean - time_means + grand) ** 2)
        for size, cells, group_mean in zip(group_sizes, cell_means, group_means)
    )
    ss_within = np.sum((values - subject_means[:, None]) ** 2)
    ss_error = max(float(ss_within - ss_time - ss_interaction), 0.0)
    df_group, df_subject_group = len(groups) - 1, n_subjects - len(groups)
    df_time = n_times - 1
    df_interaction = df_group * df_time
    df_error = df_subject_group * df_time

    def effect_row(name, ss_effect, df_effect, ss_denominator, df_denominator, epsilon=1.0):
        if ss_denominator <= 0:
            f_statistic = np.inf if ss_effect > 0 else np.nan
            p_value = 0.0 if ss_effect > 0 else np.nan
        else:
            f_statistic = float((ss_effect / df_effect) / (ss_denominator / df_denominator))
            p_value = float(stats.f.sf(f_statistic, df_effect * epsilon, df_denominator * epsilon))
        return {
            "effect": name, "F": f_statistic, "df1": float(df_effect * epsilon),
            "df2": float(df_denominator * epsilon), "p": p_value,
            "partial_eta_squared": float(ss_effect / (ss_effect + ss_denominator))
            if ss_effect + ss_denominator > 0 else np.nan,
        }

    sphericity = _sphericity(matrix)
    epsilon = sphericity["gg_epsilon"]
    effects = [
        effect_row("Group", ss_group, df_group, ss_subject_group, df_subject_group),
        effect_row("Time", ss_time, df_time, ss_error, df_error, epsilon),
        effect_row("Group × Time", ss_interaction, df_interaction, ss_error, df_error, epsilon),
    ]
    fitted = np.empty_like(values)
    for mask, cells in zip(group_masks, cell_means):
        fitted[mask] = cells
    residual_normality_p = _shapiro((values - fitted).ravel())
    try:
        subject_mean_levene_p = float(stats.levene(
            *(subject_means[mask] for mask in group_masks)
        ).pvalue)
    except ValueError:
        subject_mean_levene_p = np.nan
    posthoc = []
    interaction_p = effects[2]["p"]
    if np.isfinite(interaction_p) and interaction_p < 0.05:
        # Follow a detected interaction with simple effects.  All comparisons
        # belong to one multiplicity family for this outcome.
        for group, mask in zip(groups, group_masks):
            group_matrix = matrix.iloc[np.flatnonzero(mask)]
            for time1, time2 in itertools.combinations(times, 2):
                test = stats.ttest_rel(group_matrix[time2], group_matrix[time1])
                posthoc.append({
                    "contrast": f"{time1} vs {time2} within {group}",
                    "test": "Paired t-test", "statistic": float(test.statistic),
                    "p": float(test.pvalue),
                })
        for time in times:
            for group1, group2 in itertools.combinations(groups, 2):
                x = matrix.loc[participant_groups == group1, time]
                y = matrix.loc[participant_groups == group2, time]
                test = stats.ttest_ind(x, y, equal_var=False)
                posthoc.append({
                    "contrast": f"{group1} vs {group2} at {time}",
                    "test": "Welch's independent t-test", "statistic": float(test.statistic),
                    "p": float(test.pvalue),
                })
        finite = [index for index, row in enumerate(posthoc) if np.isfinite(row["p"])]
        if finite:
            adjusted = multipletests([posthoc[index]["p"] for index in finite], method="holm")[1]
            for index, p_adjusted in zip(finite, adjusted):
                posthoc[index]["p_corrected"] = float(p_adjusted)
    return {
        "design": "mixed-design ANOVA", "outcome": outcome_col, "times": times,
        "groups": groups, "effects": effects, "sphericity": sphericity,
        "assumptions": {
            "residual_shapiro_p": residual_normality_p,
            "subject_mean_levene_p": subject_mean_levene_p,
        },
        "n_included": n_subjects, "n_excluded_incomplete": excluded,
        "group_sizes": dict(zip(groups, group_sizes)), "complete_data": matrix,
        "plot_data": data[data[participant_col].isin(matrix.index)].copy(), "counts": counts,
        "posthoc": posthoc, "correction_method": "holm" if posthoc else None,
    }


def run_linear_mixed_model(
    df, participant_col, group_col, time_col, outcome_col, selected_times=None,
    selected_groups=None, transformation="none",
):
    """Fit a random-intercept LMM to all usable (including incomplete) observations."""
    data, times, counts = prepare_long_data(
        df, participant_col, time_col, outcome_col, selected_times, group_col,
        selected_groups, transformation,
    )
    if data[participant_col].nunique() < 3:
        raise RepeatedMeasuresValidationError("At least three participants are required for a mixed-effects model.")
    model_data = data.rename(columns={participant_col: "_participant", group_col: "_group", time_col: "_time", outcome_col: "_outcome"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = mixedlm(
                "_outcome ~ C(_group) * C(_time)",
                model_data,
                groups=model_data["_participant"],
            ).fit()
        except Exception as exc:
            raise RepeatedMeasuresValidationError(f"The linear mixed-effects model failed: {exc}") from exc
    terms = []
    confidence = fit.conf_int()
    for term in fit.fe_params.index:
        terms.append({
            "term": term, "estimate": float(fit.fe_params[term]), "se": float(fit.bse_fe[term]),
            "z": float(fit.fe_params[term] / fit.bse_fe[term]) if fit.bse_fe[term] > 0 else np.nan,
            "p": float(fit.pvalues[term]), "ci_low": float(confidence.loc[term, 0]),
            "ci_high": float(confidence.loc[term, 1]),
        })
    warning_messages = [str(item.message) for item in caught]
    if not fit.converged:
        warning_messages.append("The optimizer did not converge.")
    return {
        "design": "linear mixed-effects model", "outcome": outcome_col, "times": times,
        "terms": terms, "converged": bool(fit.converged), "warnings": warning_messages,
        "n_observations": int(len(model_data)), "n_participants": int(model_data["_participant"].nunique()),
        "plot_data": data, "counts": counts,
        "inference": "Wald z tests with asymptotic normal p-values; coefficients use treatment coding.",
    }
