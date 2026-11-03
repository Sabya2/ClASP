"""Shared statistical and plotting helpers."""

import itertools
import math
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, f_oneway, kruskal, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from srcs.utils_visualisation.plot_style import apply_graph_font






# =========================================================
# Palette helpers
# =========================================================


def get_plot_palette(selected_groups, mode="gray"):
    """
    Return a dictionary mapping each selected group to a color.
    This guarantees stable color assignment independent of grid settings.
    """
    if mode == "gray":
        colors = [
            "#B2BEB5",
            "#919191",
            "#899499",
            "#818589",
            "#71797E",
            "#708090",
            "#5F5F5F",
            "#404040",
        ]
    elif mode == "multicolor":
        colors = [
            "#3366CC",
            "#F8766D",
            "#404040",
            "#7CAE00",
            "#7A7979",
            "#00BFC4",
            "#619CFF",
            "#C77CFF",
            "#404040",
            "#FF61C3",
            "#3366CC",
            "#DC3912",
            "#FF9900",
            "#109618",
            "#990099",
            "#0099C6",
            "#DD4477",
            "#66AA00",
            "#B82E2E",
            "#316395",
            "#994499",
            "#22AA99",
        ]
    else:
        raise ValueError("mode must be 'multicolor' or 'gray'")

    return {group: colors[i % len(colors)] for i, group in enumerate(selected_groups)}


# =========================================================
# Plot style
# =========================================================

def set_paper_plot_style(show_grid=False):
    """
    Set global paper-style formatting.
    This function controls fonts, background, and grid only.
    It does NOT control category colors.
    """
    base_style = "whitegrid" if show_grid else "ticks"

    sns.set_theme(
        style=base_style,
        context="paper",
    )
    sns.set_context("paper", font_scale=0.8)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": show_grid,
    })
    apply_graph_font()

    if show_grid:
        plt.rcParams.update({
            "grid.color": "#616161",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
        })




# ---------------------------------------------------------
# Streamlit controls
# ---------------------------------------------------------


def _grid_shape(n, ncols=2):
    ncols = max(1, ncols)
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols

def get_figure_download(fig):
    """Serialize a Matplotlib figure as a PNG buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf


def stars(p):
    """Return significance stars for a p-value."""
    if pd.isna(p):
        return ""
    if p <= 1e-3:
        return "***"
    if p <= 1e-2:
        return "**"
    if p <= 5e-2:
        return "*"
    return ""


def p_to_star(p):
    """Return a compact significance marker for a p-value."""
    # print(p)
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return f"*"
    return f"ns-{p:.3f}"

def p_to_label(p):
    """Format a p-value as a readable significance label."""
    # print(p)
    if pd.isna(p):
        return "n/a"
    elif p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return "<0.01"
    elif p < 0.05:
        return "<0.05"
    else:
        return ""



def _candidate_group_columns(df, current_group=None, unique_threshold=12):
    categorical_candidates = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    low_card_num = [
        c for c in df.select_dtypes(include="number").columns
        if df[c].nunique(dropna=True) <= unique_threshold
    ]
    cols = list(dict.fromkeys(([current_group] if current_group in df.columns else []) + categorical_candidates + low_card_num))
    return cols if cols else df.columns.tolist()


def _corr_pair(x, y, method="pearson"):
    xy = pd.concat([pd.Series(x), pd.Series(y)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(xy) < 3:
        return np.nan, np.nan

    xv = xy.iloc[:, 0].to_numpy()
    yv = xy.iloc[:, 1].to_numpy()

    if method == "pearson":
        return stats.pearsonr(xv, yv)
    if method == "spearman":
        return stats.spearmanr(xv, yv)
    if method == "kendall":
        return stats.kendalltau(xv, yv)
    raise ValueError("Unsupported correlation method.")



def cohens_d(x, y):
    """Calculate Cohen's d using the pooled sample standard deviation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_sd




def build_correlation_matrices(df, method="spearman"):
    """Build pairwise correlation, p-value, and sample-size matrices."""
    df_num = df.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
    cols = df_num.columns.tolist()

    corr = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    pvals = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                corr.iloc[i, j] = 1.0
                pvals.iloc[i, j] = 0.0
            elif i > j:
                r, p = _corr_pair(df_num[a], df_num[b], method=method)
                corr.iloc[i, j] = corr.iloc[j, i] = r
                pvals.iloc[i, j] = pvals.iloc[j, i] = p

    annot = corr.copy().astype(object)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            p = pvals.iloc[i, j]
            annot.iloc[i, j] = "" if pd.isna(v) else f"{v:.2f}{stars(p)}"

    return corr, pvals, annot



def run_group_test(
    df,
    group_col,
    y_col,
    alpha=0.05,
    two_groups_only=False,
    force_parametric=False,
    correction_method='bonferroni',  # "holm", "sidak", etc.
    method="auto",
):
    """Compare independent groups under one explicit, shared selection policy.

    ``auto`` uses Shapiro-Wilk within every group: parametric tests are selected
    only when all groups have at least three observations and p >= alpha.
    Levene's test then selects equal-variance or Welch methods. ``parametric``
    and ``nonparametric`` are explicit overrides. ``force_parametric`` remains
    as a backwards-compatible alias for ``method='parametric'``.
    """
    if force_parametric:
        method = "parametric"
    if method not in {"auto", "parametric", "nonparametric"}:
        raise ValueError("method must be 'auto', 'parametric', or 'nonparametric'.")
    df_plot = df[[group_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    groups_order = (
        df_plot[group_col].dropna().value_counts(sort=False).index
    ).tolist()
    if len(groups_order) < 2:
        return None


    groups = {g: df_plot.loc[df_plot[group_col] == g, y_col].astype(float).to_numpy() for g in groups_order}
    normality = {
        g: (float(shapiro(v).pvalue) if 3 <= len(v) <= 5000 and np.ptp(v) > 0 else np.nan)
        for g, v in groups.items()
    }
    all_normal = all(np.isfinite(p) and p >= alpha for p in normality.values())
    selected_method = method if method != "auto" else ("parametric" if all_normal else "nonparametric")
    all_values = np.concatenate(list(groups.values()))
    if np.ptp(all_values) == 0:
        return {
            "groups": list(groups_order), "normality_p": normality,
            "all_normal": False, "levene_p": np.nan,
            "overall_test": "n/a (all values constant)", "overall_p": np.nan,
            "overall_p_label": "n/a", "overall_stars": "n/a", "pairwise": [],
            "correction_method": None, "groups_data": groups,
            "use_dunn_test": False, "selected_method": selected_method,
            "selection_policy": method, "design": "independent groups",
            "data_warning": "All analysed values are identical; no group test is defined.",
        }
    if selected_method == "parametric":
        try:
            p_levene = float(levene(*groups.values()).pvalue)
        except ValueError:
            p_levene = np.nan
    else:
        p_levene = np.nan
    equal_variance = bool(np.isfinite(p_levene) and p_levene >= alpha)

    arrays = list(groups.values())
    if any(len(v) < 2 for v in arrays):
        overall_test, overall_p = "n/a", np.nan
    elif len(arrays) == 2:
        if selected_method == "parametric":
            overall_test = "Student's independent t-test" if equal_variance else "Welch's independent t-test"
            overall_p = float(ttest_ind(*arrays, equal_var=equal_variance).pvalue)
        else:
            overall_test = "Mann–Whitney U"
            overall_p = float(mannwhitneyu(*arrays, alternative="two-sided").pvalue)
    elif selected_method == "parametric":
        overall_test = "One-way ANOVA" if equal_variance else "Welch's ANOVA"
        overall_p = float(f_oneway(*arrays, equal_var=equal_variance).pvalue)
    else:
        overall_test = "Kruskal–Wallis"
        try:
            overall_p = float(kruskal(*arrays).pvalue)
        except ValueError:
            overall_p = np.nan

    pairs = list(itertools.combinations(groups_order, 2))
    pairwise_results = []
    use_dunn = len(groups_order) > 2 and selected_method == "nonparametric"
    if use_dunn:
        try:
            raw_matrix = sp.posthoc_dunn(df_plot, val_col=y_col, group_col=group_col, p_adjust=None)
            adjusted_matrix = sp.posthoc_dunn(
                df_plot, val_col=y_col, group_col=group_col, p_adjust=correction_method
            ) if correction_method else raw_matrix
        except Exception as exc:
            raise RuntimeError("Dunn post-hoc calculation failed; no fallback method was substituted.") from exc
        for g1, g2 in pairs:
            p_raw = float(raw_matrix.loc[g1, g2])
            p_adjusted = float(adjusted_matrix.loc[g1, g2])
            pairwise_results.append({
                "group1": g1, "group2": g2, "test": "Dunn's test",
                "p": p_raw, "p_corrected": p_adjusted,
                "p_label": p_to_label(p_raw), "stars": p_to_star(p_raw),
                "p_corrected_label": p_to_label(p_adjusted),
                "stars_corrected": p_to_star(p_adjusted),
                "cohens_d": cohens_d(groups[g1], groups[g2]),
            })
    else:
        for g1, g2 in pairs:
            x, y = groups[g1], groups[g2]
            if len(x) < 2 or len(y) < 2:
                p_pair, pair_test = np.nan, "n/a"
            elif selected_method == "parametric":
                pair_equal = bool(np.isfinite(levene(x, y).pvalue) and levene(x, y).pvalue >= alpha)
                p_pair = float(ttest_ind(x, y, equal_var=pair_equal).pvalue)
                pair_test = "Student's independent t-test" if pair_equal else "Welch's independent t-test"
            else:
                p_pair = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
                pair_test = "Mann–Whitney U"
            pairwise_results.append({
                "group1": g1, "group2": g2, "test": pair_test, "p": p_pair,
                "p_label": p_to_label(p_pair), "stars": p_to_star(p_pair),
                "cohens_d": cohens_d(x, y),
            })
        finite_indices = [i for i, row in enumerate(pairwise_results) if np.isfinite(row["p"])]
        if correction_method and len(finite_indices) > 1:
            adjusted = multipletests(
                [pairwise_results[i]["p"] for i in finite_indices], method=correction_method
            )[1]
            for index, p_adjusted in zip(finite_indices, adjusted):
                pairwise_results[index]["p_corrected"] = float(p_adjusted)
                pairwise_results[index]["p_corrected_label"] = p_to_label(p_adjusted)
                pairwise_results[index]["stars_corrected"] = p_to_star(p_adjusted)

    return {
        "groups": list(groups_order),
        "normality_p": normality,
        "all_normal": all_normal,
        "levene_p": p_levene,
        "overall_test": overall_test,
        "overall_p": overall_p,
        "overall_p_label": p_to_label(overall_p),
        "overall_stars": p_to_star(overall_p),
        "pairwise": pairwise_results,
        "correction_method": correction_method if correction_method and len(pairwise_results) > 1 else None,
        "groups_data": groups,
        "use_dunn_test": use_dunn,
        "selected_method": selected_method,
        "selection_policy": method,
        "design": "independent groups",
    }
