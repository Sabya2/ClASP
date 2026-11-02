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
    # fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    # img.save(buf, format="png")
    # buf.seek(0)
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
):
    """
    Run group comparison for 2 or more groups.
    - If two_groups_only=True, always use t‑test / Mann–Whitney.
    - If normal & equal_variance, use ANOVA / t‑test; otherwise use Kruskal / Mann–Whitney.
    - For 3+ groups with non-parametric data: uses Dunn's test with correction (not Mann-Whitney pairwise).
    """
    df_plot = df[[group_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    groups_order = (
        df_plot[group_col].dropna().value_counts(sort=False).index
    ).tolist()
    if len(groups_order) < 2:
        return None


    # collect group‑wise values
    groups = {}
    for g in groups_order:
        vals = df_plot[df_plot[group_col] == g][y_col].astype(float).dropna().values
        groups[g] = vals


    # --- normality & variance --- (ignored if two_groups_only)
    all_normal = True
    normality = {}
    use_kruskal = False  # track if we used Kruskal-Wallis
    if not two_groups_only and not force_parametric:
        for g, vals in groups.items():
            if len(vals) >= 3:
                try:
                    _, p_norm = shapiro(vals)
                except Exception:
                    p_norm = np.nan
            else:
                p_norm = np.nan
            normality[g] = p_norm
            if np.isnan(p_norm) or p_norm < alpha:
                all_normal = False


        if len(groups_order) > 2:
            try:
                _, p_levene = levene(*[groups[g] for g in groups_order])
            except Exception:
                p_levene = np.nan
        else:
            p_levene = np.nan
    else:
        all_normal = True
        p_levene = np.nan


    # --- overall test: ANOVA vs Kruskal or t‑test vs Mann–Whitney ---
    if two_groups_only or len(groups_order) == 2:
        g1, g2 = groups_order[0], groups_order[1]
        x, y = groups[g1], groups[g2]


        if len(x) < 2 or len(y) < 2:
            overall_test = "n/a"
            overall_p = np.nan
        else:
            if all_normal and not force_parametric:
                equal_var = bool(p_levene >= alpha) if not np.isnan(p_levene) else False
                _, overall_p = ttest_ind(x, y, equal_var=equal_var, nan_policy="omit")
                overall_test = "t-test"
            else:
                _, overall_p = mannwhitneyu(x, y, alternative="two-sided")
                overall_test = "Mann–Whitney"


    else:  # 3+ groups
        if all_normal and not force_parametric:
            try:
                _, overall_p = f_oneway(*[groups[g] for g in groups_order])
                overall_test = "ANOVA"
                use_kruskal = False
            except Exception:
                overall_test = "ANOVA"
                overall_p = np.nan
                use_kruskal = False
        else:
            try:
                _, overall_p = kruskal(*[groups[g] for g in groups_order])
                overall_test = "Kruskal–Wallis"
                use_kruskal = True  # <-- flag to use Dunn's test
            except Exception:
                overall_test = "Kruskal–Wallis"
                overall_p = np.nan
                use_kruskal = True


    # --- pairwise tests & effect sizes ---
    pairwise_results = []
    if len(groups_order) >= 2:

        # For 3+ groups with Kruskal-Wallis: use Dunn's test
        if len(groups_order) > 2 and use_kruskal and not force_parametric:
            # Prepare data for scikit-posthocs
            df_for_dunn = df_plot.copy()

            try:
                # Dunn's test with p-value adjustment
                dunn_result = sp.posthoc_dunn(
                    df_for_dunn,
                    val_col=y_col,
                    group_col=group_col,
                    p_adjust=correction_method
                )

                # Extract pairwise results from Dunn's test matrix
                for i, g1 in enumerate(groups_order):
                    for j, g2 in groups_order[i+1:],:
                        if g1 != g2:
                            p_raw = dunn_result.loc[g1, g2]
                            p_corrected = p_raw  # Dunn already has correction applied

                            # Calculate Cohen's d for effect size
                            x = groups[g1]
                            y = groups[g2]
                            d = cohens_d(x, y)

                            pairwise_results.append({
                                "group1": g1,
                                "group2": g2,
                                "test": "Dunn's Test",
                                "p": p_raw,
                                "p_corrected": p_corrected,
                                "p_label": p_to_label(p_corrected),
                                "stars": p_to_star(p_corrected),
                                "stars_corrected": p_to_star(p_corrected),
                                "p_corrected_label": p_to_label(p_corrected),
                                "cohens_d": d,
                            })

            except Exception as e:
                # Fallback to Mann-Whitney if Dunn fails
                pairs = list(itertools.combinations(groups_order, 2))
                for g1, g2 in pairs:
                    x = groups[g1]
                    y = groups[g2]
                    if len(x) < 2 or len(y) < 2:
                        p_pair = np.nan
                        pair_test = "n/a"
                    else:
                        _, p_pair = mannwhitneyu(x, y, alternative="two-sided")
                        pair_test = "Mann–Whitney"
                    d = cohens_d(x, y)
                    pairwise_results.append({
                        "group1": g1,
                        "group2": g2,
                        "test": pair_test,
                        "p": p_pair,
                        "p_label": p_to_label(p_pair),
                        "stars": p_to_star(p_pair),
                        "cohens_d": d,
                    })

                # Apply correction to fallback Mann-Whitney
                if correction_method and len(pairwise_results) > 1:
                    p_values = [res["p"] for res in pairwise_results]
                    try:
                        corrected = multipletests(p_values, alpha=alpha, method=correction_method)
                        corrected_p = corrected[1]
                        for i, res in enumerate(pairwise_results):
                            res["p_corrected"] = corrected_p[i]
                            res["p_corrected_label"] = p_to_label(corrected_p[i])
                            res["stars_corrected"] = p_to_star(corrected_p[i])
                    except Exception:
                        pass

        # For 2 groups OR parametric: use standard pairwise tests
        else:
            pairs = list(itertools.combinations(groups_order, 2))
            for g1, g2 in pairs:
                x = groups[g1]
                y = groups[g2]
                if len(x) < 2 or len(y) < 2:
                    p_pair = np.nan
                    pair_test = "n/a"
                else:
                    if all_normal and not force_parametric and not two_groups_only:
                        equal_var = bool(p_levene >= alpha) if not np.isnan(p_levene) else False
                        _, p_pair = ttest_ind(x, y, equal_var=equal_var, nan_policy="omit")
                        pair_test = "t‑test"
                    else:
                        _, p_pair = mannwhitneyu(x, y, alternative="two-sided")
                        pair_test = "Mann–Whitney"
                d = cohens_d(x, y)
                pairwise_results.append({
                    "group1": g1,
                    "group2": g2,
                    "test": pair_test,
                    "p": p_pair,
                    "p_label": p_to_label(p_pair),
                    "stars": p_to_star(p_pair),
                    "cohens_d": d,
                })


            # --- multiple testing correction --- (only for non-Dunn cases)
            if correction_method and len(pairwise_results) > 1 and not use_kruskal:
                p_values = [res["p"] for res in pairwise_results]
                try:
                    corrected = multipletests(p_values, alpha=alpha, method=correction_method)
                    corrected_p = corrected[1]
                    for i, res in enumerate(pairwise_results):
                        res["p_corrected"] = corrected_p[i]
                        res["p_corrected_label"] = p_to_label(corrected_p[i])
                        res["stars_corrected"] = p_to_star(corrected_p[i])
                except Exception:
                    pass


    return {
        "groups": list(groups_order),
        "normality_p": normality if not two_groups_only and not force_parametric else None,
        "all_normal": all_normal,
        "levene_p": p_levene if not two_groups_only and not force_parametric else None,
        "overall_test": overall_test,
        "overall_p": overall_p,
        "overall_p_label": p_to_label(overall_p),
        "overall_stars": p_to_star(overall_p),
        "pairwise": pairwise_results,
        "correction_method": correction_method if correction_method and len(pairwise_results) > 1 else None,
        "groups_data": groups,
        "use_dunn_test": len(groups_order) > 2 and use_kruskal and not force_parametric,  # <-- new flag
    }
