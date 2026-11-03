"""Bootstrap mediation analysis and visualization."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import streamlit as st

from srcs.utils_visualisation.common import get_figure_download
from srcs.utils_visualisation.plot_style import apply_graph_font


@st.cache_data
def run_mediation_cached(df: pd.DataFrame, x: str, y: str, m: str,
                        covariates: list = None, alpha: float = 0.05,
                        n_boot: int = 1000, seed: int = 42) -> dict:
    """Cached mediation wrapper with optional covariates"""
    np.random.seed(seed)

    # Build formula strings
    x_var = x
    m_var = m
    y_var = y

    if covariates is None:
        covariates = []

    # Path a: X → M (+ covariates)
    formula_a = f"{m_var} ~ {x_var}"
    if covariates:
        formula_a += " + " + " + ".join(covariates)

    # Path b: M → Y | X (+ covariates)
    formula_b = f"{y_var} ~ {m_var} + {x_var}"
    if covariates:
        formula_b += " + " + " + ".join(covariates)

    # Total effect: X → Y (+ covariates)
    formula_c = f"{y_var} ~ {x_var}"
    if covariates:
        formula_c += " + " + " + ".join(covariates)

    try:
        # Fit models
        mod_a = sm.OLS.from_formula(formula_a, df).fit()
        mod_b = sm.OLS.from_formula(formula_b, df).fit()
        mod_c = sm.OLS.from_formula(formula_c, df).fit()

        # Extract coefficients
        beta_a = mod_a.params[x_var]
        beta_b = mod_b.params[m_var]
        beta_c = mod_c.params[x_var]
        beta_cprime = mod_b.params[x_var]

        # Indirect effect
        indirect = beta_a * beta_b

        # Bootstrap simulation (simple normal approximation)
        indirect_se = np.sqrt(beta_a**2 * mod_b.bse[m_var]**2 +
                             beta_b**2 * mod_a.bse[x_var]**2)
        indirect_dist = np.random.normal(indirect, indirect_se, n_boot)

        ci_lower, ci_upper = np.percentile(indirect_dist, [2.5, 97.5])

        results = {
            'beta_a': beta_a, 'p_a': mod_a.pvalues[x_var],
            'beta_b': beta_b, 'p_b': mod_b.pvalues[m_var],
            'beta_c': beta_c, 'p_c': mod_c.pvalues[x_var],
            'beta_cprime': beta_cprime, 'p_cprime': mod_b.pvalues[x_var],
            'indirect': indirect, 'indirect_ci': (ci_lower, ci_upper),
            'indirect_dist': indirect_dist,
            'n_obs': len(df),
            'covariates_used': covariates,
            'r2_a': mod_a.rsquared,
            'r2_b': mod_b.rsquared,
            'r2_c': mod_c.rsquared
        }

        return results

    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def plot_mediation_model_streamlit(indirect_dist, x_name, y_name, m_name,
                                  coef_values, p_values, significance_info,
                                  covariates_used=None):
    """Updated plot with covariate info"""
    plt.style.use('default')
    apply_graph_font()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')

    # Left: Mediation Diagram
    ax = axes[0]
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor('white')

    # Covariate note
    if covariates_used:
        ax.text(8, 0.3, f"Models controlled for: {', '.join(covariates_used)}",
                fontsize=10, ha='center', style='italic', color='gray')

    # Total effect (top)
    ax.text(2.5, 7, f"{x_name}", fontsize=13, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=1.2", fc="lightblue", ec="blue"))
    ax.text(12, 7, f"{y_name}", fontsize=13, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=1.2", fc="lightcoral", ec="red"))
    ax.annotate("", xy=(4.3, 7), xytext=(10.1, 7),
                arrowprops=dict(arrowstyle="<-", lw=2.5, ec="black"))
    ax.text(6.7, 7.2, f"Total effect - Path c:\nβ₁ = {coef_values['β1']:.3f}\n{significance_info['β1']}",
            fontsize=12, ha='center', fontweight='bold')



    # Mediation paths (bottom) Bottom section: Mediation triangle

    ax.text(2.5, 3.4, f"{x_name}", fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=1.2", fc="lightblue", ec="blue"))
    ax.text(12, 3.4, f"{y_name}", fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=1.2", fc="lightcoral", ec="red"))
    ax.text(7, 5, f"{m_name}", fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=1.2", fc="lightgreen", ec="green"))
    # ax.text(6.7, 6, f"Mediation pathway - \n{mediation} with {m_extent:3e} proportion", fontsize=12, ha='center', va='center')

    # Arrows
    ax.annotate("", xy=(3.5, 3.9), xytext=(6.5, 4.5),
                arrowprops=dict(arrowstyle="<-", lw=2.5, ec="black"))
    ax.annotate("", xy=(10.8, 3.9), xytext=(7.5, 4.5),
                arrowprops=dict(arrowstyle="->", lw=2.5, ec="black"))
    ax.annotate("", xy=(4.2, 3.4), xytext=(10.1, 3.4),
                arrowprops=dict(arrowstyle="<-", lw=2, linestyle="dashed", ec="black"))

    # Path labels
    ax.text(3.4, 4.2, f"Path a:\nβ₂= {coef_values['β2']:.3f}\n{significance_info['β2']}",
            fontsize=12, ha='center', fontweight='bold')
    ax.text(10.4, 4.2, f"Path b:\nβ₃= {coef_values['β3']:.3f}\n{significance_info['β3']}",
            fontsize=12, ha='center', fontweight='bold')
    ax.text(6.7, 2.2, f"Path c':\nβ₄= {coef_values['β4']:.3f}\n{significance_info['β4']}",
            fontsize=12, ha='center', fontweight='bold')

    # Equations
    ax.text(7, 0.6,
            f"y = β₀ + β₁·{x_name} + e (Total - path c)\n"
            f"m = β₀ + β₂·{x_name} + e\n"
            f"Y = β₀ + β₄·{x_name} + β₃·{m_name} + e (Direct+Indirect)\n\n"
            f"Indirect = β₂ × β₃ = {coef_values['indirect']:.3f} {significance_info['indirect']}\n",
            # f"Mediation pathway - \n{mediation} with {m_extent:3e} proportion",
            fontsize=11, ha='center', va='center', style='italic')


    # Right: Bootstrap distribution
    ax2 = axes[1]
    ax2.set_facecolor('white')
    sns.histplot(indirect_dist, kde=True, color='skyblue', stat='density', bins=50, ax=ax2)

    ci_lower = np.percentile(indirect_dist, 2.5)
    ci_upper = np.percentile(indirect_dist, 97.5)

    ax2.fill_betweenx([0, ax2.get_ylim()[1]], ci_lower, ci_upper,
                     color='gray', alpha=0.4, label='95% CI')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label="Null (0)")

    ax2.set_title(f"Indirect Effect Distribution (a × b)\n"
                 f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel("Indirect Effect", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.legend()

    plt.tight_layout()
    return fig



def interpret_mediation_results(results, x_var, m_var, y_var, alpha=0.05):
    """Summarize mediation effects in plain language."""
    indirect_sig = 0 not in results["indirect_ci"]
    direct_sig = results["p_cprime"] <= alpha
    total_sig = results["p_c"] <= alpha

    indirect = results["indirect"]
    direct = results["beta_cprime"]
    total = results["beta_c"]

    direction_indirect = "positive" if indirect > 0 else "negative"
    direction_direct = "positive" if direct > 0 else "negative"
    direction_total = "positive" if total > 0 else "negative"

    if indirect_sig and direct_sig:
        mediation_type = "partial mediation"
        # Both indirect and direct effects are present:
        # X affects Y through M, but there is also a remaining direct X -> Y effect.

    elif indirect_sig and not direct_sig:
        mediation_type = "indirect-only mediation"
        # The effect of X on Y appears to operate mainly through M:
        # the mediated pathway is supported, but the remaining direct effect is not.

    elif not indirect_sig and direct_sig:
        mediation_type = "direct effect without clear mediation"
        # X still has a direct association with Y,
        # but there is no clear evidence that M explains that relationship.

    else:
        mediation_type = "no clear evidence of mediation"
        # Neither the indirect pathway nor the direct pathway is clearly supported,
        # so this model does not provide strong evidence for mediation.

    text = (
        f"**{x_var} → {m_var} → {y_var}** | N={results['n_obs']} | Covariates: {', '.join(results['covariates_used']) if results['covariates_used'] else 'None'}\n"

        f"- **Indirect effect**: {direction_indirect} ({indirect:.4f}), and "
        f"{'statistically supported' if indirect_sig else 'not clearly supported'}.\n"
        f"- **Direct effect**: {direction_direct} ({direct:.4f}), and "
        f"{'significant' if direct_sig else 'not significant'}.\n"
        f"- **Total effect**: {direction_total} ({total:.4f}), and "
        f"{'significant' if total_sig else 'not significant'}.\n"
        f"- **Interpretation**: Overall, this pattern suggests {mediation_type}."
    )
    return text



def render_mediation_ascii_symbols():
    """Render a compact legend for mediation-path symbols."""
    diagram = r"""
                    a       b
                X ────▶ M ────▶ Y
                 \____________/
                        c′
    Total effect: c, Indirect effect: a×b, Direct effect: c′
            """
    st.markdown(f"```\n{diagram}\n```")


def render_mediation_page(df, continuous_cols, categorical_cols, group_col=None):
    """Mediation with covariates support"""
    with st.expander("How this page works", expanded=False):
            st.markdown(
            """
                This page tests mediation models of the form **X → M → Y**, where:

                - **X** is the predictor
                - **M** is the mediator
                - **Y** is the outcome

                The app estimates three linked regression models for each selected combination:
                - **Path a**: `M ~ X (+ covariates)`
                - **Path b / direct model**: `Y ~ M + X (+ covariates)`
                - **Total effect**: `Y ~ X (+ covariates)`

                The implementation uses ordinary least squares regression with formula-based model fitting from Statsmodels.

                **What the reported effects mean**
                - **Total effect (c)**: the overall association between X and Y.
                - **Indirect effect (a×b)**: the part of the X→Y relationship that operates through M.
                - **Direct effect (c')**: the association between X and Y after accounting for M.

                **Covariates**
                - You can add continuous and categorical covariates as adjustment variables.
                - These covariates are included in all fitted mediation models for the selected combination.

                **Two analysis modes**
                - **Run All Mediations** evaluates every valid X×M×Y combination you selected.
                - **Single Mediation Plots (Quick)** produces detailed plots and interpretation text for selected combinations.

                **How significance is decided here**
                - Path coefficients use model p-values from the fitted regressions.
                - The indirect effect is treated as supported when its bootstrap-style 95% interval does **not** include 0.
                - In the grid mode, a result is flagged as significant when path a, path b, and the indirect interval all support mediation.

                **Outputs on this page**
                - A full results table across all tested combinations.
                - A significant-only subset when any supported mediations are found.
                - Heatmaps of total, indirect, and direct effects, grouped by predictor.
                - Detailed single-mediation diagrams with coefficient labels, indirect-effect distribution, and a text interpretation.

                **Practical notes**
                - Each model uses complete-case data for X, M, Y, and all selected covariates.
                - Combinations with too few complete observations are skipped.
                - Large numbers of combinations can take longer because each model is fit separately and bootstrapped summaries are generated for each run.
                            """
                        )
    col1, col2 = st.columns([3, 2])
    with col1:
        st.header("🧬 Mediation Analysis")
        st.write("**Total**: c (X→Y) | **Indirect**: a×b (X→M→Y) | **Direct**: c'|M")
    with col2:
        render_mediation_ascii_symbols()

    if df is None or len(continuous_cols) < 3:
        st.warning("Need ≥3 continuous variables")
        return

    # Multi-select controls
    col1, col2, col3 = st.columns(3)
    x_vars = col1.multiselect("X (Predictors)", continuous_cols, max_selections=10)
    m_vars = col2.multiselect("M (Mediators)", [c for c in continuous_cols if c not in x_vars], max_selections=10)
    y_vars = col3.multiselect("Y (Outcomes)", [c for c in continuous_cols if c not in x_vars + m_vars], max_selections=10)


    with st.expander("🛡️ Covariates (Controls)", expanded=False):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            cov_continuous = st.multiselect(
                "Continuous covariates",
                [c for c in continuous_cols if c not in x_vars + m_vars + y_vars],
                max_selections=5
            )

        with col_c2:
            cov_categorical = st.multiselect(
                "Categorical covariates",
                [c for c in categorical_cols if c not in x_vars + m_vars + y_vars],
                max_selections=5
            )

        covariates = cov_continuous + cov_categorical

    # Analysis controls
    col1, col2, col3 = st.columns(3)
    with col1:
        alpha = st.slider("α", 0.01, 0.10, 0.05)
    with col2:
        n_boot = st.slider("Bootstrap samples", 500, 5000, 1000, 100)
    with col3:
        n_combos = len(x_vars) * len(m_vars) * len(y_vars)
        col_count1, col_count2 = st.columns([1, 2])
        col_count1.metric("Combinations", n_combos)
        col_count2.info(f"X×M×Y = {len(x_vars)}×{len(m_vars)}×{len(y_vars)}")

    # GRID ANALYSIS
    if st.button("🔬 Run All Mediations", type="primary") and n_combos <= 50:
        st.info(f"Computing {n_combos} mediation models with {len(covariates)} covariates...")
        all_results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (x_var, m_var, y_var) in enumerate(
                                                [(x, m, y) for x in x_vars
                                                 for m in m_vars for y in y_vars]
                                                ):
            status_text.text(f"Processing {x_var} → {m_var} → {y_var}")

            # Clean data for this combination
            cols_needed = [x_var, m_var, y_var] + covariates
            df_combo = df[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()
            df_combo = df_combo.apply(pd.to_numeric, errors='coerce').dropna()
            # print(f"Combo {i+1}/{n_combos}: {x_var} → {m_var} → {y_var} | N={len(df_combo)}")

            if len(df_combo) >= 15:  # Need more obs with covariates
                results = run_mediation_cached(df_combo, x_var, y_var, m_var,
                                             covariates, alpha, n_boot)
                print(results)
                # print(f"Combo {i+1}/{n_combos}: {x_var} → {m_var} → {y_var} | N={len(df_combo)}")
                if results:
                    results.update({
                        'x': x_var, 'm': m_var, 'y': y_var,
                        'n_samples': len(df_combo),
                        'significant': (
                            (results['p_a'] <= alpha) and
                            (results['p_b'] <= alpha) and
                            (0 not in results['indirect_ci'])
                        )
                    })
                    # print(f"Combo {i+1}/{n_combos}: {x_var} → {m_var} → {y_var} | N={len(df_combo)}")
                    all_results.append(results)

            progress_bar.progress((i+1) / n_combos)

        progress_bar.empty()
        status_text.empty()

        if all_results:
            results_df = pd.DataFrame(all_results)

            # Results table with covariate info
            display_cols = ['x', 'm', 'y', 'indirect', 'beta_cprime', 'n_samples', 'r2_a', 'r2_b']
            with st.expander("📋 All Results", expanded=True):
                st.markdown(f"**Models controlled for: {', '.join(covariates) if covariates else 'None'}**")
                st.dataframe(results_df[display_cols].round(4), width="stretch")

            # Significant results
            sig_results = results_df[results_df['significant']]
            if not sig_results.empty:
                st.success(f"🎉 {len(sig_results)} significant mediations!")
                with st.expander("⭐ Significant Only"):
                    st.dataframe(sig_results[['x','m','y','indirect','beta_cprime']].round(4))
            else:
                st.warning("No significant mediations found")

            # # Heatmaps
            st.subheader("Effect Heatmaps by X")

            for x_val in results_df['x'].unique():
                st.markdown(f"### Predictor: `{x_val}`")
                df_x = results_df[results_df['x'] == x_val].copy()
                df_x['sig_star'] = np.where(df_x['significant'], '.', '')

                col1, col2, col3 = st.columns(3)

                with col1:
                    pivot_total = df_x.pivot_table(
                        values='beta_c', index='m', columns='y',
                        aggfunc='first', fill_value=0
                    )
                    star_total = df_x.pivot_table(
                        values='sig_star', index='m', columns='y',
                        aggfunc='first', fill_value=''
                    )

                    fig_total = px.imshow(
                        pivot_total,
                        title=f"Total Effect (c)<br>{x_val} → Y",
                        color_continuous_scale="viridis",
                        aspect="auto"
                    )
                    fig_total.update_traces(
                        text=star_total.values,
                        texttemplate="%{text}",
                        textfont={"size": 18, "color": "black"}
                    )
                    fig_total.update_layout(height=350)
                    st.plotly_chart(fig_total, width="stretch")

                with col2:
                    pivot_indirect = df_x.pivot_table(
                        values='indirect', index='m', columns='y',
                        aggfunc='first', fill_value=0
                    )
                    star_indirect = df_x.pivot_table(
                        values='sig_star', index='m', columns='y',
                        aggfunc='first', fill_value=''
                    )

                    fig_indirect = px.imshow(
                        pivot_indirect,
                        title=f"Indirect Effect (a×b)<br>{x_val} → M → Y",
                        color_continuous_scale="viridis",
                        aspect="auto"
                    )
                    fig_indirect.update_traces(
                        text=star_indirect.values,
                        texttemplate="%{text}",
                        textfont={"size": 18, "color": "black"}
                    )
                    fig_indirect.update_layout(height=350)
                    st.plotly_chart(fig_indirect, width="stretch")

                with col3:
                    pivot_direct = df_x.pivot_table(
                        values='beta_cprime', index='m', columns='y',
                        aggfunc='first', fill_value=0
                    )
                    star_direct = df_x.pivot_table(
                        values='sig_star', index='m', columns='y',
                        aggfunc='first', fill_value=''
                    )

                    fig_direct = px.imshow(
                        pivot_direct,
                        title=f"Direct Effect (c')<br>{x_val} → Y | M",
                        color_continuous_scale="viridis",
                        aspect="auto"
                    )
                    fig_direct.update_traces(
                        text=star_direct.values,
                        texttemplate="%{text}",
                        textfont={"size": 18, "color": "black"}
                    )
                    fig_direct.update_layout(height=350)
                    st.plotly_chart(fig_direct, width="stretch")

            st.caption("* = significant mediation pair")




            # Downloads
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv = results_df.to_csv(index=False)
                st.download_button("📊 Full Results CSV", csv, "mediation_results.csv")
            with col_dl2:
                sig_csv = sig_results.to_csv(index=False) if not sig_results.empty else ""
                st.download_button("⭐ Significant CSV", sig_csv, "mediation_significant.csv")

        else:
            st.error("No valid data for analysis (check sample size)")


    if st.checkbox("⚡ Single Mediation Plots (Quick)"):
        col_s1, col_s2, col_s3 = st.columns(3)

        x_list = col_s1.multiselect(
            "X",
            continuous_cols,
            default=[],
            key="med_x"
        )
        m_list = col_s2.multiselect(
            "M",
            continuous_cols,
            default=[],
            key="med_m"
        )
        y_list = col_s3.multiselect(
            "Y",
            continuous_cols,
            default=[],
            key="med_y"
        )

        single_cov = []

        if "mediation_results" not in st.session_state:
            st.session_state.mediation_results = []

        if "mediation_warnings" not in st.session_state:
            st.session_state.mediation_warnings = []

        if st.button("Run Single", type="secondary"):
            saved_results = []
            saved_warnings = []

            combinations = [
                (x_var, m_var, y_var)
                for x_var, m_var, y_var in product(x_list, m_list, y_list)
                if len({x_var, m_var, y_var}) == 3
            ]

            if not combinations:
                saved_warnings.append("No valid X-M-Y combinations selected.")
            else:
                for i, (x_var, m_var, y_var) in enumerate(combinations, start=1):
                    cols_needed = [x_var, m_var, y_var] + single_cov
                    df_single = df[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()
                    df_single = df_single.apply(pd.to_numeric, errors="coerce").dropna()

                    if len(df_single) < 15:
                        saved_warnings.append(
                            f"Skipping {x_var} → {m_var} → {y_var}: fewer than 15 complete rows."
                        )
                        continue

                    results = run_mediation_cached(
                        df_single, x_var, y_var, m_var, single_cov, alpha, n_boot
                    )

                    if not results:
                        saved_warnings.append(
                            f"Skipping {x_var} → {m_var} → {y_var}: mediation model failed."
                        )
                        continue

                    saved_results.append({
                        "i": i,
                        "x_var": x_var,
                        "m_var": m_var,
                        "y_var": y_var,
                        "results": results,
                    })

            st.session_state.mediation_results = saved_results
            st.session_state.mediation_warnings = saved_warnings

        for warning_text in st.session_state.mediation_warnings:
            st.warning(warning_text)

        for item in st.session_state.mediation_results:
            i = item["i"]
            x_var = item["x_var"]
            m_var = item["m_var"]
            y_var = item["y_var"]
            results = item["results"]

            st.markdown(f"### Mediation {i}: {x_var} → {m_var} → {y_var}")

            fig = plot_mediation_model_streamlit(
                results["indirect_dist"],
                x_var,
                y_var,
                m_var,
                {
                    "β1": results["beta_c"],
                    "β2": results["beta_a"],
                    "β3": results["beta_b"],
                    "β4": results["beta_cprime"],
                    "indirect": results["indirect"],
                },
                results,
                {
                    "β1": "***" if results["p_c"] <= alpha else "ns",
                    "β2": "***" if results["p_a"] <= alpha else "ns",
                    "β3": "***" if results["p_b"] <= alpha else "ns",
                    "β4": "***" if results["p_cprime"] <= alpha else "ns",
                    "indirect": "***" if 0 not in results["indirect_ci"] else "ns",
                },
                covariates_used=single_cov,
            )

            st.pyplot(fig)

            col_dl, col_txt = st.columns([1, 2], gap="medium")

            with col_dl:
                buf = get_figure_download(fig)
                plot_name = f"{x_var}_to_{m_var}_to_{y_var}".replace(" ", "_").replace("/", "_")

                st.download_button(
                    label="Download plot",
                    data=buf,
                    file_name=f"mediation_plot_{plot_name}.png",
                    mime="image/png",
                    key=f"mediation_download_{plot_name}_{i}",
                    width="stretch",
                )

            with col_txt:
                interpretation = interpret_mediation_results(results, x_var, m_var, y_var, alpha)
                st.markdown(interpretation)

            st.caption(
                f"N={results['n_obs']} | Covariates: {', '.join(single_cov) if single_cov else 'None'}"
            )

            st.divider()
