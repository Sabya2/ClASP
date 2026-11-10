"""Bivariate and multivariable OLS analysis UI."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import streamlit as st

from srcs.utils_visualisation.common import (
    _candidate_group_columns,
    _corr_pair,
    get_figure_download,
    p_to_star,
    set_paper_plot_style,
    stars,
)



# ---------------------------------------------------------------------
# Expected project helpers
# ---------------------------------------------------------------------
# This page reuses the following helpers from your existing application:
# - set_paper_plot_style(show_grid=...)
# - get_plot_palette(groups, mode=...)
# - get_figure_download(fig)
# - p_to_star(p)
# - _candidate_group_columns(df, group_col)
#
# If they live in another module, import them at the top of this file.



def _safe_numeric_series(data, column):
    """Return a numeric, finite version of a dataframe column."""
    return pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _format_pvalue(p_value):
    """Format p-values for tables and text."""
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def _term_to_readable(term):
    """Convert statsmodels treatment-coded names to a readable display label."""
    return re.sub(r"C\((.*?)\)\[T\.(.*?)\]", r"\1 = \2", str(term))



def _reference_category(data, variable):
    """Infer the treatment reference category used by patsy/statsmodels."""
    if isinstance(data[variable].dtype, pd.CategoricalDtype):
        categories = data[variable].dtype.categories
        return categories[0] if len(categories) else "NA"
    values = data[variable].dropna().unique().tolist()
    if not values:
        return "NA"
    try:
        return sorted(values)[0]
    except TypeError:
        return sorted(map(str, values))[0]




def _fit_ols_model(data, outcome, predictors, primary_predictors, categorical_vars, robust_se=False):
    """Fit OLS once with conventional or HC3 covariance and finite-sample t inference."""
    formula_terms = []

    for var in predictors:
        if var in categorical_vars:
            formula_terms.append(f"C({var})")
        else:
            formula_terms.append(var)
    formula = f"{outcome} ~ " + " + ".join(formula_terms)
    if not predictors:
        raise ValueError("At least one predictor is required.")
    model = smf.ols(formula, data=data).fit(
        cov_type="HC3" if robust_se else "nonrobust",
        use_t=True,
    )
    return model, formula

def _model_terms_table(model):
    """Return an OLS coefficient table excluding the intercept."""
    term_names = list(model.model.exog_names)
    ci = pd.DataFrame(model.conf_int(), index=term_names)
    table = pd.DataFrame({
        "term": term_names,
        "beta": np.asarray(model.params),
        "std_error": np.asarray(model.bse),
        "t_value": np.asarray(model.tvalues),
        "p_value": np.asarray(model.pvalues),
        "ci_low": ci.iloc[:, 0].values,
        "ci_high": ci.iloc[:, 1].values,
    })
    table = table.loc[table["term"] != "Intercept"].copy()
    table["term_display"] = table["term"].map(_term_to_readable)
    table["significance"] = table["p_value"].map(p_to_star)
    table["statistic"] = "t"
    return table


def _extract_primary_coefficients(model, outcome, primary_predictors):
    """Extract adjusted coefficients for primary predictors."""

    params = model.params
    pvalues = model.pvalues
    conf_int = model.conf_int()

    # Robust results may return NumPy arrays instead of labeled Series
    if not hasattr(params, "index"):
        term_names = model.model.exog_names
        params = pd.Series(params, index=term_names)
        pvalues = pd.Series(pvalues, index=term_names)
        conf_int = pd.DataFrame(
            conf_int,
            index=term_names,
            columns=["ci_low", "ci_high"],
        )
    else:
        conf_int = conf_int.copy()
        conf_int.columns = ["ci_low", "ci_high"]

    rows = []

    for predictor in primary_predictors:
        matching_terms = [
            term for term in params.index
            if term == predictor or term.startswith(f"C({predictor})")
        ]

        for term in matching_terms:
            rows.append({
                "outcome": outcome,
                "predictor": predictor,
                "term": term,
                "term_display": f"{outcome}: {term}",
                "beta": float(params[term]),
                "ci_low": float(conf_int.loc[term, "ci_low"]),
                "ci_high": float(conf_int.loc[term, "ci_high"]),
                "p_value": float(pvalues[term]),
            })

    return pd.DataFrame(rows)



def _build_equation(model, outcome, categorical_vars):
    """Build a readable fitted-regression equation with categorical terms grouped."""

    params = pd.Series(np.asarray(model.params), index=model.model.exog_names)
    intercept = float(params.get("Intercept", 0.0))

    continuous_lines = []
    categorical_lines = {}

    for term, beta in params.items():
        if term == "Intercept":
            continue

        sign = "+" if beta >= 0 else "−"
        piece = f"{sign}{abs(beta):.6g}×({_term_to_readable(term)})"

        # Check whether this term belongs to a categorical variable
        matched_category = None

        for cat_var in categorical_vars:
            if term.startswith(f"C({cat_var})") or cat_var in term:
                matched_category = cat_var
                break

        if matched_category:
            categorical_lines.setdefault(matched_category, []).append(piece)
        else:
            continuous_lines.append(piece)

    # Start with intercept
    lines = [f"{outcome} = {intercept:.6g}"]

    # Add ALL continuous predictors first
    if continuous_lines:
        lines.append(" ".join(continuous_lines))

    # Add categorical predictors afterwards,
    # with levels of the same variable kept on one line
    for cat_var, terms in categorical_lines.items():
        lines.append(" ".join(terms))

    return "\n".join(lines)


def _build_symbolic_equation(outcome, predictors):
    """Build a readable symbolic equation for all selected predictors."""
    predictor_terms = " ".join(
        f"+ β{index}×({predictor})"
        for index, predictor in enumerate(predictors, start=1)
    )
    return f"{outcome} = β0 {predictor_terms} + ε"


def _coefficient_interpretation(term, beta, ci_low, ci_high, p_value, outcome,
                                additional_covars, data):
    """Create manuscript-style, conditional interpretation for one coefficient."""
    readable_term = _term_to_readable(term)
    significant_text = "statistically significant" if p_value < 0.05 else "not statistically significant"
    direction = "higher" if beta > 0 else "lower"

    categorical_match = re.match(r"C\((.*?)\)\[T\.(.*?)\]", str(term))
    if categorical_match:
        variable, exposed_level = categorical_match.groups()
        reference = _reference_category(data, variable)
        lead = (
            f"Compared with {variable} = {reference} (reference), {variable} = {exposed_level} "
            f"was associated with a {abs(beta):.6g}-unit {direction} {outcome}"
        )
    elif term in additional_covars:
        lead = (
            f"For each one-unit increase in {readable_term}, {outcome} was estimated to be "
            f"{abs(beta):.6g} units {direction}"
        )
    else:
        lead = (
            f"For each one-unit increase in {readable_term}, {outcome} was estimated to be "
            f"{abs(beta):.6g} units {direction}"
        )

    return (
        f"{lead}, conditional on all other variables in the model "
        f"(beta = {beta:.6g}, 95% CI {ci_low:.6g} to {ci_high:.6g}, "
        f"p = {_format_pvalue(p_value)}; {significant_text}). Statistical "
        "significance does not establish practical importance or causality."
    )


def _build_model_explanation(model, outcome, predictors,
                             additional_covars, categorical_vars, data, formula):
    """Create structured explanatory output for an OLS model."""
    coefficient_table = _model_terms_table(model)
    coefficient_text = []

    for _, row in coefficient_table.iterrows():
        coefficient_text.append(
            _coefficient_interpretation(
                term=row["term"],
                beta=row["beta"],
                ci_low=row["ci_low"],
                ci_high=row["ci_high"],
                p_value=row["p_value"],
                outcome=outcome,
                additional_covars=additional_covars,
                data=data,
            )
        )

    significant_terms = coefficient_table.loc[coefficient_table["p_value"] < 0.05, "term_display"].tolist()

    summary = {
        "formula": formula,
        "equation": _build_equation(model, outcome, categorical_vars),
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_pvalue": float(model.f_pvalue) if pd.notna(model.f_pvalue) else np.nan,
        "significant_terms": significant_terms,
        "coefficient_table": coefficient_table,
        "coefficient_text": coefficient_text,
        "covariance_type": "HC3 robust" if model.cov_type == "HC3" else "Conventional OLS",
        "inference_distribution": "Student t",
    }
    return summary



def _plot_bivariate(
    ax,
    data,
    x_col,
    y_col,
    group_col=None,
    selected_groups=None,
    add_reg=True,
    plot_mode="multicolor",
):
    """Draw a bivariate scatterplot, optionally stratified by group."""


    # -----------------------------------------
    # Prepare data
    # -----------------------------------------
    cols = [x_col, y_col]

    if group_col:
        cols.append(group_col)

    plot_data = data[cols].copy()
    plot_data[x_col] = _safe_numeric_series(plot_data,x_col)
    plot_data[y_col] = _safe_numeric_series(plot_data,y_col)
    plot_data = plot_data.dropna(subset=[x_col, y_col])

    # -----------------------------------------
    # Filter groups
    # -----------------------------------------
    if group_col and selected_groups:
        plot_data = plot_data[plot_data[group_col].isin(selected_groups)].copy()

    if plot_data.empty:
        return None

    # -----------------------------------------
    # Grouped plot
    # -----------------------------------------
    if group_col and selected_groups:

        group_order = [
            group
            for group in selected_groups
            if group in plot_data[group_col].unique()
        ]

        for group in group_order:

            subgroup = plot_data[plot_data[group_col] == group].copy()

            if subgroup.empty:
                continue

            # Scatter
            sns.scatterplot(
                data=subgroup, x=x_col, y=y_col,
                ax=ax, label=str(group), alpha=0.9, edgecolor="white", s=28,
                )

            # Regression line
            if ( add_reg and len(subgroup) >= 2 and subgroup[x_col].nunique() > 1):
                sns.regplot(
                    data=subgroup, x=x_col, y=y_col,
                    scatter=False, ax=ax, line_kws={ "linewidth": 1.3 },
                    )

        ax.legend( title=group_col, frameon=False, )

    # -----------------------------------------
    # Overall plot
    # -----------------------------------------
    else:

        sns.scatterplot(
            data=plot_data, x=x_col, y=y_col, ax=ax,
            alpha=0.9, edgecolor="white", s=28,)

        if (add_reg and len(plot_data) >= 2 and plot_data[x_col].nunique() > 1):
            sns.regplot(
                data=plot_data, x=x_col, y=y_col,
                scatter=False, ax=ax, line_kws={"linewidth": 1.3},
                )

    return plot_data



def _plot_forest_by_predictor(coefficient_table, predictor):
    """Plot one primary predictor across all outcomes."""

    plot_data = coefficient_table[coefficient_table["predictor"] == predictor].copy()

    if plot_data.empty:
        return None

    plot_data = plot_data.sort_values(["outcome", "term"]).reset_index(drop=True)
    terms_per_outcome = plot_data.groupby("outcome")["term"].transform("size")
    plot_data["plot_label"] = [
        outcome if term_count == 1
        else f"{outcome} | {_term_to_readable(term)}"
        for outcome, term, term_count in zip(
            plot_data["outcome"], plot_data["term"], terms_per_outcome
        )
    ]

    fig_height = max(1.8, 0.65 * len(plot_data) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    for idx, row in plot_data.iterrows():
        color = "#2E7D32" if row["p_value"] < 0.05 else "#607D8B"
        ax.errorbar(
            x=row["beta"], y=idx,
            xerr=[[row["beta"] - row["ci_low"]], [row["ci_high"] - row["beta"]]],
            fmt="o", color=color, capsize=3, linewidth=1.5, markersize=8,
        )

    ax.axvline(0,color="black",linestyle="--",linewidth=1)
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data["plot_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted beta coefficient (95% CI)")
    ax.set_title(f"{predictor} across fitted outcomes")
    ax.grid(axis="x", alpha=0.25)

    for idx, row in plot_data.iterrows():
        p = f"p = {_format_pvalue(row['p_value'])} {stars(row['p_value'])}"
        ax.text( 1.02, idx, p,
            transform=ax.get_yaxis_transform(),
            color = ("#2E7D32" if row["p_value"] < 0.05 else "#607D8B"),
            clip_on=False, va="center", ha="left", fontsize=8
            )
    fig.tight_layout()

    return fig


def _partial_regression_data(model, term):
    """Residualize one model term and the outcome against all other terms."""
    term_names = list(model.model.exog_names)
    if term not in term_names:
        raise ValueError(f"Model term not found: {term}")

    exog = np.asarray(model.model.exog, dtype=float)
    outcome_values = np.asarray(model.model.endog, dtype=float)
    term_index = term_names.index(term)
    predictor_values = exog[:, term_index]
    control_indices = [index for index in range(exog.shape[1]) if index != term_index]

    if control_indices:
        controls = exog[:, control_indices]
        predictor_fit = controls @ np.linalg.lstsq(
            controls, predictor_values, rcond=None
        )[0]
        outcome_fit = controls @ np.linalg.lstsq(
            controls, outcome_values, rcond=None
        )[0]
        predictor_residuals = predictor_values - predictor_fit
        outcome_residuals = outcome_values - outcome_fit
    else:
        predictor_residuals = predictor_values.copy()
        outcome_residuals = outcome_values.copy()

    return pd.DataFrame({
        "Adjusted predictor component": predictor_residuals,
        "Adjusted outcome component": outcome_residuals,
    })


def _plot_adjusted_scatter(model, outcome, term, predictor=None):
    """Plot observed values with a covariate-adjusted conditional fit.

    Points and both axes remain in the imported variables' original units.  The
    fitted line/points come from the full model while other predictors are held
    at their median (numeric) or reference category (categorical).
    """
    model_frame = model.model.data.frame.copy()
    predictor = predictor or term
    if predictor not in model_frame or outcome not in model_frame:
        raise ValueError(f"Raw plotting columns are unavailable for model term: {term}")

    plot_data = model_frame[[predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_data.empty or plot_data[predictor].nunique() < 2:
        return None, plot_data

    categorical = (
        isinstance(model_frame[predictor].dtype, pd.CategoricalDtype)
        or not pd.api.types.is_numeric_dtype(model_frame[predictor])
        or str(term).startswith(f"C({predictor})")
    )
    if categorical:
        categories = model_frame[predictor].dropna().drop_duplicates().tolist()
        category_match = re.match(rf"C\({re.escape(str(predictor))}\)\[T\.(.*?)\]", str(term))
        if category_match:
            exposed = category_match.group(1)
            reference = _reference_category(model_frame, predictor)
            categories = [value for value in categories if str(value) in {str(reference), exposed}]
        prediction_x = categories
    else:
        raw_x = pd.to_numeric(plot_data[predictor], errors="coerce").dropna()
        prediction_x = np.linspace(raw_x.min(), raw_x.max(), 100)

    prediction_data = pd.concat(
        [model_frame.iloc[[0]].copy()] * len(prediction_x), ignore_index=True
    )
    for column in model_frame.columns:
        if column in {outcome, predictor}:
            continue
        series = model_frame[column].dropna()
        if series.empty:
            continue
        if pd.api.types.is_numeric_dtype(series) and not isinstance(series.dtype, pd.CategoricalDtype):
            prediction_data[column] = series.median()
        else:
            prediction_data[column] = _reference_category(model_frame, column)
    prediction_data[predictor] = prediction_x
    prediction = model.get_prediction(prediction_data).summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    if categorical:
        visible_data = plot_data[plot_data[predictor].astype(str).isin({str(value) for value in prediction_x})]
        sns.stripplot(
            data=visible_data, x=predictor, y=outcome, order=prediction_x,
            jitter=False, ax=ax, color="#4C78A8", alpha=0.75, size=5,
        )
        observed_limits = ax.get_ylim()
        positions = np.arange(len(prediction_x))
        ax.errorbar(
            positions, prediction["mean"],
            yerr=np.vstack([
                prediction["mean"] - prediction["mean_ci_lower"],
                prediction["mean_ci_upper"] - prediction["mean"],
            ]),
            color="#B2182B", marker="o", linewidth=1.6, capsize=3,
            label="Adjusted model estimate (95% CI)",
        )
        ax.set_ylim(observed_limits)
    else:
        sns.scatterplot(
            data=plot_data, x=predictor, y=outcome, ax=ax,
            alpha=0.85, edgecolor="white", s=36,
        )
        observed_xlim, observed_ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot(prediction_x, prediction["mean"], color="#B2182B", linewidth=1.6,
                label="Adjusted model estimate")
        ax.fill_between(
            prediction_x, prediction["mean_ci_lower"], prediction["mean_ci_upper"],
            color="#B2182B", alpha=0.16, linewidth=0,
        )
        # Confidence bands must not redefine the displayed range: limits stay
        # based solely on the observed raw measurements.
        ax.set_xlim(observed_xlim)
        ax.set_ylim(observed_ylim)

    ax.set_title(f"{outcome} vs {_term_to_readable(term)} (adjusted model fit)")
    ax.set_xlabel(f"{predictor} (observed values)")
    ax.set_ylabel(f"{outcome} (observed values)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, plot_data


def _forest_interpretation_lines(plot_data, layout):
    """Return plot-specific explanations for displayed forest coefficients."""
    lines = []
    for _, row in plot_data.iterrows():
        term = _term_to_readable(row["term"])
        estimate = (
            f"beta = {row['beta']:.3f}, 95% CI {row['ci_low']:.3f} to "
            f"{row['ci_high']:.3f}, p = {_format_pvalue(row['p_value'])}"
        )
        if layout == "Combined":
            lines.append(
                f"**{row['outcome']}:** adjusted association of {term} with "
                f"{row['outcome']} ({estimate})."
            )
        else:
            lines.append(
                f"**{term}:** adjusted association with {row['outcome']} "
                f"({estimate})."
            )
    return lines


def _plot_beta_heatmap(explanations, outcomes, predictors):
    """Create an outcome-by-predictor heatmap of beta estimates.
    For categorical predictors, each contrast is shown as an individual column.
    """
    rows = []
    for outcome in outcomes:
        coef_table = explanations[outcome]["coefficient_table"].copy()
        for _, row in coef_table.iterrows():
            rows.append({
                "outcome": outcome,
                "term": row["term_display"],
                "beta": row["beta"],
                "p_value": row["p_value"],
            })

    long_data = pd.DataFrame(rows)
    if long_data.empty:
        return None, pd.DataFrame()

    beta_matrix = long_data.pivot(index="outcome", columns="term", values="beta")
    beta_matrix = beta_matrix.reindex(index=outcomes)

    annotation_matrix = long_data.copy()
    annotation_matrix["annot"] = annotation_matrix.apply(
        lambda row: f"{row['beta']:.2f}{stars(row['p_value'])}", axis=1
    )
    annotation_matrix = annotation_matrix.pivot(index="outcome", columns="term", values="annot")
    annotation_matrix = annotation_matrix.reindex(index=outcomes, columns=beta_matrix.columns)

    finite_values = beta_matrix.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        return None, beta_matrix

    max_abs = max(abs(finite_values.min()), abs(finite_values.max()))
    if max_abs == 0:
        max_abs = 1.0

    fig_width = max(8, 1.35 * len(beta_matrix.columns) + 2)
    fig_height = max(3.5, 0.9 * len(beta_matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        beta_matrix,
        annot=annotation_matrix,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Adjusted beta coefficient"},
        ax=ax,
    )
    ax.set_title("Adjusted OLS beta coefficients")
    ax.set_xlabel("Predictors / categorical contrasts")
    ax.set_ylabel("Outcomes")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    return fig, beta_matrix



def _render_explanation(explanation, outcome, analysis_mode="Multivariable OLS"):
    """
    Render equation, model fit, table, and interpretation in Streamlit.

    Uses horizontal side-by-side expanders for coefficients and interpretation
    to create a more compact, genie-like effect.
    """
    st.markdown("#### Model equation")
    st.code(explanation["equation"], wrap_lines = True, language=None)
    st.caption(
        "The equation is a fitted conditional mean model. For categorical variables, "
        "the displayed coefficient corresponds to the treatment-coded contrast against its reference level."
    )
    st.caption(
        f"Covariance: {explanation['covariance_type']}; confidence intervals and p-values "
        f"use the {explanation['inference_distribution']} distribution."
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("N", explanation["n"])
    metric_2.metric("R\u00B2", f"{explanation['r_squared']:.3f}")
    metric_3.metric("Adjusted R\u00B2", f"{explanation['adj_r_squared']:.3f}")
    metric_4.metric("Model p", _format_pvalue(explanation["f_pvalue"]))

    # -------------------------------------------------
    # Horizontal expanders (side-by-side)
    # -------------------------------------------------
    col_coeff, col_interp = st.columns([1, 1], gap="small")

    with col_coeff:
        with st.expander("\U0001F4CA Coefficients table", expanded=False):
            st.markdown("##### Coefficients")
            coefficient_table = explanation["coefficient_table"].copy()
            display_table = coefficient_table[[
                "term_display", "beta", "std_error", "ci_low", "ci_high", "p_value", "significance"
            ]].rename(columns={
                "term_display": "Term",
                "beta": "Beta",
                "std_error": "SE",
                "ci_low": "95% CI low",
                "ci_high": "95% CI high",
                "p_value": "p-value",
                "significance": "Sig.",
            })
            st.dataframe(
                display_table.style.format({
                    "Beta": "{:.3f}",
                    "SE": "{:.3f}",
                    "95% CI low": "{:.3f}",
                    "95% CI high": "{:.3f}",
                    "p-value": "{:.4f}",
                }),
                width="stretch",
                hide_index=True,
            )

    with col_interp:
        with st.expander("\U0001F4A1 Interpretation", expanded=False):
            st.markdown("##### Interpretation")
            if explanation["coefficient_text"]:
                for sentence in explanation["coefficient_text"]:
                    st.markdown(f"- {sentence}")
            else:
                st.info("No estimable coefficients were available for interpretation.")

            if explanation["significant_terms"]:
                st.success(
                    "Statistically significant adjusted terms at alpha = 0.05: "
                    + ", ".join(explanation["significant_terms"])
                    + "."
                )
            else:
                st.info("No adjusted terms reached p < 0.05 in this model.")



    st.caption(
        "Interpret coefficients as adjusted associations, not necessarily causal effects. "
        "Validity also depends on the outcome scale, model specification, linearity, residual behavior, "
        "influential observations, and multicollinearity. "
        "**R\u00B2** (coefficient of determination) indicates the proportion of variance in the outcome explained by all predictors jointly "
        "(e.g., R\u00B2 = 0.35 means 35% of the variability is accounted for by the model). "
        "**Adjusted R\u00B2** penalizes R\u00B2 for the number of predictors, making it more suitable for comparing models with different numbers of variables. "
        "**RMSE** (root mean squared error, or residual standard deviation) represents the typical prediction error in the original units of the outcome "
        "(e.g., if the outcome is in mm, RMSE = 1.2 means predictions typically deviate from observed values by about 1.2 mm)."
    )


def render_ols_page(df, continuous_cols, categorical_cols, group_col):
    """
    Render a Streamlit page for bivariate and multivariable OLS analysis.

    The page follows the layout pattern of render_linear_page:
    - variable selectors
    - optional group handling for bivariate scatter plots
    - visual style controls
    - plot grid / forest plot / heatmap
    - expandable statistics and interpretation panels

    Parameters
    ----------
    df : pandas.DataFrame
        Uploaded analysis dataframe.
    continuous_cols : list[str]
        Numeric variables available for X/Y selection.
    group_col : str or None
        Default categorical grouping variable selected elsewhere in the app.
    """
    # -----------------------------------------------------
    # Input checks
    # -----------------------------------------------------
    if df is None:
        st.warning("Upload data first.")
        return

    if len(continuous_cols) < 2:
        st.warning("Select at least two continuous variables in the sidebar.")
        return

    # -----------------------------------------------------
    # Page controls
    # -----------------------------------------------------

    # Migrate sessions created before the terminology was corrected.
    if st.session_state.get("ols_analysis_mode") == "Univariate Analysis":
        st.session_state["ols_analysis_mode"] = "Bivariate Analysis"

    analysis_mode = st.pills(
        "Analysis type",
        options=["Bivariate Analysis", "Multivariate Analysis"],
        selection_mode="single",
        default=None,
        key="ols_analysis_mode",
        help=(
            "Bivariate OLS analyzes one predictor and one outcome at a time. Multivariable OLS "
            "fits selected predictors jointly."
        ),
    )
    st.session_state["analysis_mode"] = analysis_mode
    plot_mode = "multicolor"

    # -----------------------------------------------------
    # Bivariate OLS page
    # -----------------------------------------------------
    if analysis_mode == "Bivariate Analysis":
        _render_bivariate_ols_section(df, continuous_cols, categorical_cols, group_col, plot_mode=plot_mode)
        return

    # -----------------------------------------------------
    # Multivariable OLS page
    # -----------------------------------------------------
    if analysis_mode == "Multivariate Analysis":
        _render_multivariable_ols_section(df, continuous_cols, categorical_cols,)
        return


def _render_bivariate_ols_section(df, continuous_cols, categorical_cols, group_col, plot_mode):
    """Render the bivariate OLS scatterplot section."""
    st.markdown("### Bivariate OLS")

    right_col, left_col = st.columns([1, 1], border=True, gap="small")
    with left_col:
        x_cols = st.multiselect("(X) Predictor variable(s)",
            options=continuous_cols,
            default=continuous_cols[:1],
            key="uniols_x_multiselect",
        )

    with right_col:
        possible_y = [col for col in continuous_cols if col not in x_cols]
        y_cols = st.multiselect("(Y) Outcome variable(s)",
            options=possible_y,
            key="uniols_y_multiselect",
        )

    c1, c2, c3= st.columns([1,1,4], border=False, gap="xsmall")

    with c1:
            add_reg = st.checkbox("Regression line",
                value=True,
                key="uniols_add_reg",
            )

            ncols = st.slider("Grid columns",
                min_value=1,
                max_value=5,
                value=2,
                key="uniols_grid_cols",
            )


    with c2:
        plot_by_group = st.checkbox("Plot by group",
            value=False,
            key="uniols_plot_by_group",
        )

    with c3:
        selected_groups = None

        if plot_by_group:
            plot_group_col = group_col
            if not group_col:
                st.warning("Select a valid group column in the sidebar first.")
                return

            possible_group_cols = _candidate_group_columns(df, group_col)
            g1, g2 = st.columns([1.5, 2.2], gap="small", border=False)

            with g1:
                use_alt_group = st.checkbox("Use a different group variable on this page",
                    value=False,
                    key="uniols_use_alt_group",
                )
                if use_alt_group:
                    plot_group_col = st.selectbox("Group column",
                        options=possible_group_cols,
                        index=possible_group_cols.index(group_col) if group_col in possible_group_cols else 0,
                        key="uniols_group_col_selectbox",
                    )

            with g2:
                available_groups = df[plot_group_col].dropna().unique().tolist()
                state_key = f"uniols_selected_groups__{plot_group_col}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = available_groups[: min(3, len(available_groups))]

                st.session_state[state_key] = [
                    group for group in st.session_state[state_key] if group in available_groups
                ]

                selected_groups = st.multiselect("Groups to plot",
                    options=available_groups,
                    default=st.session_state[state_key],
                    key=f"{state_key}_widget",
                )
                st.session_state[state_key] = selected_groups

                if not selected_groups:
                    st.info("Select at least one group.")
                    return

    if not x_cols:
        st.info("Select at least one X variable.")
        return
    if not y_cols:
        st.info("Select at least one Y variable.")
        return

    plot_pairs = [
        (x_col, y_col)
        for x_col in x_cols
        for y_col in y_cols
    ]

    st.markdown("### Linear plots")
    set_paper_plot_style()

    for row_start in range(0, len(plot_pairs), ncols):

        row_pairs = plot_pairs[row_start: row_start + ncols]
        row_layout = st.columns(ncols)

        for container, (x_col, y_col) in zip(row_layout, row_pairs):
            # sns.set_style("white")
            with container:

                fig, ax = plt.subplots(figsize=(6.0, 4.5))

                # -----------------------------------------
                # Plot
                # -----------------------------------------
                plot_data = _plot_bivariate(
                    ax=ax, data=df, x_col=x_col, y_col=y_col,
                    group_col=plot_group_col if plot_by_group else None,
                    selected_groups=selected_groups,
                    add_reg=add_reg,
                    plot_mode=plot_mode,
                )

                if plot_data is None or plot_data.empty:
                    st.info(f"No valid data for {y_col} vs {x_col}.")
                    plt.close(fig)
                    continue

                # -----------------------------------------
                # OLS
                # -----------------------------------------
                try:

                    model, formula = _fit_ols_model(
                        data=plot_data,
                        outcome=y_col,
                        predictors=[x_col],
                        primary_predictors=[x_col],
                        categorical_vars=[],
                        robust_se=False,
                    )

                    explanation = _build_model_explanation(
                        model=model,
                        outcome=y_col,
                        predictors=[x_col],
                        additional_covars=[],
                        categorical_vars=[],
                        data=plot_data,
                        formula=formula,
                    )

                    title = (
                        f"{y_col} vs {x_col} | "
                        f"n={int(model.nobs)}"
                    )

                except Exception as error:

                    st.warning(
                        f"OLS could not be fitted for "
                        f"{y_col} vs {x_col}: {error}"
                    )

                    plt.close(fig)
                    continue

                # -----------------------------------------
                # Formatting
                # -----------------------------------------
                ax.set_title(title)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)

                fig.tight_layout()

                st.pyplot(
                    fig,
                    width="stretch"
                )

                plt.close(fig)


                with st.expander(f"OLS result and explanation: {y_col} vs {x_col}", expanded=False):
                    _render_explanation(explanation, y_col)


                    # -------------------------------------------------
                    # Per-group statistics: only for grouped plots
                    # -------------------------------------------------
                    if plot_by_group and plot_group_col and selected_groups:
                        group_stats_lines = []

                        # Use the same finite numeric data plotted by _plot_bivariate().
                        stats_columns = [x_col, y_col, plot_group_col]
                        stats_data = df[stats_columns].copy()
                        stats_data[x_col] = _safe_numeric_series(stats_data, x_col)
                        stats_data[y_col] = _safe_numeric_series(stats_data, y_col)
                        stats_data = stats_data.replace([np.inf, -np.inf], np.nan).dropna()
                        stats_data = stats_data.loc[
                            stats_data[plot_group_col].isin(selected_groups)
                        ].copy()

                        group_order = [
                            group for group in selected_groups
                            if group in stats_data[plot_group_col].unique()
                        ]

                        for group in group_order:
                            grp_df = stats_data.loc[stats_data[plot_group_col] == group].copy()

                            # Need at least 3 observations and variation in X for valid simple OLS.
                            if len(grp_df) < 3 or grp_df[x_col].nunique() < 2:
                                group_stats_lines.append(
                                    f"**{group}**  \n"
                                    f"- n = {len(grp_df)}; insufficient observations or no variation in {x_col} for OLS."
                                )
                                continue

                            try:
                                # Use the same model fitting approach as multivariable OLS
                                grp_model, grp_formula = _fit_ols_model(
                                    data=grp_df, outcome=y_col, predictors=[x_col],
                                    primary_predictors=[x_col], categorical_vars=[], robust_se=False,
                                )
                                grp_explanation = _build_model_explanation(
                                    model=grp_model, outcome=y_col, predictors=[x_col],
                                    additional_covars=[], categorical_vars=[], data=grp_df,
                                    formula=grp_formula,
                                )

                                # Get correlation for additional context
                                grp_r, grp_p = _corr_pair(grp_df[x_col], grp_df[y_col], method="pearson")

                                # Build equation string
                                sign = "+" if grp_explanation["coefficient_table"]["beta"].iloc[0] >= 0 else "-"
                                beta_val = abs(grp_explanation["coefficient_table"]["beta"].iloc[0])
                                intercept_val = float(np.asarray(grp_model.params)[0])
                                equation = f"{y_col} = {intercept_val:.3f} {sign} {beta_val:.3f} \u00d7 {x_col}"

                                # Format significance stars
                                r_star = p_to_star(grp_p) if pd.notna(grp_r) else ""
                                beta_row = grp_explanation["coefficient_table"].iloc[0]
                                beta_star = p_to_star(beta_row["p_value"])

                                group_stats_lines.append(
                                    (
                                        f"**{group}**  \n"
                                        f"- **Equation**: {equation}  \n"
                                        f"- **N** = {grp_explanation['n']} | **R\u00B2** = {grp_explanation['r_squared']:.3f} | **Adj. R\u00B2** = {grp_explanation['adj_r_squared']:.3f}  \n"
                                        f"- **Model p** = {_format_pvalue(grp_explanation['f_pvalue'])}  \n"
                                        f"- **Pearson r** = {grp_r:.2f}{r_star} (p = {_format_pvalue(grp_p)})  \n"
                                        f"- **Beta** = {beta_row['beta']:.3f}{beta_star} (p = {_format_pvalue(beta_row['p_value'])})  \n"
                                        # Uncomment the line below if you want RMSE
                                        # f"- **RMSE** = {grp_explanation['coefficient_table']['noise_sd'].iloc[0]:.3f}"
                                    )
                                )

                            except Exception as error:
                                group_stats_lines.append(
                                    f"**{group}**  \n"
                                    f"- OLS model could not be fitted: {error}"
                                )

                        if group_stats_lines:
                            with st.expander(
                                f"Per-group statistics: {y_col} vs {x_col}",
                                expanded=False,
                            ):
                                st.markdown("\n\n".join(group_stats_lines))
                                st.caption(
                                    "These are separate, unadjusted simple OLS models fitted within each selected group. "
                                    "They are not the same as a pooled model with group adjustment or an interaction term."
                                )

                buf = get_figure_download(fig)
                st.download_button(
                    label=f"Download {y_col} vs {x_col}",
                    data=buf,
                    file_name=f"bivariate_ols_{x_col}_vs_{y_col}.png",
                    mime="image/png",
                    key=f"uniols_download_{x_col}_{y_col}",
                )
                plt.close(fig)




def _render_multivariable_ols_section(
    df,
    continuous_cols,
    categorical_cols,
    # group_col,
):
    """Render multivariable OLS models, forest plots, and optional heatmap."""

    # -----------------------------------------------------
    # 1. Select outcomes and predictors
    # -----------------------------------------------------
    st.markdown("### Multivariable OLS")

    st.caption(
        "Each selected outcome is modeled separately using the same predictor set. "
        "All coefficients are mutually adjusted for the other variables in the model."
    )

    available_columns = list(dict.fromkeys(continuous_cols + categorical_cols))

    selector_left, selector_right = st.columns([1, 1],border=True,gap="small",)
    with selector_right:
        y_cols = st.multiselect("(Y) Outcome variable(s)",
            options=available_columns,
            placeholder="Select one or more outcomes",
            key="multiols_y_multiselect",
            help=(
                "Select one or more continuous outcomes. "
                "A separate OLS model is fitted for each outcome."
            ),
        )

    with selector_left:
        candidate_x = [col for col in available_columns if col not in y_cols]
        x_cols = st.multiselect("(X) Predictor variable(s)",
            options=candidate_x,
            placeholder="Select one or more predictors",
            key="multiols_x_multiselect",
            help=(
                "All selected predictors enter every outcome model "
                "simultaneously."
            ),
        )

    if not y_cols and not x_cols:
        st.info("Select at least one outcome and one predictor variable.")
        return

    # -----------------------------------------------------
    # 2. Select predictor roles
    # -----------------------------------------------------

    st.markdown("### Predictor roles")
    st.caption(
        "Primary predictors are the main variables of interest. "
        "All other selected X variables are included as adjustment covariates."
    )

    role_left, role_right = st.columns([2, 2],border=True,gap="small",)
    with role_left:
        primary_predictors = st.multiselect("Primary predictors",
            options=x_cols,
            default=None,
            placeholder="Select primary predictor(s)",
            key="multiple_primary_predictors",
            help=(
                "These variables will be displayed in the forest plots. "
                "All selected X variables still enter every model."
            ),
        )

    additional_covars = [col for col in x_cols if col not in primary_predictors]
    with role_right:
        categorical_candidates = [col for col in x_cols if col in categorical_cols]
        categorical_vars = []
        if categorical_candidates:
            categorical_vars = st.multiselect("Categorical variables",
                options=categorical_candidates,
                default=[],
                key="multiols_categorical_multiselect",
                help=(
                    "Selected variables are treated as categorical using "
                    "C(variable) in the statsmodels formula."
                ),
            )
            if categorical_vars:
                variable_labels = []
                for col in categorical_vars:
                    n_levels = df[col].dropna().nunique()
                    if n_levels == 2:
                        variable_type = "binary"
                    elif n_levels >= 3:
                        variable_type = "multilevel/ordinal"
                    else:
                        variable_type = "insufficient levels"
                    variable_labels.append(
                        f"{col}: {variable_type}"
                    )
                st.caption(" | ".join(variable_labels))

    # -----------------------------------------------------
    # 3. Model options
    # -----------------------------------------------------


    role_left, role_right = st.columns([3, 1],border=False,gap="small",)
    with role_right:
        robust_se = st.checkbox("Use HC3 robust standard errors",
            value=False,
            key="multiols_robust_se",
            help=("Changes standard errors, confidence intervals, and p-values. "
                  "Coefficient estimates remain unchanged."
            ),
        )

    with role_left:
        st.markdown("### Combined Forest plot or separate linear plots?")
        forest_layout = st.pills(
            "Plot layout",
            options=["Separate", "Combined"],
            default="Separate",
            selection_mode="single",
            key="multiols_forest_layout",
            help=(
            "Separate creates raw-value scatterplots with an adjusted model fit "
                "for every outcome–primary-predictor pair. Combined preserves the predictor-across-outcomes "
                "forest-plot view."
            ),
        )

    if primary_predictors is not None and len(primary_predictors) > 0:
        run_analysis = st.button("Run multivariable OLS analysis",
                                 type="primary",key="multiple_ols_run_button",
                                 use_container_width=True,)
        if not run_analysis:
            st.info(
                        "Select outcomes and predictors, then click "
                        "'Run multivariable OLS analysis'."
                    )
            return

    # -----------------------------------------------------
    # 4. Validate variables
    # -----------------------------------------------------
    required_columns = list(dict.fromkeys(y_cols + x_cols))

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error("These selected variables are missing from the dataframe: " + ", ".join(missing_columns))
        return

    # -----------------------------------------------------
    # 5. Fit one model per outcome
    # -----------------------------------------------------
    results = {}
    explanations = {}
    failed_models = {}

    for outcome_var in y_cols:
        model_columns = [outcome_var] + x_cols
        model_data = ( df[model_columns] .replace([np.inf, -np.inf], np.nan) .dropna() )
        minimum_n = max( 10, len(x_cols) + 3, )

        if len(model_data) < minimum_n:
            failed_models[outcome_var] = (
                "Too few complete observations after missing-data "
                f"removal (n={len(model_data)}, required={minimum_n})."
            )
            continue

        try:
            model, formula = _fit_ols_model(
                data=model_data,
                outcome=outcome_var,
                predictors=x_cols,
                primary_predictors=primary_predictors,
                categorical_vars=categorical_vars,
                robust_se=robust_se,
            )

            results[outcome_var] = model

            explanations[outcome_var] = _build_model_explanation(
                model=model,
                outcome=outcome_var,
                predictors=x_cols,
                additional_covars=additional_covars,
                categorical_vars=categorical_vars,
                data=model_data,
                formula=formula,
            )

        except Exception as error:
            failed_models[outcome_var] = str(error)

    # -----------------------------------------------------
    # 6. Display failed models
    # -----------------------------------------------------
    for outcome_var, message in failed_models.items():
        st.warning(f"{outcome_var}: model not fitted — {message}")

    if not results:
        st.error("No model could be fitted with the selected variables.")
        return

    fitted_outcomes = list(results.keys())

    # -----------------------------------------------------
    # 7. Extract primary-predictor coefficients
    # -----------------------------------------------------
    coefficient_tables = []

    for outcome_var, model in results.items():
        coefficient_table = _extract_primary_coefficients(model=model,
                                                          outcome=outcome_var,
                                                          primary_predictors=primary_predictors,)
        coefficient_tables.append(coefficient_table)
    if not coefficient_tables:
        st.warning("No coefficients were found for the selected primary predictors.")
        return

    all_coefficients = pd.concat(coefficient_tables,ignore_index=True,)
    if all_coefficients.empty:
        st.warning("No coefficients were found for the selected primary predictors.")
        return

    # -----------------------------------------------------
    # 8. Forest plots
    # -----------------------------------------------------
    st.markdown("### Adjusted estimates")

    if forest_layout == "Combined":
        st.caption(
            "Combined view compares one primary predictor across all fitted "
            "outcomes. Each predictor is presented in its own summary card."
        )

        for predictor in primary_predictors:
            predictor_data = all_coefficients[
                all_coefficients["predictor"] == predictor
            ].copy().sort_values(["outcome", "term"])
            if predictor_data.empty:
                st.warning(f"No fitted coefficient found for primary predictor: {predictor}")
                continue

            fig = _plot_forest_by_predictor(all_coefficients, predictor)
            if fig is None:
                continue

            other_predictors = [
                variable for variable in x_cols if variable != predictor
            ]
            adjustment_text = (
                ", ".join(other_predictors)
                if other_predictors else "no other predictors"
            )

            with st.container(border=True):
                st.markdown(f"#### Primary predictor: {predictor}")
                st.caption(
                    f"Compared across {predictor_data['outcome'].nunique()} fitted "
                    f"outcome(s) | Adjusted for: {adjustment_text}"
                )
                st.pyplot(fig, width="stretch")
                st.markdown(
                    f"**Simple interpretation:** Each point shows the adjusted "
                    f"association of **{predictor}** with the outcome named on its "
                    "row. Horizontal lines are 95% confidence intervals; the dashed "
                    "vertical line represents no association. Green points have "
                    "p < 0.05 and gray points have p ≥ 0.05."
                )

                with st.expander("Detailed outcome estimates", expanded=False):
                    for line in _forest_interpretation_lines(
                        predictor_data, "Combined"
                    ):
                        st.markdown(f"- {line}")
                    st.caption(
                        "Coefficient sizes are expressed in each outcome's original "
                        "units. Magnitudes should not be compared directly when the "
                        "outcomes use different scales."
                    )

                forest_buffer = get_figure_download(fig)
                st.download_button(
                    label=f"Download forest plot: {predictor}",
                    data=forest_buffer,
                    file_name=f"multivariable_ols_combined_forest_{predictor}.png",
                    mime="image/png",
                    key=f"multivariable_ols_combined_forest_download_{predictor}",
                    use_container_width=True,
                )
            plt.close(fig)
    else:
        st.caption(
            "Separate view: each outcome–primary-predictor pair is shown as an "
            "raw-value scatterplot with a model-adjusted fitted relationship. "
            "The axes and points use the imported values; other predictors are "
            "held constant for the red fitted line. "
            "Plots are grouped by outcome, with up to three predictors per row."
        )

        for outcome in fitted_outcomes:
            outcome_data = all_coefficients[
                all_coefficients["outcome"] == outcome
            ].copy()
            if outcome_data.empty:
                st.warning(f"No primary-predictor coefficients found for outcome: {outcome}")
                continue

            # with st.container(border=True):
            with st.expander(f'Outcome: {outcome}', expanded=True):
                # st.markdown(f"#### Outcome: {outcome}")
                st.markdown("**Complete model equation**")
                # st.code(
                #     _build_symbolic_equation(outcome, x_cols),
                #     language=None,
                #     wrap_lines=True,
                # )
                # st.markdown("**Fitted equation**")
                st.code(
                    explanations[outcome]["equation"],
                    language=None,
                    wrap_lines=True,
                )
                st.caption(
                    "Each beta belongs to the predictor in the same position. "
                    "Every coefficient is estimated while holding the other "
                    "predictors in this equation constant."
                )

            outcome_coefficients = [
                coefficient for _, coefficient in outcome_data.iterrows()
            ]

            for row_start in range(0, len(outcome_coefficients), 3):
                row_coefficients = outcome_coefficients[row_start:row_start + 3]
                plot_columns = st.columns(len(row_coefficients), gap="small")

                for plot_column, coefficient in zip(
                    plot_columns, row_coefficients
                ):
                    with plot_column:
                        term = coefficient["term"]
                        fig, partial_data = _plot_adjusted_scatter(
                            results[outcome], outcome, term,
                            predictor=coefficient["predictor"],
                        )
                        if fig is None:
                            st.warning(
                                f"Adjusted scatterplot could not be created for "
                                f"{outcome} vs {_term_to_readable(term)}."
                            )
                            continue

                        st.pyplot(fig, width="stretch")
                        other_predictors = [
                            predictor for predictor in x_cols
                            if predictor != coefficient["predictor"]
                        ]
                        if other_predictors:
                            adjusted_for = ", ".join(other_predictors)
                        #     st.markdown(
                        #         f"**Simple interpretation:** The beta for "
                        #         f"**{_term_to_readable(term)}** represents its "
                        #         f"association with **{outcome}**, adjusted for "
                        #         f"{adjusted_for}."
                        #     )
                        # else:
                        #     st.markdown(
                        #         f"**Simple interpretation:** The beta for "
                        #         f"**{_term_to_readable(term)}** represents its "
                        #         f"unadjusted association with **{outcome}**."
                        #     )

                        with st.expander(
                            "**Interpretation**", expanded=False):
                            st.markdown(
                                        f"The beta for "
                                        f"**{_term_to_readable(term)}** represents its "
                                        f"association with **{outcome}**, adjusted for "
                                        f"{adjusted_for}. "
                                        "The points and axes show the original observed values. "
                                        "The red line is the full model's adjusted relationship, "
                                        "with the other numeric predictors held at their medians "
                                        "and categorical predictors at their reference levels."
                            )

                            # st.markdown(
                            #     f"Each point compares the part of "
                            #     f"**{_term_to_readable(term)}** not explained by "
                            #     f"the other selected predictors with the part of "
                            #     f"**{outcome}** not explained by those predictors. "
                            #     "The red line is the adjusted linear relationship; "
                            #     "its slope equals the coefficient from the full "
                            #     "multivariable model."
                            # )
                            for line in _forest_interpretation_lines(
                                pd.DataFrame([coefficient]), "Separate"
                            ):
                                st.markdown(f"- {line}")

                        scatter_buffer = get_figure_download(fig)
                        safe_term = re.sub(
                            r"[^A-Za-z0-9_.-]+", "_", str(term)
                        )
                        st.download_button(
                            label=(
                                f"Download adjusted scatterplot: {outcome} vs "
                                f"{_term_to_readable(term)}"
                            ),
                            data=scatter_buffer,
                            file_name=(
                                f"multivariable_ols_adjusted_scatter_"
                                f"{outcome}_{safe_term}.png"
                            ),
                            mime="image/png",
                            key=(
                                f"multivariable_ols_adjusted_scatter_download_"
                                f"{outcome}_{safe_term}"
                            ),
                        )
                        plt.close(fig)

    # -----------------------------------------------------
    # 9. Optional heatmap
    # -----------------------------------------------------
    if len(fitted_outcomes) > 1:
        with st.expander("Beta heatmap", expanded=False):
            st.markdown(
            "This heatmap shows adjusted beta coefficients for each outcome-predictor pair. "
            "For categorical predictors, each contrast is shown as an individual column.")

            heatmap_fig, beta_matrix = _plot_beta_heatmap(explanations=explanations, outcomes=fitted_outcomes, predictors=x_cols,)

            if heatmap_fig is not None:
                st.pyplot(heatmap_fig,width="stretch",)
                heatmap_buffer = get_figure_download(heatmap_fig)
                col1, col2 = st.columns([1, 1], gap="small")

                with col1:
                    st.download_button(label="Download beta heatmap",
                        data=heatmap_buffer,
                        file_name=("multivariable_ols_beta_heatmap.png"),
                        mime="image/png",
                        key="multivariable_ols_heatmap_download",
                    )

                with col2:
                    st.download_button(label="Download beta matrix (.csv)",
                        data=beta_matrix.to_csv(index=True).encode("utf-8"),
                        file_name=("multivariable_ols_beta_matrix.csv"),
                        mime="text/csv",
                        key="multiols_beta_csv_download",
                    )

                # plt.close(heatmap_fig)

    # -----------------------------------------------------
    # 10. Model results and explanations
    # -----------------------------------------------------
    st.markdown("### Model results and interpretation")
    with st.expander("Each outcome model is shown separately with equation, " \
                "coefficients, and interpretation.",expanded=False):
        st.markdown(
                "- **Continuous X:** beta is the adjusted expected change in Y "
                "per one-unit increase in X.\n"
                "- **Binary X coded 0/1:** beta is the adjusted difference in Y "
                "for X=1 versus X=0.\n"
                "- **Categorical X:** each coefficient is relative to the "
                "statsmodels treatment reference category.\n"
                "- **Ordinal X as numeric:** beta represents the adjusted change "
                "in Y per one-level increase in the ordinal score.\n"
                "- **95% CI:** intervals excluding zero are consistent with a "
                "two-sided p-value below 0.05 under the selected model and "
                "standard-error specification.\n"
                "- **Heatmap:** cells show adjusted beta coefficients. Do not "
                "compare unstandardized beta magnitudes across variables with "
                "different measurement units."
            )
        for outcome in fitted_outcomes:
            is_single_outcome = len(fitted_outcomes) == 1

            with st.expander(f"{outcome} ~ equation, coefficients, and interpretation",expanded=is_single_outcome,):
                _render_explanation(explanations[outcome],outcome,)
                coefficient_table = explanations[outcome].get("coefficient_table")

                if coefficient_table is not None:
                    result_csv = coefficient_table.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label=(f"Download {outcome} coefficient table ""(.csv)"),
                        data=result_csv,
                        file_name=(f"multivariable_ols_"f"{outcome}_coefficients.csv"),
                        mime="text/csv",
                        key=f"multiols_table_download_{outcome}",
                    )

                with st.expander("Full statsmodels summary",expanded=False,):
                    st.text(results[outcome].summary().as_text())

