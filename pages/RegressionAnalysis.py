
"""Render OLS regression analysis from shared application state."""

import streamlit as st

from srcs.utils_features.common import get_state
from srcs.utils_visualisation.render_ols_page import render_ols_page


state = get_state()
df = state["df"]
group_col = state["group_col"]
continuous_cols = state["continuous_cols"]
categorical_cols = state["categorical_cols"]

# --- Page overview and how to use ---
st.markdown("## Regression Analysis")


with st.expander("How this page works and what to expect", expanded=False):
    st.markdown(
        "Choose **Bivariate Analysis** to examine one predictor–outcome "
        "relationship at a time, or **Multivariate Analysis** to estimate "
        "mutually adjusted associations."
    )

    bivariate_help, multivariable_help = st.columns(
        [1, 1], border=True, gap="medium"
    )

    with bivariate_help:
        st.markdown(
            r"""
            ### Bivariate Analysis

            Fits a separate simple OLS model for every selected X–Y pair:

            $$Y = \beta_0 + \beta_1X + \epsilon$$

            #### How to interpret it

            - $\beta_0$ is the expected value of Y when X = 0.
            - $\beta_1$ is the **unadjusted association** of X with Y: the expected change in Y for a one-unit increase in X.
            - A positive $\beta_1$ indicates higher Y as X increases; a negative $\beta_1$ indicates lower Y.
            - The 95% confidence interval describes the uncertainty around the estimated coefficient. An interval excluding zero corresponds to p < 0.05 for the same two-sided test.
            - R² is the proportion of variation in Y explained by X.

            This mode is useful for exploring crude relationships, linearity, and possible outliers before adding covariates. Plotting by group produces separate visual or within-group relationships; it does not adjust the overall coefficient for group.

            #### What you receive

            - Scatterplots with optional regression lines and group stratification.
            - Fitted equation, coefficient table, 95% CI, p-value, and plain-language interpretation.
            - Optional per-group statistics and downloadable plots.
            """
        )

    with multivariable_help:
        st.markdown(
            r"""
            ### Multivariate Analysis

            Fits each selected outcome using all selected predictors together:

            $$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_kX_k + \epsilon$$

            #### How to interpret it

            Each coefficient is the association of its predictor with Y, **adjusted for (controlling for) every other predictor in the model**.

            For example:

            $$BP = \beta_0 + \beta_1BMI + \beta_2Age + \beta_3Smoking + \epsilon$$

            - $\beta_1$: association of BMI with BP, adjusted for Age and Smoking.
            - $\beta_2$: association of Age with BP, adjusted for BMI and Smoking.
            - $\beta_3$: association of Smoking with BP, adjusted for BMI and Age.
            - For a continuous X, beta is the adjusted expected change in Y per one-unit increase in X.
            - For a binary X, beta is the adjusted difference from its reference category. Other categorical coefficients are also contrasts against a reference category.

            These are adjusted associations and do not, by themselves, establish causality.

            #### What you receive

            - **Separate** plots (default): an adjusted scatterplot for every outcome–primary-predictor pair. Plots are grouped by outcome with up to two predictors per row, and each fitted slope equals the adjusted coefficient from the full model.
            - **Combined** forest plots: one clear summary card per primary predictor, compared across all outcomes, with detailed estimates collapsed underneath.
            - Plot-specific interpretation text in a visible box beneath every plot.
            - An optional outcome-by-predictor beta heatmap.
            - Model-fit statistics, coefficient tables, interpretations, full summaries, and CSV downloads.
            - Optional HC3 robust standard errors, which change SEs, CIs, and p-values but not coefficient estimates.
            """
        )

    st.markdown(
        """
        #### General notes

        All models use complete cases for the variables included in that model. Check linearity, residual patterns, influential observations, and multicollinearity before drawing substantive conclusions.
        """
    )



# --- 1. OLS regression analysis (bivariate + multivariable) ---
if df is not None and len(continuous_cols) >= 2:
    render_ols_page(
        df=df,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        group_col=group_col,
    )
else:
    st.info("Upload data and select at least two continuous variables.")
