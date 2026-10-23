
"""Render OLS regression analysis from shared application state."""

import streamlit as st

from srcs.utils_features.common import get_state
from srcs.utils_visualisation.render_ols_page import render_ols_page
from srcs.utils_visualisation.regression_analysis import render_interactive_linear_plot


state = get_state()
df = state["df"]
group_col = state["group_col"]
continuous_cols = state["continuous_cols"]
categorical_cols = state["categorical_cols"]
participants_col = state["participants_col"]

# --- Page overview and how to use ---
st.markdown("## Regression Analysis")


with st.expander("How this page works and what to expect", expanded=False):
    st.markdown(
        r"""

        ### 1. OLS regression analysis

        This section has two modes:

        #### Univariate OLS

        - **What it does**: Fits one simple OLS model at a time (Y ~ X) for each selected X–Y pair.
        - **Output**:
          - Scatterplot with optional group stratification and regression line.
          - Expandable panel with:
            - Regression equation.
            - Coefficient table (beta, SE, 95% CI, p-value).
            - Interpretation sentence for the slope.
            - Summary of statistical significance.
        - **When to use**: To understand crude (unadjusted) associations before adding covariates.
        - **Options**:
                  - *Plot by group*: splits points and regression lines by a categorical variable (e.g. diagnostic group, sex).
                  - *Regression line*: overlays an OLS fit line for each group or for the overall sample.
                  - *Grid columns*: controls how many plots appear per row.
                - **Output**:
                  - Visual inspection of linearity, outliers, and group differences.
                  - Expandable panels with:
                    - Pearson/Spearman correlation (r, p-value).
                    - Univariate OLS results (beta, intercept, R, R, RMSE).
                - **When to use**: Quick exploratory check of bivariate relationships before formal modeling.


        #### Multivariable OLS

        - **What it does**: Fits multiple linear regression models of the form:
          $$
          Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k + \varepsilon
          $$
          for each selected outcome Y, using the same set of predictors X.
        - **Inputs**:
          - One or more outcome variables (Y).
          - One or more predictor / adjustment variables (X).
          - Optional: mark some X variables as categorical (treatment-coded).
          - Optional: use HC3 robust standard errors.
        - **Output**:
          - For **one outcome**: forest plot of adjusted beta coefficients with 95% CIs.
          - For **multiple outcomes**: heatmap of beta coefficients (outcomes × predictors), with significance stars.
          - For each outcome:
            - Regression equation.
            - Model fit statistics (n, R, adjusted R, model p-value).
            - Coefficient table (beta, SE, 95% CI, p-value, significance).
            - Plain-language interpretation of each coefficient, conditional on all other predictors.
            - Summary of statistically significant adjusted terms.
            - Full statsmodels summary (expandable).
            - Downloadable CSV of coefficients.
        - **Interpretation**:
          - **Continuous X**: beta = adjusted expected change in Y per one-unit increase in X.
          - **Binary X (0/1)**: beta = adjusted difference in Y for X=1 vs X=0.
          - **Categorical X**: each displayed contrast is relative to the reference category used by statsmodels.
          - Coefficients are *adjusted associations*, not necessarily causal effects.
        - **When to use**:
          - To estimate associations between predictors and outcomes while controlling for confounders.
          - To compare how the same set of predictors relates to multiple outcomes.

        ---

        ### 2. Interactive pair plot

        - **What it does**: Focuses on a single X–Y pair at a time, with interactive controls and detailed OLS output.
        - **Options**:
          - Select X and Y from dropdowns.
          - Optional group stratification.
          - Toggle regression line.
        - **Output**:
          - One high-resolution scatterplot.
          - Expandable panel with:
            - Correlation statistics.
            - Full univariate OLS model summary (beta, SE, t, p, R, intercept, RMSE).
            - Interpretation of the slope in plain language.
        - **When to use**: When you want to inspect one relationship in detail, including residuals and model diagnostics.



        ### General notes

        - All models use complete-case analysis (rows with missing values in any selected variable are dropped).
        - For categorical predictors, treatment coding is used (one reference category, contrasts for others).
        - Robust (HC3) standard errors change SEs, CIs, and p-values, but not the beta coefficients.
        - Always check:
          - Linearity and residual patterns (via diagnostics or residual plots).
          - Influential observations and multicollinearity if you plan to interpret coefficients causally.
        """
    )



# --- 1. OLS regression analysis (univariate + multivariable) ---
if df is not None and len(continuous_cols) >= 2:
    render_ols_page(
        df=df,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        group_col=group_col,
    )
else:
    st.info("Upload data and select at least two continuous variables.")

# --- 2. Interactive pair plot ---
st.markdown("---")

title_col, toggle_col = st.columns([1, 1.5], gap="xxsmall", vertical_alignment="bottom")
with title_col:
    st.markdown("### Interactive pair plot")

with toggle_col:
    interactive_pair = st.checkbox(
        "Enable",
        value=False,
        key="interactive_pair",
        help="Toggle to enable interactive linear plot (scatter with OLS) for continuous variables.",
    )

if interactive_pair and df is not None and len(continuous_cols) >= 2:
    render_interactive_linear_plot(
        df=df,
        continuous_cols=continuous_cols,
        group_col=group_col,
        participants_col=participants_col,
    )
