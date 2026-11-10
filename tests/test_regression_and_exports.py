import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pytest
from PIL import Image

from srcs.utils_visualisation.common import get_figure_download
from srcs.utils_visualisation.render_ols_page import (
    _build_model_explanation,
    _build_symbolic_equation,
    _forest_interpretation_lines,
    _fit_ols_model,
    _model_terms_table,
    _partial_regression_data,
    _plot_adjusted_scatter,
    _plot_forest_by_predictor,
)


def test_symbolic_equation_maps_each_beta_to_its_predictor():
    equation = _build_symbolic_equation("BP", ["BMI", "Age", "Smoking"])
    assert equation == "BP = β0 + β1×(BMI) + β2×(Age) + β3×(Smoking) + ε"


def test_ols_covariance_checkbox_matches_statsmodels_reference():
    df = pd.DataFrame({"y": [2, 4, 5, 8, 9, 13], "x": [0, 1, 2, 3, 4, 5]})
    conventional, formula = _fit_ols_model(df, "y", ["x"], ["x"], [], False)
    robust, _ = _fit_ols_model(df, "y", ["x"], ["x"], [], True)
    reference_conventional = smf.ols(formula, df).fit(cov_type="nonrobust", use_t=True)
    reference_robust = smf.ols(formula, df).fit(cov_type="HC3", use_t=True)

    np.testing.assert_allclose(conventional.params, robust.params)
    for actual, expected in ((conventional, reference_conventional), (robust, reference_robust)):
        np.testing.assert_allclose(actual.bse, expected.bse)
        np.testing.assert_allclose(actual.pvalues, expected.pvalues)
        np.testing.assert_allclose(actual.conf_int(), expected.conf_int())
        assert len(_model_terms_table(actual)) == 1
    assert conventional.cov_type == "nonrobust"
    assert robust.cov_type == "HC3"


def test_regression_explanations_use_variable_type_and_small_values():
    df = pd.DataFrame({
        "y": [1, 1.000001, 2, 2.000001, 3, 3.000001],
        "x": [0, 1, 0, 1, 0, 1],
        "group": pd.Categorical(["B", "B", "A", "A", "C", "C"], categories=["B", "A", "C"]),
    })
    model, formula = _fit_ols_model(df, "y", ["x", "group"], ["x"], ["group"], False)
    explanation = _build_model_explanation(model, "y", ["x", "group"], ["x"], ["group"], df, formula)
    text = " ".join(explanation["coefficient_text"])
    assert "one-unit increase in x" in text
    assert "group = B (reference)" in text
    assert "1e-06" in text
    assert "causality" in text


def test_grouped_bivariate_models_use_each_subgroup_observations():
    df = pd.DataFrame({
        "group": ["a"] * 4 + ["b"] * 4 + ["invalid"] * 2,
        "x": [0, 1, 2, 3, 0, 1, 2, 3, 1, 1],
        "y": [1, 3, 5, 7, 2, 1, 0, -1, 5, 6],
    })
    expected_slopes = {"a": 2.0, "b": -1.0}
    for group, slope in expected_slopes.items():
        subgroup = df.loc[df.group == group]
        model, formula = _fit_ols_model(
            data=subgroup, outcome="y", predictors=["x"],
            primary_predictors=["x"], categorical_vars=[], robust_se=False,
        )
        explanation = _build_model_explanation(
            model=model, outcome="y", predictors=["x"], additional_covars=[],
            categorical_vars=[], data=subgroup, formula=formula,
        )
        assert model.nobs == 4
        assert explanation["coefficient_table"].iloc[0]["beta"] == pytest.approx(slope)
    invalid = df.loc[df.group == "invalid"]
    assert len(invalid) < 3 or invalid.x.nunique() < 2


def test_separate_adjusted_scatter_slope_matches_multivariable_beta():
    df = pd.DataFrame({
        "x": np.arange(12, dtype=float),
        "z": [0, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6],
    })
    df["y"] = 2.0 + 1.5 * df["x"] - 0.7 * df["z"]
    model, _ = _fit_ols_model(df, "y", ["x", "z"], ["x"], [], False)

    partial_data = _partial_regression_data(model, "x")
    partial_slope = np.polyfit(
        partial_data["Adjusted predictor component"],
        partial_data["Adjusted outcome component"],
        1,
    )[0]
    assert partial_slope == pytest.approx(model.params["x"])

    fig, plotted_data = _plot_adjusted_scatter(model, "y", "x", predictor="x")
    assert fig is not None
    assert len(plotted_data) == len(df)
    assert list(plotted_data.columns) == ["x", "y"]
    np.testing.assert_allclose(plotted_data["x"], df["x"])
    np.testing.assert_allclose(plotted_data["y"], df["y"])
    assert fig.axes[0].get_xlabel() == "x (observed values)"
    assert fig.axes[0].get_ylabel() == "y (observed values)"
    adjusted_line = next(line for line in fig.axes[0].lines if line.get_label() == "Adjusted model estimate")
    assert np.polyfit(adjusted_line.get_xdata(), adjusted_line.get_ydata(), 1)[0] == pytest.approx(model.params["x"])
    plt.close(fig)


def test_combined_forest_plot_uses_clear_outcome_labels():
    coefficients = pd.DataFrame({
        "outcome": ["BP", "HR"],
        "predictor": ["BMI", "BMI"],
        "term": ["BMI", "BMI"],
        "term_display": ["BP: BMI", "HR: BMI"],
        "beta": [1.2, -0.4],
        "ci_low": [0.5, -0.8],
        "ci_high": [1.9, 0.0],
        "p_value": [0.01, 0.05],
    })

    fig = _plot_forest_by_predictor(coefficients, "BMI")
    assert fig.axes[0].get_title() == "BMI across fitted outcomes"
    assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == ["BP", "HR"]
    assert fig.axes[0].get_xlabel() == "Adjusted beta coefficient (95% CI)"
    plt.close(fig)


def test_figure_download_serializes_supplied_figure_and_rewinds():
    fig1, ax1 = plt.subplots()
    ax1.set_title("FIRST UNIQUE TITLE")
    fig2, ax2 = plt.subplots()
    ax2.set_title("SECOND UNIQUE TITLE")
    first = get_figure_download(fig1)
    second = get_figure_download(fig2)
    assert first.tell() == second.tell() == 0
    first_bytes, second_bytes = first.read(), second.read()
    assert first_bytes != second_bytes
    assert Image.open(io.BytesIO(first_bytes)).format == "PNG"
    assert Image.open(io.BytesIO(second_bytes)).format == "PNG"
    plt.close(fig1)
    plt.close(fig2)
