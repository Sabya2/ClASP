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
    _fit_ols_model,
    _model_terms_table,
)


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
