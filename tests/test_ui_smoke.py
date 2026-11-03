from streamlit.testing.v1 import AppTest
import pandas as pd
import pytest


def test_main_application_starts_without_streamlit_exception():
    app = AppTest.from_file("public_app.py").run(timeout=20)
    assert not app.exception


@pytest.mark.parametrize(
    "page",
    [
        "DataDescription.py", "DataDistribution.py", "RegressionAnalysis.py",
        "Correlation.py", "GroupComparison.py", "Temporal-Dynamics.py",
        "Responder.py", "Contact-Request.py",
    ],
)
def test_supported_page_starts_with_synthetic_data(page):
    df = pd.read_csv("tests/data/synthetic_clinical.csv")
    app = AppTest.from_file(f"pages/{page}")
    state = {
        "df": df,
        "continuous_cols": [
            "outcome", "predictor", "pre", "post", "retest1", "retest2",
            "marker_T1", "marker_T2",
        ],
        "categorical_cols": ["group"],
        "group_col": "group",
        "participants_col": "participant_id",
        "temporal_cols": [],
        "temporal_families": {},
        "temporal_summary": None,
    }
    for key, value in state.items():
        app.session_state[key] = value
    app.run(timeout=20)
    assert not app.exception


@pytest.mark.parametrize(
    "mode, expected_subheader",
    [
        ("Within-Family dynamics", "Within-Family Dynamics"),
        ("Cross-Family dynamics", "Cross-Family Dynamics"),
    ],
)
def test_temporal_views_offer_explanation_and_image_download(mode, expected_subheader):
    df = pd.read_csv("tests/data/synthetic_clinical.csv")
    app = AppTest.from_file("pages/Temporal-Dynamics.py")
    state = {
        "df": df,
        "continuous_cols": ["outcome", "marker_T1", "marker_T2"],
        "categorical_cols": ["group"],
        "group_col": "group",
        "participants_col": "participant_id",
        "temporal_cols": [],
        "temporal_families": {},
        "temporal_summary": None,
        "temporal_columns_signature": None,
        "temporal_dynamics_type_v2": mode,
    }
    for key, value in state.items():
        app.session_state[key] = value
    app.run(timeout=20)

    assert not app.exception
    assert expected_subheader in [element.value for element in app.subheader]
    assert "How baseline and interval settings are interpreted" in [
        element.label for element in app.expander
    ]
    assert app.get("download_button")
