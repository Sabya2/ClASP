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
