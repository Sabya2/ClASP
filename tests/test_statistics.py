import numpy as np
import pandas as pd
import pytest
import scikit_posthocs as sp

from srcs.utils_features.data_description import build_continuous_summary_table
from srcs.utils_features.responder_analysis import calculate_swc_te, icc_31_from_two_timepoints
from srcs.utils_visualisation.common import run_group_test
from srcs.utils_visualisation.temporal_stats import summarize_median_trajectory


def test_icc_absolute_agreement_reference_cases():
    t1 = np.arange(1.0, 11.0)
    perfect = pd.DataFrame({"t1": t1, "t2": t1})
    biased = pd.DataFrame({"t1": t1, "t2": t1 + 10})
    missing = pd.concat([biased, pd.DataFrame({"t1": [np.nan], "t2": [4]})], ignore_index=True)
    assert icc_31_from_two_timepoints(perfect, "t1", "t2")[0] == pytest.approx(1.0)
    assert icc_31_from_two_timepoints(biased, "t1", "t2")[0] == pytest.approx(0.1549295775)
    assert icc_31_from_two_timepoints(missing, "t1", "t2")[0] == pytest.approx(0.1549295775)
    with pytest.raises(ValueError, match="undefined"):
        icc_31_from_two_timepoints(pd.DataFrame({"t1": [1, 1], "t2": [1, 1]}), "t1", "t2")


def test_responder_uses_group_retest_data_and_supplied_icc():
    df = pd.DataFrame({
        "id": range(8), "group": ["a"] * 4 + ["b"] * 4,
        "pre": [10, 11, 12, 13, 20, 22, 24, 26],
        "post": [12, 13, 14, 15, 21, 23, 25, 27],
        "t1": [1, 2, 3, 4, 10, 20, 30, 40],
        "t2": [1, 2.1, 2.9, 4, 12, 18, 34, 36],
    })
    summary, individual, _, _ = calculate_swc_te(
        df, "id", "pre", "post", "t1", "t2", group_col="group",
        threshold_type="Zero-Based", error_metric="SEM", icc_value=0.75,
    )
    assert set(summary["icc_a1"]) == {0.75}
    np.testing.assert_allclose(summary["sem_change"], np.sqrt(2) * summary["sem_single_observation"])
    assert individual["interval_margin"].notna().all()


@pytest.mark.parametrize("group_count, expected_pairs", [(3, 3), (4, 6)])
def test_dunn_unique_pairs_and_reference(group_count, expected_pairs):
    df = pd.DataFrame({
        "group": np.repeat(list("ABCD")[:group_count], 5),
        "value": np.concatenate([np.arange(5) + i * 3 for i in range(group_count)]),
    })
    result = run_group_test(df, "group", "value", method="nonparametric", correction_method="holm")
    assert len(result["pairwise"]) == expected_pairs
    assert len({(r["group1"], r["group2"]) for r in result["pairwise"]}) == expected_pairs
    reference = sp.posthoc_dunn(df, val_col="value", group_col="group", p_adjust="holm")
    for row in result["pairwise"]:
        assert row["p_corrected"] == pytest.approx(reference.loc[row["group1"], row["group2"]])


def test_dunn_failure_is_explicit_and_never_mixes_methods(monkeypatch):
    df = pd.DataFrame({"group": np.repeat(list("ABC"), 3), "value": range(9)})
    monkeypatch.setattr(sp, "posthoc_dunn", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="no fallback"):
        run_group_test(df, "group", "value", method="nonparametric")


def test_force_parametric_means_parametric_and_summary_policy_is_independent_of_format():
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "y": [0, 0, 0, 0, 10, 2, 3, 4, 5, 6]})
    result = run_group_test(df, "g", "y", force_parametric=True)
    assert "t-test" in result["overall_test"]
    mean_table = build_continuous_summary_table(df, ["y"], "g", True, summary_format="mean_sd")
    median_table = build_continuous_summary_table(df, ["y"], "g", True, summary_format="median_iqr")
    assert mean_table.loc[0, "test"] == median_table.loc[0, "test"]
    assert mean_table.loc[0, "p_value"] == median_table.loc[0, "p_value"]


def test_constant_group_data_are_reported_explicitly():
    df = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "y": [4.0] * 6})
    result = run_group_test(df, "g", "y")
    assert result["overall_test"] == "n/a (all values constant)"
    assert result["pairwise"] == []
    assert "no group test" in result["data_warning"]


def test_temporal_bootstrap_edge_cases_and_participant_resampling():
    df = pd.DataFrame({
        "family": ["x"] * 7, "time": [1, 1, 1, 2, 2, 3, 4],
        "id": [1, 1, 2, 1, 2, 1, 1], "Value": [1, 3, 5, 7, 7, 9, np.nan],
    })
    out = summarize_median_trajectory(df, ["family", "time"], participant_col="id", n_boot=500)
    time1 = out.loc[out.time == 1].iloc[0]
    # Duplicate ID 1 becomes median 2, so participant-level median is (2+5)/2.
    assert time1["median"] == pytest.approx(3.5)
    constant = out.loc[out.time == 2].iloc[0]
    singleton = out.loc[out.time == 3].iloc[0]
    assert constant["ci_low"] == constant["ci_high"] == 7
    assert singleton["ci_low"] == singleton["ci_high"] == 9
    assert 4 not in set(out.time)
