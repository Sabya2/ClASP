import numpy as np
import pandas as pd
import pytest
import scikit_posthocs as sp

from srcs.utils_features.data_description import build_continuous_summary_table
from srcs.utils_features.responder_analysis import calculate_swc_te, icc_31_from_two_timepoints
from srcs.utils_visualisation.common import run_group_test
from srcs.utils_visualisation.group_comparison_plots import (
    build_group_method_explanation,
    format_group_summary,
)
from srcs.utils_visualisation.repeated_measures_stats import (
    RepeatedMeasuresValidationError,
    run_mixed_anova,
    run_repeated_measures,
    wide_to_long,
)
from srcs.utils_visualisation.temporal_stats import (
    build_temporal_explanation,
    safe_temporal_filename,
    summarize_median_trajectory,
)


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


def test_group_summary_explains_automatic_and_welch_selection():
    df = pd.DataFrame({
        "g": ["a"] * 6 + ["b"] * 6,
        "y": [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    })
    result = run_group_test(df, "g", "y", method="parametric")
    explanation = build_group_method_explanation(result)
    assert "Parametric (explicit override)" in explanation
    assert "Levene's variance test" in explanation
    assert "Welch unequal-variance inference" in explanation

    summary = format_group_summary(result, "g", "y")
    assert "Statistical method used" in summary
    assert result["overall_test"] in summary
    assert "independent groups" in summary


def test_group_summary_explains_nonparametric_rank_interpretation():
    df = pd.DataFrame({
        "g": np.repeat(["a", "b", "c"], 4),
        "y": [1, 1, 2, 9, 2, 3, 3, 10, 4, 4, 5, 11],
    })
    result = run_group_test(df, "g", "y", method="nonparametric")
    explanation = build_group_method_explanation(result)
    assert "Non-parametric (explicit override)" in explanation
    assert "compare ranks" in explanation
    assert "Dunn's comparisons" in explanation


def test_two_time_repeated_matches_scipy_paired_t_reference():
    before = np.array([10, 12, 9, 15, 11, 13], dtype=float)
    after = np.array([12, 13, 11, 14, 14, 16], dtype=float)
    df = pd.DataFrame({
        "id": np.repeat(np.arange(6), 2), "time": ["pre", "post"] * 6,
        "value": np.column_stack([before, after]).ravel(),
    })
    result = run_repeated_measures(
        df, "id", "time", "value", ["pre", "post"], method="parametric"
    )
    reference = __import__("scipy").stats.ttest_rel(after, before)
    assert result["n_included"] == 6
    assert result["result"]["statistic"] == pytest.approx(reference.statistic)
    assert result["result"]["p"] == pytest.approx(reference.pvalue)
    assert result["result"]["mean_change"] == pytest.approx(np.mean(after - before))


def test_repeated_measures_reports_incomplete_participants_and_uses_complete_pairs():
    df = pd.DataFrame({
        "id": [1, 1, 2, 2, 3], "time": ["a", "b", "a", "b", "a"],
        "value": [1.0, 2.0, 2.0, 4.0, 9.0],
    })
    result = run_repeated_measures(df, "id", "time", "value", ["a", "b"], method="parametric")
    assert result["n_included"] == 2
    assert result["n_excluded_incomplete"] == 1
    assert result["result"]["mean_change"] == pytest.approx(1.5)


def test_duplicate_participant_time_is_rejected_without_aggregation():
    df = pd.DataFrame({
        "id": [1, 1, 1, 2, 2], "time": ["a", "a", "b", "a", "b"],
        "value": [1, 2, 3, 4, 5],
    })
    with pytest.raises(RepeatedMeasuresValidationError, match="Duplicate participant/time"):
        run_repeated_measures(df, "id", "time", "value")


def test_three_time_repeated_anova_matches_statsmodels_reference():
    from statsmodels.stats.anova import AnovaRM

    values = np.array([
        [8.0, 9.0, 12.0], [7.0, 8.5, 10.0], [9.0, 10.0, 13.0],
        [6.0, 8.0, 9.0], [10.0, 10.5, 14.0], [7.5, 9.5, 11.0],
    ])
    df = pd.DataFrame({
        "id": np.repeat(np.arange(len(values)), 3), "time": np.tile(["t0", "t1", "t2"], len(values)),
        "value": values.ravel(),
    })
    result = run_repeated_measures(df, "id", "time", "value", method="parametric")
    reference = AnovaRM(df, "value", "id", within=["time"]).fit().anova_table.iloc[0]
    assert result["result"]["statistic"] == pytest.approx(reference["F Value"])
    assert result["result"]["p_uncorrected"] == pytest.approx(reference["Pr > F"])
    assert len(result["pairwise"]) == 3
    assert all("p_corrected" in row for row in result["pairwise"])


def test_mixed_anova_detects_known_group_by_time_interaction_with_unequal_groups():
    rows = []
    # Control is stable while treatment increases; deterministic residual
    # variation keeps all ANOVA denominators estimable.
    for group, size, change in [("control", 5, 0.2), ("treatment", 7, 4.0)]:
        for subject in range(size):
            identifier = f"{group}-{subject}"
            baseline = 10 + subject * .4 + (subject % 2) * .15
            rows.extend([
                {"id": identifier, "group": group, "time": "pre", "value": baseline},
                {"id": identifier, "group": group, "time": "post", "value": baseline + change + (subject % 3) * .08},
            ])
    df = pd.DataFrame(rows)
    result = run_mixed_anova(df, "id", "group", "time", "value", ["pre", "post"])
    interaction = next(row for row in result["effects"] if row["effect"] == "Group × Time")
    assert result["group_sizes"] == {"control": 5, "treatment": 7}
    assert interaction["p"] < 0.001
    assert interaction["df1"] == pytest.approx(1)
    assert interaction["df2"] == pytest.approx(10)
    assert result["posthoc"]


def test_mixed_design_rejects_group_switching_and_constant_transformation():
    switching = pd.DataFrame({
        "id": [1, 1, 2, 2], "group": ["a", "b", "a", "a"],
        "time": [0, 1, 0, 1], "value": [1, 2, 3, 4],
    })
    with pytest.raises(RepeatedMeasuresValidationError, match="more than one group"):
        run_mixed_anova(switching, "id", "group", "time", "value")

    constant = pd.DataFrame({
        "id": np.repeat(range(4), 2), "time": [0, 1] * 4, "value": [5.0] * 8,
    })
    with pytest.raises(RepeatedMeasuresValidationError, match="No usable observations"):
        run_repeated_measures(constant, "id", "time", "value", transformation="z-score")


def test_wide_conversion_uses_explicit_columns_and_preserves_missing_values():
    wide = pd.DataFrame({"id": [1, 2], "group": ["a", "b"], "week0": [1.0, 2.0], "week4": [3.0, np.nan]})
    long = wide_to_long(wide, "id", ["week0", "week4"], group_col="group")
    assert list(long.columns) == ["id", "group", "time", "outcome"]
    assert len(long) == 4
    assert long["outcome"].isna().sum() == 1


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


def test_temporal_explanation_matches_baseline_and_interval_settings():
    normalized = build_temporal_explanation("Yes", "95% bootstrap CI", "participant_id")
    assert "value / baseline × 100" in normalized["baseline"]
    assert "participant_id" in normalized["baseline"]
    assert "2,000 participant-level resamples" in normalized["interval"]

    raw_iqr = build_temporal_explanation("No", "IQR")
    assert "imported measurement units" in raw_iqr["baseline"]
    assert "not a confidence interval" in raw_iqr["interval"]
    assert safe_temporal_filename("Blood pressure / arm A") == "Blood_pressure_arm_A"
