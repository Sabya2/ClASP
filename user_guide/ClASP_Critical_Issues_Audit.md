# ClASP Critical Issues Audit

**Review date:** 1 October 2026  
**Scope:** Static review performed while preparing the ClASP user guide  
**Status:** Reconciled against the publication-readiness branch on 3 November 2026

## Reconciliation with the current repository

The historical text below is retained as an audit trail and describes the state
observed at the original review. The current disposition is authoritative:

| Historical finding | Disposition | Current evidence |
|---|---|---|
| Application source was untracked | Fixed before this reconciliation | Commit `6475526` contains the reviewed application; the new work is isolated on `publication-readiness-fixes`. |
| Correlation threshold disconnected | Confirmed, fixed | The slider is passed as `min_abs_corr`; tests check 0, 0.3, and 0.8 with a fixed -1…1 colour scale. |
| Categorical mediation covariates coerced to numeric | Still open | Mediation is not in the supported navigation and was outside the targeted correction set; do not claim it as validated. |
| ICC mislabeled/incorrect | Confirmed, fixed | Responder code now implements and tests ICC(A,1), including the systematic-bias reference value 0.1549295775. |
| Mean-style CI placed around temporal medians | Confirmed, fixed | Both temporal views use a documented participant-level percentile bootstrap CI for the median. |
| Global group automatically selected without a global selector | Still open | Page-specific selectors exist, but a single global group selector remains an interface decision. |
| Group-test selection differed by page | Confirmed, fixed for supported Data Summary and Group Comparison workflows | Both use `run_group_test`; display summary format no longer selects the hypothesis test. |
| Figure export ignored the supplied figure | Confirmed, fixed | `fig.savefig` is used and regression tests distinguish two simultaneous figures. |
| Categorical-summary p-values omitted | Fixed | Grouped categorical summaries include their chi-square p-value. |
| No structured automated tests | Confirmed, fixed for targeted workflows | Synthetic tests cover loading, summaries, tests, correlations, OLS, responder metrics, temporal summaries, exports, and application startup. |

Remaining caveats: the automated suite is targeted rather than a complete clinical
validation; mediation, inactive pages, browser-level interaction, and deployment
privacy/security controls remain unvalidated. The DOCX, ODT, and PPTX guides were
not regenerated and should be treated as historical artifacts until the first tag.

## Executive summary

The manual-building process exposed several material reproducibility, implementation, and statistical-method issues. The most urgent problem is that most application files are not tracked by Git, meaning the displayed commit does not reproduce the inspected application. Several statistical features also need correction or independent validation before their results should be treated as publication-ready.

| Severity | Issue | Main consequence |
|---|---|---|
| Critical | Most application files are untracked in Git | The documented commit cannot reproduce the inspected application |
| High | Correlation threshold control is disconnected from heatmap masking | The title and color scale can misrepresent what was filtered |
| High | Categorical mediation covariates are coerced to numeric | Text categories can become missing and eliminate observations |
| High | Responder ICC is labeled as absolute agreement without a clearly corresponding implementation | Reliability estimates may be mislabeled or incorrect |
| High | Temporal confidence intervals use mean-based SEM around displayed medians | Error bars are not valid confidence intervals for the plotted statistic |
| Medium–high | The global group variable is automatically selected but cannot be changed globally | Analyses can silently use the wrong grouping field |
| Medium | Statistical test selection differs between pages | Identical data can receive different tests depending on the page |
| Medium | Figure export ignores the supplied figure object | Downloads can contain the wrong or an empty plot |
| Medium | Categorical-summary p-values are calculated and then omitted | Users cannot see the comparison result |
| Medium | There is no structured automated test suite | Statistical regressions can enter the application unnoticed |

## 1. Critical reproducibility problem

During the review, `git status --short` reported:

```text
 D LICENSE
?? .gitignore
?? .streamlit/
?? README.md
?? app.py
?? pages/
?? public_app.py
?? requirements.txt
?? srcs/
?? test.ipynb
?? user_guide/
```

Almost the entire application is untracked. Consequently, commit `9364a92` does not identify the application code that was inspected while writing the manual.

### Consequences

- Another user cannot reproduce the documented software from the commit hash.
- Changes cannot be reliably reviewed or compared.
- The manual's statement that it describes application snapshot `9364a92` is not sufficiently accurate.
- A deployment may differ from the local files without a traceable history.
- The deleted or missing license creates a distribution and reuse concern.

### Recommended action

1. Review all untracked files and remove generated or sensitive files that should not be committed.
2. Restore or deliberately replace the license.
3. Commit the actual application source, requirements, configuration, and tests.
4. Create a release tag.
5. Regenerate the manual against that tagged release.

## 2. Correlation threshold is functionally disconnected

The heatmap function accepts `min_abs_corr` and uses it to create a threshold mask in [`correlation_plots.py`](../srcs/utils_visualisation/correlation_plots.py):

```python
def plot_correlation_heatmap_with_significance_plotly(
    df,
    title="Correlation Heatmap",
    method="spearman",
    min_abs_corr=0.0,
    ...
):
    ...
    threshold_mask = corr.abs() < min_abs_corr
```

However, the calls around lines 320 and 352 do not pass the slider's selected value. The function therefore retains its default threshold of `0.0`, regardless of the user's selection.

At the same time, the session-state slider value is used for:

- `zmin`
- `zmax`
- The heatmap title claiming that a threshold was applied

### Consequences

- A title such as `|r| ≥ 0.50` can be displayed while weaker cells remain present.
- The slider changes color saturation rather than reliably filtering cells.
- Selecting `0.0` can produce `zmin = zmax = 0`, creating a degenerate color scale.
- The current manual's description of the threshold behavior is incorrect.

### Recommended action

Pass the value explicitly:

```python
fig = plot_correlation_heatmap_with_significance_plotly(
    corr_df,
    method=method,
    min_abs_corr=min_corr_to_plot,
    ...
)
```

The display threshold and color-scale range should preferably be separate controls. Automated tests should verify the number and location of masked cells at several thresholds.

## 3. Categorical mediation covariates are likely unusable

The mediation interface offers categorical covariates, but the selected model data are subsequently converted entirely to numeric in [`mediation_plots.py`](../srcs/utils_visualisation/mediation_plots.py):

```python
cov_categorical = st.multiselect(
    "Categorical covariates",
    ...
)

covariates = cov_continuous + cov_categorical
...
df_combo = df_combo.apply(pd.to_numeric, errors="coerce").dropna()
```

A string category such as `Female`, `Control`, or `Responder` is converted to `NaN`, after which the affected row is removed.

### Consequences

- A categorical covariate can reduce the complete-case dataset to zero rows.
- Users may believe the model adjusted for a categorical factor when it did not run correctly.
- Reference categories and contrasts are not explicit.

### Recommended action

- Apply explicit dummy or treatment coding to categorical covariates.
- Display the reference category for every categorical variable.
- Report the encoded design matrix and model-specific sample size.
- Test string, Boolean, binary numeric, and multi-level categorical covariates.

## 4. Responder ICC requires statistical validation

[`responder_analysis.py`](../srcs/utils_features/responder_analysis.py) describes its calculation as:

> ICC(3,1), two-way mixed-effects, absolute-agreement, single-measurement model

The terminology is questionable. ICC(3,1) is commonly associated with a consistency formulation, while an absolute-agreement ICC must account for systematic measurement-session or rater differences. The implemented error calculation does not clearly reproduce a standard absolute-agreement ANOVA decomposition.

### Consequences

- The displayed reliability estimate may be mislabeled.
- Systematic test–retest bias may not be handled as users expect from an absolute-agreement ICC.
- Responder confidence intervals and classifications may be affected if this estimate influences error selection or interpretation.

### Recommended action

- Define the intended ICC model using a recognized notation such as ICC(C,1) or ICC(A,1).
- Validate results against a recognized implementation such as `pingouin.intraclass_corr` or an established R package.
- Use simulated datasets with known agreement, systematic bias, and random noise.
- Report the model, unit, type, and agreement definition in the interface and downloads.

Until validated, the output should not be presented simply as an absolute-agreement ICC.

## 5. Molecular temporal confidence intervals do not match the displayed median

The temporal-molecular page aggregates the median but calculates uncertainty using:

```python
sem = std / sqrt(n)
ci = 1.96 * sem
```

See [`temporal_molecular.py`](../srcs/utils_visualisation/temporal_molecular.py).

That formula approximates uncertainty around a mean under particular assumptions. Centering it on a median does not produce a valid confidence interval for the median.

### Recommended action

Choose one internally consistent presentation:

- Mean with SD, SEM, or a mean confidence interval;
- Median with Q1–Q3 or IQR; or
- Median with a bootstrap confidence interval.

The page should also state that it is descriptive visualization, not a repeated-measures or mixed-effects analysis. Within-participant dependence is not modeled by these error bars.

## 6. The global grouping variable can silently be wrong

[`common.py`](../srcs/utils_features/common.py) automatically chooses a group variable using preferred column names:

```python
priority_names = [
    "AgeGroup", "group_label", "group", "Group", "GROUP", "Cohort",
    "status", "chohort_exp_age_group", "sex", "Group_biologicalAge",
]
```

If none matches, it selects the first object-like or low-cardinality column. The global sidebar provides participant-ID, continuous-column, and categorical-column selectors, but no global group-column selector.

### Consequences

- A dataset containing `sex`, `status`, `responder`, and `AgeGroup` can silently use an unintended field.
- Page-specific group overrides are inconsistent.
- Pages without an override can analyze or display the wrong grouping variable.

### Recommended action

Add a visible global group selector with `None` as an option. Show the active group prominently on every group-dependent page and include it in every exported result.

## 7. Group-testing logic is inconsistent

### Data Summary

In [`data_description.py`](../srcs/utils_features/data_description.py), the inferential test is selected from the requested display format:

- `Mean ± SD` selects one-way ANOVA.
- `Median + IQR` selects Kruskal–Wallis.

The user's preferred descriptive display should not, by itself, determine the inferential test.

### Group Comparison

In [`common.py`](../srcs/utils_visualisation/common.py), Group Comparison uses Shapiro–Wilk results to route between parametric and non-parametric tests.

Additional concerns include:

- Levene's test is calculated for three or more groups but does not prevent ordinary ANOVA when variances differ.
- For two groups, Levene's test is not calculated; normally distributed data default to Welch's t-test but are labeled only as `t-test`.
- `force_parametric=True` appears to route into non-parametric branches, contrary to its name.
- Automatic normality pretesting is fragile, particularly with very small or large samples.
- Pairwise corrections do not address multiplicity across multiple outcome variables.

### Recommended action

- Use one documented testing policy across pages.
- Distinguish Student's, Welch's, and non-parametric tests by name.
- Use Welch ANOVA or an appropriate robust alternative when variances differ.
- Treat assumption plots and scientific design as primary, rather than relying only on normality pretests.
- Add false-discovery-rate control across multiple outcomes where appropriate.

## 8. Figure export may save the wrong figure

[`get_figure_download()`](../srcs/utils_visualisation/common.py) receives a figure but ignores it:

```python
def get_figure_download(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    return buf
```

`plt.savefig()` saves the currently active Matplotlib figure, which is not guaranteed to be the supplied `fig` when multiple figures exist.

### Recommended implementation

```python
def get_figure_download(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf
```

Downloads should be tested for correct dimensions, non-empty content, and correspondence with the displayed plot.

## 9. Categorical-summary p-values are calculated and then removed

[`data_description.py`](../srcs/utils_features/data_description.py) calculates a chi-square p-value and adds it to the first category row:

```python
row["p_value"] = format_p_value(pval) if idx == 0 else ""
```

It then excludes that field from the returned columns:

```python
column_order = [
    "variable", "category", "subcategory", "Full cohort", ...
]  # + ["p_value"]
```

### Consequence

The comparison is calculated but never displayed or exported.

### Recommended action

Restore the p-value column, report the chi-square statistic and degrees of freedom, and warn when expected cell counts make the chi-square approximation unreliable. Consider Fisher's exact or an appropriate alternative for sparse tables.

## 10. Vascular temporal analysis uses unnecessarily broad complete-case deletion

[`temporal_vascular.py`](../srcs/utils_visualisation/temporal_vascular.py) applies:

```python
filtered_df = df[[
    group_col,
    *vascular_age_y_cols,
    *vascular_fitness_y_cols,
]].dropna()
```

Selecting several markers removes a participant from every plot if any selected marker is missing—even when that marker is not needed for a specific separate plot.

### Recommended action

Apply complete-case filtering separately for each plotted combination and display the effective sample size on every plot.

## 11. Navigation contains incomplete or unreachable features

[`public_app.py`](../public_app.py) registers only a subset of files from `pages/`. As a result:

- The PCA page exists but is only a placeholder.
- UMAP code exists but is not registered in the active navigation.
- A standalone Box/Violin page exists but is not registered.

This is not necessarily a defect if those pages are intentionally experimental, but they should be clearly marked, removed from release builds, or placed behind an explicit experimental-features switch.

## 12. Multiple-testing limitations

Correlation heatmaps display significance markers for many pairwise tests without matrix-wide multiplicity correction. For example, an 18-variable heatmap contains 153 distinct variable pairs. At an unadjusted 0.05 threshold, false-positive findings are expected even when all null hypotheses are true.

Similarly, running group comparisons or regressions across many outcomes requires correction across outcomes, not merely within each variable's post-hoc comparisons.

### Recommended action

- Provide Benjamini–Hochberg FDR-adjusted q-values for exploratory high-dimensional analyses.
- Let users download coefficient, raw-p, adjusted-p, and pairwise-n matrices.
- Distinguish exploratory from prespecified tests.

## 13. Testing and release-quality gaps

No structured automated test suite was found. `test.ipynb` is not a substitute for repeatable unit, integration, and statistical-reference tests.

Priority tests should cover:

1. CSV and Excel ingestion.
2. Variable-type and group inference.
3. Correlation coefficients, p-values, masking, and downloads.
4. Student/Welch t-tests, ANOVA/Welch ANOVA, Kruskal–Wallis, Dunn tests, and multiplicity corrections.
5. OLS coefficients, categorical coding, HC3 intervals, and missing-data handling.
6. Mediation with continuous and categorical covariates.
7. Responder TE, SEM, SWC, confidence intervals, ICC, and classifications.
8. Temporal reshaping, normalization, summary statistics, and uncertainty intervals.
9. Figure-export identity and non-empty output.
10. End-to-end Streamlit smoke tests with a small synthetic dataset.

## 14. Documentation and interface quality issues

Additional issues observed during the manual review include:

- Product naming alternates between `CLASP` and `ClASPx`.
- Some interface text contains unfinished citation placeholders such as `[web:48]`.
- Some README sections are commented out despite describing important functionality.
- Statistical terminology is occasionally imprecise, such as generic `t-test` labeling.
- Several methods need clearer disclosure of sample-size rules, missing-data handling, units, reference categories, and multiplicity treatment.

## Recommended remediation order

1. Commit and tag the actual source code.
2. Restore or replace the license deliberately.
3. Fix the correlation threshold and add tests.
4. Correct categorical mediation handling.
5. Validate or replace the ICC implementation.
6. Correct temporal uncertainty intervals.
7. Add a visible global group-column selector.
8. Standardize group-test selection and reporting.
9. Fix figure export and categorical p-value output.
10. Add automated statistical-reference and Streamlit smoke tests.
11. Resolve experimental page/navigation status.
12. Regenerate the DOCX, ODT, and PPTX manuals against a tagged release.

## Effect on the current user guide

The existing manual is visually usable, but it should be considered a draft because:

- Its commit-based version statement is not reproducible while the application source remains untracked.
- Its description of the correlation threshold follows the intended behavior rather than the current disconnected implementation.
- Mediation guidance assumes categorical covariates function as presented in the interface.
- ICC and temporal-confidence-interval descriptions require stronger qualification.

After the application is fixed and committed, the manual should be updated and regenerated with a release tag, revision history, and verification date.

