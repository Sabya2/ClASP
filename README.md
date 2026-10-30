# ClASPx

## Clinical and Statistical Analysis Platform eXplorer

ClASPx is a Streamlit-based platform for exploratory clinical and statistical data analysis. It is designed to help users inspect relationships between variables, fit statistical models, and evaluate individual responses to an intervention.

> **Important:** ClASPx supports statistical analysis but does not replace careful study design, domain expertise, diagnostic checks, or independent statistical review. Results should be interpreted in the context of the study design and data quality.

## Quick start

1. Prepare a CSV or Excel file using the input requirements below and upload the file in ClASPx.
2. Check that the columns have been detected correctly.
3. Select the variables and analysis options on the relevant page.
4. Download results where available.

## Input data with Minimum structure

ClASPx accepts:

- CSV files (`.csv`).
- Excel files (`.xlsx` or `.xls`).

A typical dataset should `first row` with column name and each subsequent row should represent one participant or observation.
Column names should be clear and unique. Avoid using the same name for two columns.

- Column names should contain only letters, numbers, and underscores (`_`). Avoid spaces, leading or trailing whitespace, and other special characters, as they can interfere with the analysis code and cause the application to fail. The **Data description** page displays a red error when invalid names are detected; expand **View invalid column names** to see the affected columns.
- One optional participant identifier column, such as `participant_id`.
- One optional grouping column, such as `group`, `sex`, `age_type`, or `diagnosis`.
- Numeric columns containing measurements for statistical analyses.
- Optional pre-intervention, post-intervention, and reliability columns for responder analysis.

### Example CSV/Excel input

The following small table illustrates the expected structure:


| participant_id | group   | age | feature1_pre | feature1_post | feature2 | feature3 | Sex | feature4 |
|----------------|---------|-----|---------|----------|---------|----------|--------|--------|
| P001           | Control | 45  | 32.5    | 33.1     | 138.0   | 134.0    | 0    | 3.5    |
| P002           | Control | 52  | 28.0    | 27.6     | 145.0   | 146.0    | 1    | 3.5    |
| P003           | Training| 41  | 30.2    | 33.8     | 142.0   | 135.0    | 0    | 3.5    |
| P004           | Training| 49  | 26.5    | 29.0     | 151.0   | 143.0    | 1    | 3.5    |
| P005           | Control | 38  | 35.1    | 35.7     | 128.0   | 126.0    | 1    | 3.5    |

This example contains:

- `participant_id`: participant identifier. 
- `group`: categorical grouping variable. 
- `age`: numeric participant characteristic.
- `Sex`: numeric participant characteristic.
- `feature1_pre` and `feature1_post`: pre- and post-intervention measurements.
- `feature1, feature2, feature3, feature4`: More measurements.

### Numeric formatting

Numeric values must be stored as numbers, not as text. Use a **period (`.`) as the decimal separator**:
Do not use commas as decimal separators in the uploaded data. In a CSV file, commas normally separate columns, so a value such as `32,5` may be interpreted as two separate fields or as text.
Thousands separators should preferably be removed. Do not include units, currency symbols, or other text inside numeric cells:

```text
Correct: 32.5
Incorrect for the standard input format: 32,5
    -> Hence convert 32,5  →  32.5
Avoid: 138 mmHg
Avoid: €1,234.50
```

Missing values should be left blank or represented consistently. Do not mix several missing-value codes such as `?`, `missing`, `-`, and `N/A` in the same column.

### CSV recommendations

- Save the file as UTF-8 CSV where possible.
- Use commas to separate columns; Keep the first row as the header.
- Ensure every data row has the same number of columns.
- Check that decimal values use periods and not comma(,).
- Remove duplicate participant IDs unless repeated measurements are intentional.
- Incase of `Excel` avoid merged cells, title rows above the header, and multiple tables on one worksheet.
- Do not upload names, addresses, patient numbers, or other directly identifying information unless this is explicitly permitted by your governance and data-protection procedures.

<!-- ### Excel recommendations

For Excel files:

- Avoid merged cells, title rows above the header, and multiple tables on one worksheet.
- Store numeric measurements as numeric cells rather than formatted text.
- Remove formulas or confirm that the uploaded workbook contains the intended calculated values. -->

## Missing data and data quality

Before analysis, check the **Data description** section:

- Whether the column-name validation reports spaces, leading or trailing whitespace, or other unsupported characters.
- Whether numeric columns were imported as numeric values.
- Whether decimal separators were interpreted correctly.
- Whether participant IDs are present and unique where required.
- Whether group labels are consistent, for example `Training` versus `training`.
- Whether there are enough observations for each selected analysis or subgroup.
- Whether extreme values and outliers are plausible.

ClASPx may exclude incomplete rows for a specific analysis. Always review the reported sample size and the data-processing information shown with the results.


## Correlation page

The Correlation page calculates correlation matrices and displays them as interactive heatmaps.
A correlation describes an association, not necessarily a causal relationship. Pearson correlation is most directly associated with linear relationships; Spearman and Kendall are rank-based methods for monotonic relationships.

<!-- ### What to provide

- Two or more numeric variables.
- An optional categorical grouping variable.

### What you can do

- Select the numeric variables to include.
- Choose Pearson, Spearman, or Kendall correlation.
- Set a minimum absolute-correlation threshold to hide weaker cells visually.
- Optionally create a separate heatmap for each group.

### What to expect

- A square correlation matrix.
- An interactive heatmap showing the direction and strength of associations.
- Hover information containing the variable pair, correlation coefficient, and p-value where available.
- Group-specific heatmaps when grouped analysis is selected.

Selected columns are converted to numeric values before analysis. Correlations are calculated pairwise using available values. Variables or groups with insufficient valid data may be skipped.

A correlation describes an association, not necessarily a causal relationship. Pearson correlation is most directly associated with linear relationships; Spearman and Kendall are rank-based methods for monotonic relationships. -->

## OLS regression page

The OLS Regression page provides three complementary views of linear relationships:
```text
(Univariate) Y = intercept + beta × X + error
(Multivariate) Y = beta0 + beta1 X1 + beta2 X2 + ... + betak Xk + error
```

1. Univariate OLS regression.
2. Multivariable OLS regression.
3. An interactive pair plot for one selected X–Y relationship.

<!-- ### What to provide

- At least one numeric outcome variable (`Y`).
- At least one numeric predictor variable (`X`).
- Optional additional predictors or adjustment variables.
- Optional categorical predictors.
- Optional grouping variable for plots.

### Univariate OLS

Univariate OLS fits one model for each selected outcome–predictor pair:

```text
Y = intercept + beta × X + error
```

The page provides a scatterplot, optional regression line, coefficient table, confidence interval, p-value, model statistics, and a plain-language interpretation of the slope.

Use this view to inspect crude, unadjusted associations before building a multivariable model.

### Multivariable OLS

Multivariable OLS fits models of the form:

```text
Y = beta0 + beta1 X1 + beta2 X2 + ... + betak Xk + error
```

The page provides adjusted coefficients, confidence intervals, p-values, model-fit statistics, forest plots or coefficient heatmaps, and downloadable coefficient tables.

An adjusted coefficient describes the association between a predictor and the outcome while holding the other selected predictors constant. It should not automatically be interpreted as a causal effect.

### Categorical predictors

Categorical predictors use treatment coding. Each displayed category is compared with a reference category selected by the statistical software. For example, if `Control` is the reference group, the coefficient for `Training` represents the adjusted difference between Training and Control, conditional on the other predictors.

### Robust standard errors

The HC3 option changes standard errors, confidence intervals, and p-values. It does not change the estimated beta coefficients.

### Complete-case analysis

For each model, rows with missing values in any selected outcome or predictor are excluded from that model. The number of observations can therefore differ between models.

Always inspect linearity, residual patterns, influential observations, and multicollinearity before drawing strong conclusions from an OLS model. -->

## Responder Analysis page

The Responder Analysis page evaluates individual responses to an intervention using the Smallest Worthwhile Change (SWC) and Typical Error (TE).
```text
feature1_pre, feature1_post
```

<!-- ### What to provide

- One participant identifier column.
- Pre-intervention measurement columns.
- Post-intervention measurement columns.
- Reliability measurements used to estimate typical error.
- Optional grouping variable.

The pre- and post-intervention variables must correspond to the same outcome. For example:

```text
feature1_pre  and  feature1_post
```

### What you can select

- Pre/post variable pairs.
- Reliability inputs.
- SWC multiplier, commonly `0.2` for a small effect.
- Confidence level: 68%, 90%, 95%, or 99%.
- Optional group stratification.

### What to expect

Participants are classified as:

- **Reliable meaningful responders:** the lower confidence-interval bound exceeds the SWC threshold.
- **Potential responders:** the point estimate exceeds the SWC, but the confidence interval includes non-response.
- **Non-responders:** no meaningful change is detected.

The page provides individual classifications, responder plots, and downloadable results where available. If a group or participant does not contain sufficient valid information, the result may be omitted.

Responder classifications depend on the selected SWC, typical error, confidence level, and measurement quality. They should be interpreted as statistical classifications rather than definitive biological labels. -->


## Reproducibility and citation

Record the following when using ClASPx:

- ClASPx version.
- Date of analysis.
- Input-data preparation steps.
- Selected variables.
- Statistical method and options.
- Grouping variable, if used.
- Downloaded output files.
- DOI of the specific ClASPx release, when available.

For a published study, cite the exact software version used rather than only the project homepage.

## Feedback

Use the **Send feedback** link in the application to report a problem or suggest an improvement. Do not include participant names, patient identifiers, clinical records, or other confidential information in feedback messages.

## Privacy and security

Do not upload identifiable clinical data unless the use of the application is approved for that data. Prefer pseudonymous participant identifiers and synthetic or de-identified datasets for demonstrations. Review the ClASPx privacy notice for information about login data, uploaded files, usage tracking, and retention.

## Support

When reporting an issue, include:

- ClASPx version.
- Analysis page.
- File type and approximate file size.
- The selected analysis options.
- The exact error message.
- A small synthetic example that reproduces the problem, if possible.

Never send confidential or identifiable participant data when requesting support.
