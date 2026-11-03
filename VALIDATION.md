# Validation record

## Local targeted validation

- Date: 3 November 2026
- Branch: `publication-readiness-fixes`
- Starting commit: `6475526`
- Operating context: Linux, Python 3.12.9
- Command: `MPLCONFIGDIR=/tmp/clasp-mpl .venv/bin/pytest -q`
- Outcome: 25 passed
- Static check: `python -m compileall -q public_app.py pages srcs` passed
- Citation check: `cffconvert --validate` passed against CFF schema 1.2.0

Key package versions used locally:

| Package | Version |
|---|---:|
| NumPy | 2.4.4 |
| pandas | 2.3.3 |
| SciPy | 1.17.1 |
| statsmodels | 0.14.6 |
| Streamlit | 1.55.0 |
| scikit-posthocs | 0.14.0 |
| Matplotlib | 3.10.8 |
| Plotly | 6.6.0 |
| openpyxl | 3.1.5 |
| scikit-learn | 1.8.0 |

The tests cover content-based CSV/Excel identity, statistical summaries, independent
group tests and Dunn comparisons, correlation display filtering, conventional and
HC3 OLS reference calculations, regression explanations, ICC(A,1), responder error
intervals, temporal median bootstrap summaries, Matplotlib export identity, metadata
consistency, and application startup through Streamlit's test harness.

## Limits

This is a targeted regression and numerical-reference suite, not full clinical
validation. Browser-level interaction, large-file performance, deployment security,
inactive mediation/UMAP/PCA workflows, every Excel producer, and every combination
of missingness and categorical coding remain unvalidated. CI is configured for clean
Python 3.11 and 3.12 environments but was not executed on GitHub during this local
review.
