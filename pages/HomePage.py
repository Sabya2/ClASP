"""CLASP welcome page and usage guidance."""

import streamlit as st


HELP_TOPICS = {
    "Data summary": "pages/DataDescription.py",
    "Column types": "pages/DataDescription.py",
    "Missing data": "pages/DataDescription.py",
    "Histograms": "pages/DataDistribution.py",
    "Density plots": "pages/DataDistribution.py",
    "Scatter plots": "pages/RegressionAnalysis.py",
    "Linear regression": "pages/RegressionAnalysis.py",
    "Correlation": "pages/Correlation.py",
    "Correlation heatmaps": "pages/Correlation.py",
    "Box plots": "pages/GroupComparison.py",
    "Violin plots": "pages/GroupComparison.py",
    "Group tests": "pages/GroupComparison.py",
    "Mediation": "pages/Mediation.py",
    "Time series": "pages/Temporal-Dynamics.py",
    "Trajectories": "pages/Temporal-Dynamics.py",
    "Responders": "pages/Responder.py",
    "Pre/post change": "pages/Responder.py",
    "SWC and TE": "pages/Responder.py",
}

GUIDE_CONTENT = {
    "Input data": {
        "Overview": [
            "ClASPx accepts CSV (`.csv`) and Excel (`.xlsx`, `.xls`) files.",
            "First row must contain column names; each subsequent row represents one participant or observation.",
            "Include clear, unique column names; avoid duplicate names.",
            "Typical columns: optional participant ID, optional grouping variable, numeric measurement columns, and optional pre/post/reliability columns for responder analysis.",
            "For group comparisons, include at least one categorical grouping variable and one or more numeric outcome columns.",
        ],
        "Numeric": [
            "Use periods (`.`) as decimal separators (e.g., `32.5`, not `32,5`).",
            "Avoid units, symbols, or thousands separators inside numeric cells.",
            "Leave missing values blank or use a consistent missing-value code.",
        ],
        "Column names": [
            "Column names should be clear and unique.",
            "Use only letters, numbers, and underscores (`_`) in column names. Avoid spaces and other special characters, as they can interfere with the analysis code and cause the application to fail.",
        ],
        "CSV": [
            "Save as UTF-8 CSV; use commas to separate columns; keep the first row as the header.",
            "Ensure every data row has the same number of columns.",
            "Remove duplicate participant IDs unless repeated measurements are intentional.",
        ],
        "Excel": [
            "Avoid merged cells, title rows above the header, and multiple tables on one worksheet.",
            "Store numeric measurements as numeric cells, not formatted text.",
            "Remove formulas or confirm the uploaded workbook contains the intended calculated values.",
        ],
    },
    "Data quality": {
        "Import checks": [
            "ClASPx attempts to convert selected columns to numeric, interpret decimal separators correctly, and validate participant IDs where required."
        ],
        "Consistency": [
            "Group label consistency is checked (e.g., `Training` vs. `training`)."
        ],
        "Sample quality": [
            "Analyses require sufficient observations; extreme values and outliers may be flagged.",
            "Incomplete rows may be excluded; always review the reported sample size and data-processing information.",
        ],
    },
    "Analyses": {
        "Correlation": [
            "Calculate Pearson, Spearman, or Kendall correlation matrices and display interactive heatmaps, optionally stratified by group."
        ],
        "Regression": [
            "Fit univariate or multivariable linear models, view adjusted coefficients, and explore X–Y relationships with interactive pair plots."
        ],
        "Responders": [
            "Classify individual responses using Smallest Worthwhile Change (SWC) and Typical Error (TE), with responder plots and downloadable results."
        ],
        "Group comparison": [
            "Visualize continuous variables across groups with box/violin plots, optional y-axis transformations, and omnibus plus pairwise statistical tests."
        ],
    },
    "Privacy": {
        "Safe uploads": [
            "Do not upload identifiable clinical data unless the application use is approved for that data."
        ],
        "De-identification": [
            "Prefer pseudonymous participant identifiers and de-identified datasets for demonstrations."
        ],
        "Privacy notice": [
            "Review the ClASPx privacy notice for details on login data, uploaded files, usage tracking, and retention."
        ],
    },
}


quick_start_col, help_col = st.columns(2, gap="large")

with quick_start_col:
    st.markdown("### Quick start")
    st.markdown("1. Prepare a CSV/Excel file following the input requirements below.")
    st.markdown("2. Upload the file and verify columns are detected correctly.")
    st.markdown("3. Select variables and analysis options on the relevant page.")
    st.markdown("4. Download results where available.")

with help_col:
    st.markdown("### Help")
    st.caption("Select a keyword to open the relevant ClASPx page.")
    selected_topic = st.pills(
        "Help topics",
        options=list(HELP_TOPICS),
        label_visibility="collapsed",
    )
    if selected_topic:
        st.switch_page(HELP_TOPICS[selected_topic])



st.markdown("### Reproducibility and citation")
st.markdown("Record ClASPx version, analysis date, input-data preparation steps, selected variables, statistical method and options, grouping variable (if used), and downloaded output files.")

st.markdown("#### DOI")
st.markdown("DOI: _[to be added upon release]_")

st.info("Please use this citation upon usage of the tool.")


selected_section = st.segmented_control(
    "Guidance section",
    options=list(GUIDE_CONTENT),
    default=None,
    label_visibility="collapsed",
)

if selected_section:
    selected_subtopic = st.segmented_control(
        "Guidance topic",
        options=list(GUIDE_CONTENT[selected_section]),
        default=None,
        label_visibility="collapsed",
        key=f"guide_topic_{selected_section}",
    )
    if selected_subtopic:
        for paragraph in GUIDE_CONTENT[selected_section][selected_subtopic]:
            st.markdown(paragraph)


st.markdown("### Feedback and support")
st.markdown("Use the **Send feedback** link in the application to report problems or suggest improvements; do not include confidential or identifiable data.")
st.markdown("When reporting an issue, include ClASPx version, analysis page, file type and size, selected options, exact error message, and a small synthetic example if possible.")
