"""CLASP welcome page and usage guidance."""

import streamlit as st

st.markdown("### Quick start")
st.markdown("1. Prepare a CSV/Excel file following the input requirements below.")
st.markdown("2. Upload the file and verify columns are detected correctly.")
st.markdown("3. Select variables and analysis options on the relevant page.")
st.markdown("4. Download results where available.")



st.markdown("### Reproducibility and citation")
st.markdown("Record ClASPx version, analysis date, input-data preparation steps, selected variables, statistical method and options, grouping variable (if used), and downloaded output files.")

st.markdown("#### DOI")
st.markdown("DOI: _[to be added upon release]_")

st.info("Please use this citation upon usage of the tool.")


with st.expander("Input data requirements", expanded=False):
    st.markdown("ClASPx accepts CSV (`.csv`) and Excel (`.xlsx`, `.xls`) files.")
    st.markdown("First row must contain column names; each subsequent row represents one participant or observation.")
    st.markdown("Include clear, unique column names; avoid duplicate names.")
    st.markdown("Typical columns: optional participant ID, optional grouping variable, numeric measurement columns, and optional pre/post/reliability columns for responder analysis.")
    st.markdown("For group comparisons, include at least one categorical grouping variable and one or more numeric outcome columns.")

    st.markdown("#### Numeric formatting")
    st.markdown("Use periods (`.`) as decimal separators (e.g., `32.5`, not `32,5`).")
    st.markdown("Avoid units, symbols, or thousands separators inside numeric cells.")
    st.markdown("Leave missing values blank or use a consistent missing-value code.")

    st.markdown("#### CSV recommendations")
    st.markdown("Save as UTF-8 CSV; use commas to separate columns; keep the first row as the header.")
    st.markdown("Ensure every data row has the same number of columns.")
    st.markdown("Remove duplicate participant IDs unless repeated measurements are intentional.")

    st.markdown("#### Excel recommendations")
    st.markdown("Avoid merged cells, title rows above the header, and multiple tables on one worksheet.")
    st.markdown("Store numeric measurements as numeric cells, not formatted text.")
    st.markdown("Remove formulas or confirm the uploaded workbook contains the intended calculated values.")


with st.expander("Data quality checks", expanded=False):
    st.markdown("ClASPx attempts to convert selected columns to numeric, interpret decimal separators correctly, and validate participant IDs where required.")
    st.markdown("Group label consistency is checked (e.g., `Training` vs. `training`).")
    st.markdown("Analyses require sufficient observations; extreme values and outliers may be flagged.")
    st.markdown("Incomplete rows may be excluded; always review the reported sample size and data-processing information.")


with st.expander("Available analyses", expanded=False):
    st.markdown("**Correlation:** Calculate Pearson, Spearman, or Kendall correlation matrices and display interactive heatmaps, optionally stratified by group.")
    st.markdown("**OLS Regression:** Fit univariate or multivariable linear models, view adjusted coefficients, and explore X–Y relationships with interactive pair plots.")
    st.markdown("**Responder Analysis:** Classify individual responses using Smallest Worthwhile Change (SWC) and Typical Error (TE), with responder plots and downloadable results.")
    st.markdown("**Group Comparison:** Visualize continuous variables across groups with box/violin plots, optional y-axis transformations, and omnibus plus pairwise statistical tests.")


with st.expander("Privacy and security", expanded=False):
    st.markdown("Do not upload identifiable clinical data unless the application use is approved for that data.")
    st.markdown("Prefer pseudonymous participant identifiers and de-identified datasets for demonstrations.")
    st.markdown("Review the ClASPx privacy notice for details on login data, uploaded files, usage tracking, and retention.")


st.markdown("### Feedback and support")
st.markdown("Use the **Send feedback** link in the application to report problems or suggest improvements; do not include confidential or identifiable data.")
st.markdown("When reporting an issue, include ClASPx version, analysis page, file type and size, selected options, exact error message, and a small synthetic example if possible.")
