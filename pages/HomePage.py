"""CLASP welcome page and usage guidance."""

import streamlit as st


HELP_TOPICS = {
    "Data summary": "pages/DataDescription.py",
    "Wrong column names": "pages/DataDescription.py",
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
    "Temporal dynamics": "pages/Temporal-Dynamics.py",
    "Time series": "pages/Temporal-Dynamics.py",
    "Trajectories": "pages/Temporal-Dynamics.py",
    "Repeated measures": "pages/Temporal-Dynamics.py",
    "Longitudinal data": "pages/Temporal-Dynamics.py",
    "Within-family dynamics": "pages/Temporal-Dynamics.py",
    "Cross-family dynamics": "pages/Temporal-Dynamics.py",
    "Temporal grouping": "pages/Temporal-Dynamics.py",
    "Baseline normalization": "pages/Temporal-Dynamics.py",
    "Responders": "pages/Responder.py",
    "Pre/post change": "pages/Responder.py",
    "SWC and TE": "pages/Responder.py",
    "Contact / Request Analysis": "pages/Contact-Request.py",
    "Analysis request": "pages/Contact-Request.py",
    "Feature request": "pages/Contact-Request.py",
    "Feedback": "pages/Contact-Request.py",
    "Bug report": "pages/Contact-Request.py",
    "Support": "pages/Contact-Request.py",
    "General inquiry": "pages/Contact-Request.py",
}

GUIDE_CONTENT = {
    "Input data": {
        "Overview": [
            "CLASP accepts CSV (`.csv`) and Excel (`.xlsx`, `.xls`) files.",
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
            "Use only letters, numbers, and underscores (`_`) in column names. Avoid spaces, leading or trailing whitespace, and other special characters, as they can interfere with the analysis code and cause the application to fail.",
            "After uploading your data, open **Data summary**. The Data description page displays a red error when incorrectly named columns are detected; expand **View invalid column names** to see the affected columns.",
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
            "CLASP attempts to convert selected columns to numeric, interpret decimal separators correctly, and validate participant IDs where required.",
            "Check **Data summary** for incorrectly named columns. Column-name validation identifies spaces, leading or trailing whitespace, and other unsupported characters.",
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
        "Temporal dynamics": [
            "Open the **Temporal Dynamics** page to analyze repeated measurements whose column names share a common structure and differ at a numbered measurement position.",
            "Examples include `Glucose_T1`, `Spiro_3min_HR`, `OCTA_hour_4`, and `Spiro_Level1_[W]`. The page groups matching columns into families and orders them by their detected timepoint.",
            "Use **Within-Family dynamics** to inspect selected families in their own trajectory graphs, choose SD, SE, or 95% CI error bars, and optionally display an overall median or separate trajectories from an existing group column.",
            "Use **Cross-Family dynamics** to compare any number of families. Separate layout distributes families across graphs in groups of two or three; Combined layout overlays every selected family in one graph.",
            "When a participant ID is selected, baseline normalization can set each participant's first available value within each family to 100% before summary statistics are calculated.",
            "Review **Detected repeated-measure families** on the page before analysis to confirm the inferred family names, source columns, and numerical timepoints.",
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
            "Confirm local deployment and institutional data-governance requirements before uploading data."
        ],
    },
}


quick_start_col, help_col = st.columns(2, gap="large")

with quick_start_col:
    st.markdown("### Quick start")
    st.markdown("1. Prepare a CSV/Excel file using the "
        "[input requirements](#guidelines) below."
    )
    st.markdown("2. Upload the file and verify columns are detected correctly.")
    st.markdown("3. Open Data Summary using the link below and review any incorrectly "
        "named columns. The page lists affected columns so you can rename them "
        "before analysis."
    )
    st.page_link(
        "pages/DataDescription.py",
        label="Open Data Summary", icon="📋", )

    st.markdown("4. Update invalid names to use letters, numbers, and underscores only "
        "(for example, `systolic_bp_T1`) to avoid analysis errors."
    )
    st.markdown("5. Use the clickable Help topics to open the relevant analysis page, "
        "then select variables and analysis options."
    )
    st.markdown("6. Download results where available.")
    st.markdown("7. Open Contact / Request Analysis using the link below to request an "
        "analysis, suggest a feature, send feedback, or report a problem."
    )
    st.page_link(
        "pages/Contact-Request.py",label="Open Contact / Request Analysis",icon="✉️",)

with help_col:
    st.markdown("### Help")
    st.caption("Select a keyword to open the relevant CLASP page.")
    selected_topic = st.pills(
        "Help topics",
        options=list(HELP_TOPICS),
        label_visibility="collapsed",
    )
    if selected_topic:
        st.switch_page(HELP_TOPICS[selected_topic])


st.markdown("#### Guidelines")

section_col, section_line_col, topic_col, topic_line_col, guidance_col = st.columns(
    [1, 0.18, 1.2, 0.18, 3.25],
    gap="small",
    vertical_alignment="top",
)

with section_col:
    with st.container(border=False):
        selected_section = st.segmented_control(
            "Guidance section",
            options=list(GUIDE_CONTENT),
            default="Input data",
            label_visibility="collapsed",
        )

with section_line_col:
    st.markdown(
        """<div style="display:flex;align-items:center;margin-top:2rem">
        <span style="width:7px;height:7px;border-radius:50%;background:var(--primary-color, #76B5A5)"></span>
        <span style="flex:1;border-top:2px solid var(--primary-color, #76B5A5)"></span>
        <span style="width:7px;height:7px;border-radius:50%;background:var(--primary-color, #76B5A5)"></span>
        </div>""",
        unsafe_allow_html=True,
    )

if selected_section:
    subtopics = list(GUIDE_CONTENT[selected_section])
    default_subtopic = "Column names" if selected_section == "Input data" else subtopics[0]

    with topic_col:
        with st.container(border=False):
            selected_subtopic = st.segmented_control(
                "Guidance topic",
                options=subtopics,
                default=default_subtopic,
                label_visibility="collapsed",
                key=f"guide_topic_{selected_section}",
            )

    with topic_line_col:
        st.markdown(
            """<div style="display:flex;align-items:center;margin-top:2rem">
            <span style="width:7px;height:7px;border-radius:50%;background:var(--primary-color, #76B5A5)"></span>
            <span style="flex:1;border-top:2px solid var(--primary-color, #76B5A5)"></span>
            <span style="width:7px;height:7px;border-radius:50%;background:var(--primary-color, #76B5A5)"></span>
            </div>""",
            unsafe_allow_html=True,
        )

    if selected_subtopic:
        with guidance_col:
            with st.container(border=True):
                for paragraph in GUIDE_CONTENT[selected_section][selected_subtopic]:
                    st.markdown(paragraph)


st.markdown("### Reproducibility and citation")
st.markdown("Record CLASP version, analysis date, input-data preparation steps, selected variables, statistical method and options, grouping variable (if used), and downloaded output files.")

st.markdown("#### How to cite CLASP")
st.markdown(
    "If you use CLASP in your research, please cite the software version used "
    "in your analysis."
)
st.markdown(
    "DOI: [https://doi.org/10.5281/zenodo.23976795]"
    "(https://doi.org/10.5281/zenodo.23976795)"
)

st.markdown("### Feedback and support")
st.markdown(
    "Open **Contact / Request Analysis** to request an analysis, suggest a "
    "feature, send feedback, report a bug, or submit a general inquiry."
)
st.page_link(
    "pages/Contact-Request.py",
    label="Contact / Request Analysis",
    icon="✉️",
)
st.warning(
    "Do not include confidential or identifiable participant data in the "
    "message or attachments."
)
st.markdown("When reporting an issue, include CLASP version, analysis page, file type and size, selected options, exact error message, and a small synthetic example if possible.")
