"""CLASP application entry point and page navigation."""

from pathlib import Path

import streamlit as st

from srcs.utils_features.common import (
    init_session_state,
    load_dataframe_ui,
    render_sidebar_variable_selector,
    set_paper_plot_style,
)


DEFAULT_PAGE_CONFIG = {
    "HomePage.py": {"title": "HomePage", "icon": "🏠"},
    "DataDescription.py": {"title": "Data Summary", "icon": "📋"},
    "DataDistribution.py": {"title": "Data Distribution", "icon": "📊"},
    "RegressionAnalysis.py": {"title": "Regression Analysis", "icon": "📈"},
    "Correlation.py": {"title": "Correlation Analysis", "icon": "🔗"},
    "GroupComparison.py": {"title": "Group Comparison Analysis", "icon": "⚖️"},
    # "Mediation.py": {"title": "Mediation Analysis", "icon": "🔀"},
    "Temporal-Dynamics.py": {"title": "Temporal Dynamics", "icon": "🕒"},
    "Responder.py": {"title": "Responder Analysis", "icon": "🎯"},
}


def build_analysis_pages(pages_dir: str = "pages") -> list:
    """Build the configured Streamlit pages found in ``pages_dir``."""
    pages_path = Path(pages_dir)
    if not pages_path.exists():
        return []

    # Iterate over the configuration itself so the navigation follows its
    # insertion order instead of sorting pages alphabetically by filename.
    page_files = [
        pages_path / filename
        for filename in DEFAULT_PAGE_CONFIG
        if (pages_path / filename).is_file()
    ]
    return [
        st.Page(str(path), **DEFAULT_PAGE_CONFIG[path.name])
        for path in page_files
    ]


def main() -> None:
    """Configure CLASP, load data, and run the selected analysis page."""
    st.set_page_config(
        page_title="CLinical and Statistical Analysis Software Platform",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stLinkButton"] a:hover,
        div[data-testid="stPills"] button:hover,
        div[data-testid="stSegmentedControl"] button:hover,
        div[data-testid="stButtonGroup"] button:hover {
            background-color: #A7D8CC !important;
            border-color: #A7D8CC !important;
            color: #111827 !important;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="display:flex;align-items:baseline;gap:12px">
          <span style="font-size:36px;font-weight:700">CLASP</span>
          <span style="font-size:20px;color:#5F9F93">
            - CLinical and Statistical Analysis Software Platform
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    set_paper_plot_style()
    init_session_state()

    df = load_dataframe_ui()
    if df is not None:
        st.session_state["df"] = df
        render_sidebar_variable_selector(df)

    analysis_pages = build_analysis_pages()
    if not analysis_pages:
        st.error("No analysis pages found in the 'pages' directory.")
        return

    st.navigation({"Analysis": analysis_pages}).run()


if __name__ == "__main__":
    main()
