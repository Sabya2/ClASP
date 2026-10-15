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
    "1_Clasp-intro.py": {"title": "HomePage", "icon": "📊"},
    "Data-Description.py": {"title": "Data Summary", "icon": "📊"},
    "Histograms.py": {"title": "Data Distribution", "icon": "📶"},
    "Linear.py": {"title": "Regression Analysis", "icon": "📈"},
    "Correlation.py": {"title": "Correlation Analysis", "icon": "🔗"},
    "Group-comparison.py": {"title": "Group Comparison Analysis", "icon": "⚖️"},
    "Mediation.py": {"title": "Mediation Analysis", "icon": "🔀"},
    "Temporal-Dynamics.py": {"title": "Temporal Dynamics", "icon": "⏱️"},
    "Responder.py": {"title": "Responder Analysis", "icon": "🔗"},
}


def build_analysis_pages(pages_dir: str = "pages") -> list:
    """Build the configured Streamlit pages found in ``pages_dir``."""
    pages_path = Path(pages_dir)
    if not pages_path.exists():
        return []

    page_files = sorted(
        (path for path in pages_path.glob("*.py") if path.name in DEFAULT_PAGE_CONFIG),
        key=lambda path: path.name,
    )
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
        <div style="display:flex;align-items:baseline;gap:12px">
          <span style="font-size:36px;font-weight:700">CLASP</span>
          <span style="font-size:20px;color:#666">
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
