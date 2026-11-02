"""Project-wide graph typography without changing Streamlit UI text."""

import matplotlib as mpl
import plotly.io as pio


GRAPH_FONT_OPTIONS = {
    "STIXGeneral": {
        "mathtext": "stix",
        "plotly": "STIXGeneral, STIX Two Text, serif",
    },
    "DejaVu Serif": {
        "mathtext": "dejavuserif",
        "plotly": "DejaVu Serif, serif",
    },
    "Liberation Serif": {
        "mathtext": "dejavuserif",
        "plotly": "Liberation Serif, Times New Roman, serif",
    },
    "Noto Sans": {
        "mathtext": "dejavusans",
        "plotly": "Noto Sans, Arial, sans-serif",
    },
    "DejaVu Sans": {
        "mathtext": "dejavusans",
        "plotly": "DejaVu Sans, Arial, sans-serif",
    },
    "Liberation Sans": {
        "mathtext": "dejavusans",
        "plotly": "Liberation Sans, Arial, sans-serif",
    },
}

# Change only this value to switch the font used by every graph.
DEFAULT_GRAPH_FONT = "Liberation Serif"#"Noto Sans"


def apply_graph_font(font_family=DEFAULT_GRAPH_FONT):
    """Apply a supported graph font to Matplotlib and Plotly templates."""
    if font_family not in GRAPH_FONT_OPTIONS:
        available = ", ".join(GRAPH_FONT_OPTIONS)
        raise ValueError(
            f"Unsupported graph font {font_family!r}. Available options: {available}"
        )

    font_config = GRAPH_FONT_OPTIONS[font_family]
    mpl.rcParams.update({
        "font.family": font_family,
        "mathtext.fontset": font_config["mathtext"],
    })

    for template_name in (
        "plotly",
        "plotly_white",
        "plotly_dark",
        "simple_white",
    ):
        if template_name in pio.templates:
            pio.templates[template_name].layout.font.family = (
                font_config["plotly"]
            )
