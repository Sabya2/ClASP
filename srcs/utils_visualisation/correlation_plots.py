"""Interactive correlation matrices and downloads."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re

from srcs.utils_visualisation.common import build_correlation_matrices


def _render_plotly_download(fig: go.Figure, filename_stem: str = "correlation_heatmap", key_prefix=None):
    """Render PNG and HTML download controls for a Plotly figure."""
    key_prefix = key_prefix or filename_stem
    with st.expander("⬇️ Download this chart", expanded=False):
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            try:
                png_bytes = fig.to_image(format="png", scale=2)
                st.download_button(
                    label="📥 Download PNG",
                    data=png_bytes,
                    file_name=f"{filename_stem}.png",
                    mime="image/png",
                    width="stretch",
                    key=f"{key_prefix}_png",
                )
            except Exception as e:
                st.warning(f"PNG export requires `kaleido`: `pip install kaleido`\n\n{e}")

        with dl_col2:
            html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button(
                label="📥 Download HTML",
                data=html_bytes,
                file_name=f"{filename_stem}.html",
                mime="text/html",
                width="stretch",
                key=f"{key_prefix}_html",
            )


def plot_correlation_heatmap_with_significance_plotly(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    method: str = "spearman",
    min_abs_corr: float = 0.0,
    width: int = 900,
    height: int = 900,
    show_text: bool = True,
    # wrap_width: int = 18,
    keep_diagonal: bool = True,
):
    """
    Plot a full correlation heatmap with significance stars using Plotly.
    """

    corr, pvals, annotated = build_correlation_matrices(df, method=method)

    corr = corr.copy()
    pvals = pvals.copy()
    annotated = annotated.copy()

    # --------------------------
    # Threshold filter only
    # Keep full square matrix (no triangle mask)
    # --------------------------
    threshold_mask = corr.abs() < min_abs_corr

    if keep_diagonal:
        np.fill_diagonal(threshold_mask.values, False)

    z = corr.mask(threshold_mask, np.nan)
    p_show = pvals.mask(threshold_mask, np.nan)
    text = annotated.mask(threshold_mask, "")

    label_map = {
        "pearson": "Pearson r",
        "spearman": "Spearman ρ",
        "kendall": "Kendall τ",
    }

    hover_text = []
    for i, row_name in enumerate(z.index):
        hover_row = []
        for j, col_name in enumerate(z.columns):
            z_val = z.iloc[i, j]
            p_val = p_show.iloc[i, j]

            if pd.isna(z_val):
                hover_row.append("")
            else:
                p_txt = "NA" if pd.isna(p_val) else f"{p_val:.3g}"
                hover_row.append(
                    f"<b>{row_name}</b> vs <b>{col_name}</b><br>"
                    f"r = {z_val:.3f}<br>"
                    f"p = {p_txt}"
                )
        hover_text.append(hover_row)
    LABEL_COLOR = 'black'
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z.values,
            x=list(z.columns),
            y=list(z.index),
            text=text.values if show_text else None,
            texttemplate="%{text}" if show_text else None,
            textfont={"size": 7},
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title=label_map.get(method, "Correlation"),
                len=0.5, outlinewidth=0,
                tickfont=dict(color=LABEL_COLOR),
                title_font=dict(color=LABEL_COLOR),
            ),
            xgap=0.2,
            ygap=1,
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=dict(text=f"{title} (displaying |r| ≥ {min_abs_corr:.2f}; diagonal {'kept' if keep_diagonal else 'hidden'})",
                   x=0.25,
                   font=dict(color=LABEL_COLOR),),
        width=width,
        height=height,
        template="simple_white", #"plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=100, b=10),
    )

    fig.update_xaxes(
        tickangle=270,
        tickvals=list(z.columns),
        side="bottom",
        showgrid=False,
        showline = False,
        zeroline=False,
        automargin=True,
        tickfont=dict(color=LABEL_COLOR),
    )

    fig.update_yaxes(
        tickvals=list(z.index),
        showgrid=False,
        showline = False,
        zeroline=False,
        autorange="reversed",
        scaleanchor="x",
        scaleratio=1,
        automargin=True,
        tickfont=dict(color=LABEL_COLOR),
    )

    return fig




def render_correlation_page(df, continuous_cols, group_col=None, participants_col=None):
    """Render correlation controls, heatmaps, and downloads."""
    st.header("Correlation plot")

    with st.expander("How this page works", expanded=False):
        st.markdown(
            """
        This page computes and displays correlation matrices for selected variables as interactive heatmaps.

        **What you can do here**
        - Select the numeric variables to include in the correlation matrix.
        - Choose the correlation method:
          - **Pearson**: linear association
          - **Spearman**: rank-based monotonic association
          - **Kendall**: rank-based association with a more conservative interpretation
        - Set a **minimum absolute correlation threshold** so weaker correlations are hidden from the heatmap display.
        - Optionally split the analysis by a selected grouping variable and generate one heatmap per group.

        **How the heatmap is built**
        - The selected variables are coerced to numeric values before correlation is computed.
        - Correlations are computed pairwise across variables, excluding missing values as needed.
        - The matrix remains square, and cells below the selected absolute-correlation threshold are hidden visually.
        - The diagonal can be retained so each variable still appears correlated with itself.

        **How to read the plot**
        - Cell color shows the direction and strength of association:
          - negative values on one side of the scale
          - positive values on the other
          - larger absolute values indicate stronger associations
        - Hover over any visible cell to inspect the variable pair, correlation coefficient, and p-value.
        - If text annotations are shown, they summarize the cell value and significance markers directly inside the heatmap.
        - When many variables are selected, text annotations may be hidden automatically to preserve readability.

        **Grouped analysis**
        - If **Plot separately by group** is enabled, the app computes a separate correlation matrix for each selected subgroup.
        - This helps compare whether variable relationships are stable or different across strata.

        **Practical notes**
        - Very high variable counts can make annotated heatmaps harder to read.
        - The minimum absolute-correlation slider filters only the display. It does not change coefficients or p-values and does not adjust for multiple testing.
        - The colour scale is fixed from -1 to 1; the diagonal is retained explicitly.
        - If a selected group has too few valid rows, its heatmap is skipped.
                    """
                )

    if df is None:
        st.warning("Upload data first.")
        return

    default_cols = continuous_cols if continuous_cols else df.select_dtypes(include="number").columns.tolist()

    # --------------------------git li
    # Variable selection
    # --------------------------
    col1, col2, col3 = st.columns([2, 1.0, 0.8], gap="small")

    with col1:
        corr_cols = st.multiselect(
            "Variables for correlation",
            options=df.columns.tolist(),
            default=default_cols[: min(12, len(default_cols))],
            key="corr_multiselect",
        )

    with col2:
        method = st.selectbox(
            "Correlation Method",
            ["pearson", "spearman", "kendall"],
            index=1,
            key="corr_method",
        )

    # with col3:
        split_by_group = st.checkbox(
            "Plot separately by group",
            value=False,
            key="corr_split_by_group",
            help="If checked, select a grouping variable and specific groups below. "
                 "If unchecked, a single correlation heatmap is shown for the whole dataset."
        )

    with col3:
        min_corr_to_plot = st.slider(
            "Minimum |correlation| to display",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="corr_min_abs_value",
            help="Hides weaker cells. The colour scale remains fixed from -1 to 1."
        )
    if len(corr_cols) < 2:
        st.info("Select at least two variables.")
        return

    selected_groups = None

    if split_by_group:
        g1, g2 = st.columns([2, 1.8], gap="small")

        with g1:
            possible_group_cols = df.columns.tolist()
            default_idx = possible_group_cols.index(group_col) if group_col in possible_group_cols else 0
            group_col = st.selectbox(
                "Select a Group column",
                options=possible_group_cols,
                index=default_idx,
                key="corr_group_col",
                help="Select a categorical column to split the heatmap by group."
            )

        with g2:
            available_groups = df[group_col].dropna().unique().tolist()
            default_groups_to_show = available_groups[:3]
            multiselect_key = f"corr_selected_groups_{group_col}"

            selected_groups = st.multiselect(
                "Groups to plot",
                options=available_groups,
                default=default_groups_to_show,
                key=multiselect_key,
            )

            if not selected_groups:
                st.info("Select at least one group.")
                return

    # --------------------------
    # Prepare dataframe
    # --------------------------
    corr_df = df[corr_cols].apply(pd.to_numeric, errors="coerce")

    # --------------------------
    # Overall plot
    # --------------------------
    if not split_by_group:
        n_vars = len(corr_cols)

        if n_vars > 40:
            st.warning(
                "Too many variables for a single highly readable annotated heatmap. "
                "Annotations are hidden automatically. Consider selecting fewer variables "
                "or increasing the minimum correlation threshold."
            )
        elif n_vars > 20:
            st.info("Annotations are hidden automatically for readability. Hover over cells to inspect values.")

        show_text = n_vars <= 20

        base_size = min(max(700, 32 * n_vars + 180), 1400)

        fig = plot_correlation_heatmap_with_significance_plotly(
            corr_df,
            title=f"Correlation heatmap ({method})",
            method=method,
            min_abs_corr=min_corr_to_plot,
            width=base_size,
            height=base_size,
            show_text=show_text,
            keep_diagonal=True,
        )

        st.plotly_chart(fig, width="content")
        _render_plotly_download(fig, filename_stem=f"corr_heatmap_{method}", key_prefix=f"corr_all_{method}")
        return

    # --------------------------
    # Group-wise plots
    # --------------------------
    for grp in selected_groups:
        st.markdown(f"### Group: {grp}")

        sub_df = df[df[group_col] == grp].copy()
        sub_corr_df = sub_df[corr_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")

        if sub_corr_df.shape[0] < 2:
            st.warning(f"Not enough rows for group '{grp}' to compute correlations.")
            continue

        n_vars = len(corr_cols)
        show_text = n_vars <= 20
        base_size = min(max(650, 30 * n_vars + 180), 1200)

        try:
            fig = plot_correlation_heatmap_with_significance_plotly(
                sub_corr_df,
                title=f"Correlation heatmap: {grp} ({method})",
                method=method,
                min_abs_corr=min_corr_to_plot,
                width=base_size,
                height=base_size,
                show_text=show_text,
                keep_diagonal=True,
            )
            st.plotly_chart(fig, width="content")
            safe_group = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(grp)).strip("_") or "group"
            stem = f"corr_heatmap_{method}_{safe_group}"
            _render_plotly_download(fig, filename_stem=stem, key_prefix=stem)

        except Exception as e:
            st.warning(f"Could not compute correlation for group '{grp}': {e}")
            continue
