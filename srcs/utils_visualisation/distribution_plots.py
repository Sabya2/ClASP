"""Histogram, box, and violin plot renderers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from srcs.utils_visualisation.common import (
    _candidate_group_columns,
    get_figure_download,
    get_plot_palette,
    set_paper_plot_style,
)





# -------------------------------------------------------------------------
# Below are the histogram render functions called from the page scripts
# -------------------------------------------------------------------------
def render_histogram_page(df, continuous_cols, group_col):
    """Render configurable histogram grids for continuous variables."""
    st.subheader("Histogram plots")
    # -----------------------------------------------------
    # Style
    # -----------------------------------------------------
    set_paper_plot_style()

    # -----------------------------------------------------
    # Input checks
    # -----------------------------------------------------
    if df is None:
        st.warning("Upload data first.")
        return

    if not continuous_cols:
        st.warning("Select at least one continuous column in the sidebar.")
        return

    # -----------------------------------------------------
    # Variable selection
    # -----------------------------------------------------
    cols_to_plot = st.multiselect(
        "Select variables",
        options=continuous_cols,
        default=continuous_cols[: min(6, len(continuous_cols))],
        key="hist_cols_multiselect",
    )
    if not cols_to_plot:
        st.info("Select at least one variable.")
        return

    # -----------------------------------------------------
    # Controls
    # -----------------------------------------------------
    col1, col2, col3 = st.columns([0.75, 0.75, 2.5], gap="small")

    with col1:
        ncols = st.slider(
            "Grid columns",
            min_value=1,
            max_value=4,
            value=2,
            key="hist_grid_cols",
        )

    with col2:
        bins = st.slider(
            "Bins",
            min_value=5,
            max_value=80,
            value=25,
            key="hist_bins",
        )

    with col3:
        use_group = st.checkbox(
            "Color by group",
            value=bool(group_col),
            key="hist_use_group",
        )

        # -----------------------------------------------------
        # Group controls
        # -----------------------------------------------------
        plot_group_col = group_col
        selected_groups = None

        if use_group:
            if not group_col:
                st.warning("Select a valid group column in the sidebar first.")
                return

            available_group_cols = _candidate_group_columns(df, group_col)

            g1, g2 = st.columns([1, 2], gap="xsmall")

            with g1:
                use_alt_group = st.checkbox(
                    "Different group variable",
                    value=False,
                    key="hist_use_alt_group",
                )

                if use_alt_group:
                    plot_group_col = st.selectbox(
                        "Select another group variable",
                        options=available_group_cols,
                        index=available_group_cols.index(group_col) if group_col in available_group_cols else 0,
                        key="hist_group_selectbox",
                    )

                if not plot_group_col:
                    st.warning("Select a valid group column.")
                    return

            with g2:
                st.markdown(f"Group values for **{plot_group_col}**")

                groups_to_show = df[plot_group_col].dropna().unique().tolist()

                state_key = f"hist_selected_groups__{plot_group_col}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = groups_to_show[: min(6, len(groups_to_show))]

                st.session_state[state_key] = [
                    g for g in st.session_state[state_key] if g in groups_to_show
                ]

                selected_groups = st.multiselect(
                    "Select the groups to compare",
                    options=groups_to_show,
                    default=st.session_state[state_key],
                    key=f"{state_key}_widget",
                )

                st.session_state[state_key] = selected_groups

                if not selected_groups:
                    st.info("Select at least one group.")
                    return

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------

    for row_start in range(0, len(cols_to_plot), ncols):
        row_vars = cols_to_plot[row_start: row_start + ncols]
        cols_layout = st.columns(ncols)

        for col_container, col in zip(cols_layout, row_vars):
            with col_container:
                needed_cols = [col]
                if use_group and plot_group_col:
                    needed_cols.append(plot_group_col)

                plot_df = df[needed_cols].copy()
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
                plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col])

                if plot_df.empty:
                    st.info(f"No data for {col}")
                    continue

                fig, ax = plt.subplots(figsize=(5.2, 4.0))

                if use_group and plot_group_col and plot_group_col in plot_df.columns:
                    plot_df = plot_df.dropna(subset=[plot_group_col])
                    plot_df = plot_df[plot_df[plot_group_col].isin(selected_groups)].copy()

                    if plot_df.empty:
                        st.info(f"No valid grouped data for {col}")
                        plt.close(fig)
                        continue

                    group_order = [g for g in selected_groups if g in plot_df[plot_group_col].unique()]

                    sns.histplot(
                        data=plot_df,
                        x=col,
                        hue=plot_group_col,
                        hue_order=group_order,
                        bins=bins,
                        kde=True,
                        element="step",
                        stat="count",
                        common_norm=False,
                        multiple="layer",
                        ax=ax,
                    )
                else:
                    sns.histplot(
                        data=plot_df,
                        x=col,
                        bins=bins,
                        kde=True,
                        ax=ax,
                    )

                ax.set_title(col)
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.grid(True, axis="y", alpha=0.25)

                plt.tight_layout()
                st.pyplot(fig)

                buf = get_figure_download(fig)
                st.download_button(
                    label=f"Download {col}",
                    data=buf,
                    file_name=f"histogram_{col}.png",
                    mime="image/png",
                    key=f"hist_download_{col}",
                )

                plt.close(fig)



# -------------------------------------------------------------------------
# Below are the box/violin render functions called from the page scripts
# -------------------------------------------------------------------------


def render_box_violin_page(df, continuous_cols, group_col):
    """
    Render boxplot / violinplot page with optional grayscale mode
    and stable group color mapping.
    """
    # -----------------------------------------------------
    # Input checks
    # -----------------------------------------------------
    if df is None:
        st.warning("Upload data first.")
        return

    if not continuous_cols:
        st.warning("Select at least one continuous column in the sidebar.")
        return

    if not group_col:
        st.warning("Select a valid group column.")
        return

    # -----------------------------------------------------
    # Plot appearance controls
    # -----------------------------------------------------
    use_monochrome = st.checkbox(
        "Use grayscale palette",
        value=True,
        key="plot_use_grayscale",
    )
    plot_mode = "gray" if use_monochrome else "multicolor"

    show_grid = st.checkbox(
        "Show grid",
        value=False,
        key="plot_show_grid",
    )

    # Grid only affects layout style, not colors
    set_paper_plot_style(show_grid=show_grid)

    # -----------------------------------------------------
    # Plot variable controls
    # -----------------------------------------------------
    c1, c2, c3 = st.columns([3, 1, 2], gap="small")

    with c1:
        cols_to_plot = st.multiselect(
            "Select variables",
            options=continuous_cols,
            default=continuous_cols[: min(6, len(continuous_cols))],
            key="dist_cols_multiselect",
        )

    with c2:
        plot_kind = st.radio(
            "Plot type",
            ["box", "violin"],
            horizontal=False,
            key="dist_plot_kind",
        )

    with c3:
        ncols = st.slider(
            "Grid columns",
            min_value=1,
            max_value=4,
            value=2,
            key="dist_grid_cols",
        )
        show_jitter = st.checkbox(
            "Show jitters",
            value=True,
            key="dist_show_jitter",
        )

    # -----------------------------------------------------
    # Group controls
    # -----------------------------------------------------
    plot_group_col = group_col
    available_group_cols = _candidate_group_columns(df, group_col)

    g1, g2 = st.columns([1, 2], gap="small")

    with g1:
        use_alt_group = st.checkbox(
            "Use a different group variable on this page",
            value=False,
            key="dist_use_alt_group",
        )

        if use_alt_group:
            plot_group_col = st.selectbox(
                "Select another group variable",
                options=available_group_cols,
                index=available_group_cols.index(group_col) if group_col in available_group_cols else 0,
                key="dist_group_selectbox",
            )

        if not plot_group_col:
            st.warning("Select a valid group column.")
            return

    with g2:
        st.markdown(f"Group values for **{plot_group_col}**")

        groups_to_show = df[plot_group_col].dropna().unique().tolist()

        state_key = f"dist_selected_groups__{plot_group_col}"
        if state_key not in st.session_state:
            st.session_state[state_key] = groups_to_show[: min(6, len(groups_to_show))]

        st.session_state[state_key] = [
            g for g in st.session_state[state_key] if g in groups_to_show
        ]

        selected_groups = st.multiselect(
            "Select the groups to compare",
            options=groups_to_show,
            default=st.session_state[state_key],
            key=f"{state_key}_widget",
        )

        st.session_state[state_key] = selected_groups

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------
    if not cols_to_plot:
        st.info("Select at least one variable.")
        return

    if not selected_groups:
        st.info("Select at least one group.")
        return

    # -----------------------------------------------------
    # Filter data and define palette
    # -----------------------------------------------------
    plot_df = df[df[plot_group_col].isin(selected_groups)].copy()
    palette_map = get_plot_palette(selected_groups, mode=plot_mode)

    st.subheader(f"{plot_kind.capitalize()} plots")

    # -----------------------------------------------------
    # Draw plots
    # -----------------------------------------------------
    for row_start in range(0, len(cols_to_plot), ncols):
        row_vars = cols_to_plot[row_start: row_start + ncols]
        row_layout = st.columns(ncols)

        for container, var in zip(row_layout, row_vars):
            with container:
                sub = plot_df[[plot_group_col, var]].copy()
                sub[var] = pd.to_numeric(sub[var], errors="coerce")
                sub = sub.dropna()

                if sub.empty:
                    st.info(f"No data for {var}")
                    continue

                fig, ax = plt.subplots(figsize=(5.2, 4.2))

                if plot_kind == "box":
                    sns.boxplot(data=sub, x=plot_group_col, y=var,
                                hue=plot_group_col, order=selected_groups,  palette=palette_map,
                                dodge=False, legend=False, ax=ax,
                                    )
                else:
                    sns.violinplot(data=sub, x=plot_group_col, y=var,
                                hue=plot_group_col, order=selected_groups,  palette=palette_map,
                                inner="box", dodge=False, legend=False, ax=ax,
                                    )

                if show_jitter:
                    sns.stripplot(data=sub,x=plot_group_col,y=var,
                                order=selected_groups,color="#222222",
                                alpha=0.65,size=2,jitter=0.18,ax=ax,
                                    )

                counts = sub[plot_group_col].value_counts()
                ax.set_xticks(range(len(selected_groups)))
                ax.set_xticklabels(
                    [f"{g}\n(n={counts.get(g, 0)})" for g in selected_groups],
                    rotation=0,
                )

                ax.set_title(var)
                ax.set_xlabel("")
                ax.set_ylabel(var)

                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

                plt.tight_layout()
                st.pyplot(fig)

                buf = get_figure_download(fig)
                st.download_button(
                    label=f"Download {var}",
                    data=buf,
                    file_name=f"{plot_kind}_{var}.png",
                    mime="image/png",
                    key=f"download_{plot_kind}_{var}",
                )



def render_interactive_box_violin_plot(df, continuous_cols, group_col, participants_col):
    """Render one interactive box or violin plot."""
    if df is None or len(continuous_cols) < 2:
        st.warning("Upload data and select at least two continuous variables first.")
        return

    # -------------------------
    # Select variables and the dataframe subset
    # -------------------------
    var_cols1, var_cols2 = st.columns([2, 1], border=True, gap="xxsmall")
    with var_cols1:
        y_col = st.selectbox("Y variable", options=continuous_cols, index=0, key="interactive_y")
    with var_cols2:
        st.markdown("Group toggle")
        color_by_group = st.checkbox("Color by group",
                                    value=bool(group_col and group_col in df.columns),
                                    key="interactive_color_by_group",)

    cols = [y_col]
    if participants_col and participants_col in df.columns:
        cols.append(participants_col)
    if group_col and group_col in df.columns:
        cols.append(group_col)

    sub = df[cols].copy()
    # sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    color_col = group_col if color_by_group and group_col in sub.columns else None

    # -------------------------
    # Plot
    # -------------------------
    col1, col2 = st.columns([2, 1], border=True, gap="xsmall")

    with col1:
        fig = px.violin(
            sub,
            y=y_col,
            x=color_col if color_col else None,   # group on x if available
            color=color_col,
            box=True,        # adds boxplot inside violin (nice!)
            points="all",    # shows raw data points
            template="plotly_dark",
            height=500,
            title=f"{y_col} distribution",
            hover_data=[participants_col] if participants_col in sub.columns else None,
        )

        fig.update_layout(
            plot_bgcolor="white",
            font=dict(color="black"),
        )

        fig.update_xaxes(showgrid=True, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridcolor="lightgray")

        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown(
            "This interactive plot allows you to explore pairwise relationships between continuous variables. "
            "You can hover over points to see details, and the OLS line helps identify trends. "
            "Use the group column (if selected) to color points by category."
        )
        display_cols = [c for c in [y_col, color_col] if c is not None and c in sub.columns]
        st.dataframe(sub[display_cols].head(10), width='stretch')

    return y_col








# -------------------------------------------------------------------------
# Below are the linear render functions called from the page scripts
# -------------------------------------------------------------------------
