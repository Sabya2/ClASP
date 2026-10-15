"""Interactive bivariate regression plots."""

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px





# -------------------------------------------------------------------------
# Below are the linear render functions called from the page scripts
# -------------------------------------------------------------------------


def render_interactive_linear_plot(df, continuous_cols, group_col, participants_col):
    """Render one interactive X-Y regression plot."""

    if df is None or len(continuous_cols) < 2:
        st.warning("Upload data and select at least two continuous variables first.")
        return

    # -------------------------
    # Select variables and teh dataframe subset
    # -------------------------
    var_cols1, var_cols2, var_cols3 = st.columns([1.5, 1.5, 2], border=True, gap="xxsmall")
    with var_cols1:
        x_col = st.selectbox("X variable", options=continuous_cols, index=0, key="interactive_x")
    with var_cols2:
        possible_y = [c for c in continuous_cols if c != x_col]
        y_col = st.selectbox("Y variable", options=possible_y, index=0, key="interactive_y")
    with var_cols3:
        plot_by_group = st.checkbox(
            "Plot by group",
            value=False, #bool(group_col),
            key="interactive_line_plot_by_group"
        )

        if plot_by_group:
            g1, g2 = st.columns([0.8,1], gap="small")

            with g1:
                possible_group_cols = df.columns.tolist()
                default_idx = possible_group_cols.index(group_col) if group_col in possible_group_cols else 0
                group_col = st.selectbox(
                    "Group column",
                    options=possible_group_cols,
                    index=default_idx,
                    key="interactive_group_col",
                    help="Select a categorical column to split the heatmap by group."
                )

            with g2:
                available_groups = df[group_col].dropna().unique().tolist()
                default_groups_to_show = available_groups[:3]
                multiselect_key = f"interactive_selected_groups_{group_col}"

                selected_groups = st.multiselect(
                    "Groups to plot",
                    options=available_groups,
                    default=default_groups_to_show,
                    key=multiselect_key,
                )

                if not selected_groups:
                    st.info("Select at least one group.")
                    return

    cols = [x_col, y_col]
    if participants_col and participants_col in df.columns:
        cols.append(participants_col)
    if group_col and group_col in df.columns:
        cols.append(group_col)

    sub = df[cols].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    color_col = group_col if plot_by_group and group_col in sub.columns else None

    # -------------------------
    # Plot
    # -------------------------
    col1, col2 = st.columns([3, 2], border=True, gap="xsmall")

    with col1:
        fig = px.scatter(sub, x=x_col, y=y_col, color=color_col,
                        trendline="ols", template="plotly_dark", height=500,
                        trendline_scope="overall",
                        trendline_color_override="red",
                        title=f"{y_col} vs {x_col}",
                        hover_data=[participants_col] if participants_col in sub.columns else None,
        )
        fig.update_layout(
            template="ggplot2",
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
        display_cols = [c for c in [x_col, y_col, color_col] if c is not None and c in sub.columns]
        st.dataframe(sub[display_cols].head(10), width='stretch')

    return x_col, y_col
