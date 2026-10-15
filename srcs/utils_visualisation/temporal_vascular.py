
"""Vascular and biomarker temporal visualizations."""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def fig_to_png_bytes(fig, dpi=300):
    """Serialize a Matplotlib figure as PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def draw_one_multi_y_grouped_boxplot_with_median_lines(
    df,
    group_col,
    y_cols,
    selected_groups=None,
    offset=0.20,
    box_width=0.08,
    axis_spacing=0.12,
    colors=None,
    markers=None,
    linestyles=None,
    figsize=(7, 5),
):
    """Draw grouped boxplots with two to four aligned y-axes."""
    if len(y_cols) < 2:
        raise ValueError("Pass at least 2 y columns.")
    if len(y_cols) > 4:
        raise ValueError("Supports up to 4 y columns per figure for readability.")

    fig, ax0 = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.78 + max(0, len(y_cols) - 2) * 0.10)

    sub = df[[group_col] + y_cols].copy()
    for c in y_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=[group_col])

    if selected_groups is not None and len(selected_groups) > 0:
        sub = sub[sub[group_col].isin(selected_groups)]
        groups_order = [f'{g}' for g in selected_groups if g in sub[group_col].unique()]
    else:
        groups_order = sorted(sub[group_col].dropna().unique().tolist())

    if len(groups_order) == 0:
        return fig, [ax0]

    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    if markers is None:
        markers = ["o", "s", "^", "D"]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"]

    axes = [ax0]
    for i in range(1, len(y_cols)):
        ax_new = ax0.twinx()
        if i >= 2:
            ax_new.spines["right"].set_position(("axes", 1 + (i - 1) * axis_spacing))
            ax_new.set_frame_on(True)
            ax_new.patch.set_visible(False)
        axes.append(ax_new)

    x = np.arange(len(groups_order))
    n = len(y_cols)
    x_offsets = np.linspace(-offset * (n - 1) / 2, offset * (n - 1) / 2, n)

    legend_handles = []
    legend_labels = []

    for i, (ax_i, col) in enumerate(zip(axes, y_cols)):
        stats = []
        medians = []

        for g in groups_order:
            vals = sub.loc[sub[group_col] == g, col].dropna().values

            if len(vals) == 0:
                stats.append(
                    dict(
                        label=g,
                        med=np.nan,
                        q1=np.nan,
                        q3=np.nan,
                        whislo=np.nan,
                        whishi=np.nan,
                        fliers=[]
                    )
                )
                medians.append(np.nan)
                continue

            q1 = np.percentile(vals, 25)
            med = np.percentile(vals, 50)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            valid_low = vals[vals >= lower_bound]
            valid_high = vals[vals <= upper_bound]

            whislo = valid_low.min() if len(valid_low) else q1
            whishi = valid_high.max() if len(valid_high) else q3

            stats.append(
                dict(
                    label=g,
                    med=med,
                    q1=q1,
                    q3=q3,
                    whislo=whislo,
                    whishi=whishi,
                    fliers=[]
                )
            )
            medians.append(med)

        pos = x + x_offsets[i]

        ax_i.bxp(
            stats,
            positions=pos,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=0.22,
                linewidth=1.2
            ),
            whiskerprops=dict(color=colors[i], linewidth=1.0),
            capprops=dict(color=colors[i], linewidth=1.0),
            medianprops=dict(color=colors[i], linewidth=1.8)
        )

        line, = ax_i.plot(
            pos,
            medians,
            color=colors[i],
            marker=markers[i],
            linestyle=linestyles[i],
            linewidth=2,
            markersize=5,
            label=col,
            zorder=3
        )

        ax_i.set_ylabel(col, color=colors[i])
        ax_i.tick_params(axis="y", colors=colors[i])

        if i == 0:
            ax_i.spines["left"].set_color(colors[i])
        else:
            ax_i.spines["right"].set_color(colors[i])

        legend_handles.append(line)
        legend_labels.append(col)

    group_counts = sub[group_col].value_counts()
    x_labels = [f"{g}\n(n={group_counts.get(g, 0)})" for g in groups_order]

    ax0.set_xticks(x)
    ax0.set_xticklabels(x_labels, rotation=0)

    # ax0.set_xticks(x)
    # ax0.set_xticklabels(groups_order, rotation=0)
    ax0.set_xlabel(group_col)
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(legend_handles, legend_labels, loc="best")
    fig.tight_layout()

    return fig, axes


def render_plot_block(
    df,
    group_col,
    y_cols,
    selected_groups,
    offset,
    box_width,
    axis_spacing,
    title,
    download_name,
):
    """Render and expose downloads for one vascular plot block."""
    use_cols = [group_col] + y_cols
    fig, axes = draw_one_multi_y_grouped_boxplot_with_median_lines(
        df=df[use_cols].copy(),
        group_col=group_col,
        y_cols=y_cols,
        selected_groups=selected_groups,
        offset=offset,
        box_width=box_width,
        axis_spacing=axis_spacing,
        figsize=(6, 4.8),
    )

    axes[0].set_title(title)
    st.pyplot(fig, width="stretch")

    png_bytes = fig_to_png_bytes(fig, dpi=300)
    st.download_button(
        label=f"Download {download_name}",
        data=png_bytes,
        file_name=f"{download_name}.png",
        mime="image/png",
        key=f"download_{download_name}"
    )

    plt.close(fig)


def render_vascular_temporal_dynamics(
    df,
    group_col,
):
    """Render grouped vascular and biomarker comparisons."""
    st.subheader("Vascular Dynamics")

    if df is None:
        st.warning("Upload data first.")
        return

    candidate_cols = [
        c for c in df.columns
        if c != group_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(candidate_cols) < 2:
        st.warning("At least two numeric columns are required.")
        return


    if 'vascular_age_y_cols' not in st.session_state:
        st.write('not present')


    group_options = df[group_col].dropna().unique().tolist()

    c1, c2, c3, c4 = st.columns([1.5, 1.0, 1.0, 1.0], gap="small")

    with c1:
        with st.container():
            vascular_age_y_cols = st.multiselect("Select vascular age columns",
                options=candidate_cols,
                key="vascular_age_y_cols",
                help="One or more vascular age markers."
            )

        with st.container():
            vascular_fitness_y_cols = st.multiselect("Select biomarker columns",
                options=candidate_cols,
                key="vascular_fitness_y_cols",
                help="One or more vascular fitness markers."
            )


    with c2:
        selected_groups = st.multiselect("Groups",
            options=group_options,
            key="vascular_selected_groups"
        )


    with c3:
        with st.container():
            plots_per_row = st.slider("Plots per row",
                min_value=1,
                max_value=4,
                value=2,
                key="vascular_plots_per_row"
            )

        with st.container():
            plot_mode = st.radio("Plot mode",
                options=["Separate", "Combined", ],
                horizontal=True,
                key="vascular_plot_mode",
                help="Combined: one plot per age marker. Separate: one plot per age-fitness pair." \
                    "Combined mode: if 2 age and 2 fitness columns are selected, you get 2 plots. " \
                    "Separate mode: you get 4 plots, one for each age-fitness combination."
            )

    with c4:
        with st.container(height = 'content'):
            offset = st.slider("Horizontal offset",
                min_value=0.05,
                max_value=0.40,
                value=0.22,
                step=0.01,
                key="vascular_offset"
            )

            box_width = st.slider( "Box width",
                min_value=0.03,
                max_value=0.20,
                value=0.08,
                step=0.01,
                key="vascular_box_width"
            )

            axis_spacing = st.slider("label Axis spacing",
                    min_value=0.1,
                    max_value=0.30,
                    value=0.14,
                    step=0.01,
                    key="vascular_axis_spacing"
                )


    if not vascular_age_y_cols:
        st.info("Please select at least one vascular age column.")
        return

    if not vascular_fitness_y_cols:
        st.info("Please select at least one vascular fitness column.")
        return

    if not selected_groups:
        st.info("Please select at least one group.")
        return

    plot_specs = []

    if plot_mode == "Combined":
        for age_col in vascular_age_y_cols:
            y_cols = list(dict.fromkeys([age_col] + vascular_fitness_y_cols))

            if len(y_cols) > 4:
                st.warning(
                    f"Skipping combined plot for {age_col}: {len(y_cols)} y-columns selected. "
                    f"Keep total y-columns per figure to 4 or fewer."
                )
                continue

            plot_specs.append({
                "title": f"{age_col} vs fitness markers",
                "download_name": f"{age_col}_vs_fitness_markers",
                "y_cols": y_cols,
            })

    elif plot_mode == "Separate":
        for age_col in vascular_age_y_cols:
            for fitness_col in vascular_fitness_y_cols:
                y_cols = [age_col, fitness_col]
                plot_specs.append({
                    "title": f"{age_col} vs {fitness_col}",
                    "download_name": f"{age_col}_vs_{fitness_col}",
                    "y_cols": y_cols,
                })

    if len(plot_specs) == 0:
        st.warning("No valid plots could be generated.")
        return

    st.subheader("Plots")
    filtered_df = df[[group_col,
                      *vascular_age_y_cols,
                      *vascular_fitness_y_cols]].dropna()

    for row_start in range(0, len(plot_specs), plots_per_row):
        row_specs = plot_specs[row_start: row_start + plots_per_row]
        row_cols = st.columns(plots_per_row)

        for container, spec in zip(row_cols, row_specs):
            with container:
                render_plot_block(
                    df=filtered_df,
                    group_col=group_col,
                    y_cols=spec["y_cols"],
                    selected_groups=selected_groups,
                    offset=offset,
                    box_width=box_width,
                    axis_spacing=axis_spacing,
                    title=spec["title"],
                    download_name=spec["download_name"],
                )

    with st.expander("Show selected columns preview"):
        preview_cols = list(dict.fromkeys(
            [group_col] + vascular_age_y_cols + vascular_fitness_y_cols
        ))
        st.dataframe(
            df[preview_cols].copy(),
            width="stretch",
            height=300,
            hide_index=True
        )
