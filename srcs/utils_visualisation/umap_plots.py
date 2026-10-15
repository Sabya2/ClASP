"""UMAP embedding, visualization, and download helpers."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler


def _render_umap_downloads(fig: go.Figure, embedding_df: pd.DataFrame, filename_stem: str = "umap_projection"):
    with st.expander("⬇️ Download this chart", expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            try:
                png_bytes = fig.to_image(format="png", scale=2)
                st.download_button(
                    label="📥 Download PNG",
                    data=png_bytes,
                    file_name=f"{filename_stem}.png",
                    mime="image/png",
                    width="stretch",
                )
            except Exception as e:
                st.warning(f"PNG export requires `kaleido`: `pip install kaleido`\n\n{e}")

        with c2:
            html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button(
                label="📥 Download HTML",
                data=html_bytes,
                file_name=f"{filename_stem}.html",
                mime="text/html",
                width="stretch",
            )

        with c3:
            csv_bytes = embedding_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name=f"{filename_stem}.csv",
                mime="text/csv",
                width="stretch",
            )



def compute_umap_embedding(
    df: pd.DataFrame,
    feature_cols: list = None,
    color_col: str = None,
    hover_cols: list = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    scale_data: bool = True,
    random_state: int = 42,
):
    """Compute a two-dimensional UMAP embedding from selected columns."""
    work_df = df.copy()

    if feature_cols is None:
        exclude_cols = [color_col] if color_col is not None else []
        X = work_df.drop(columns=exclude_cols, errors="ignore")
        X = X.select_dtypes(include=np.number)
        feature_cols = X.columns.tolist()
    else:
        missing_cols = [c for c in feature_cols if c not in work_df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        X = work_df[feature_cols].copy()
        X = X.select_dtypes(include=np.number)

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns available for UMAP.")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    if scale_data:
        X_input = StandardScaler().fit_transform(X)
    else:
        X_input = X.to_numpy()

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(X_input)

    embedding_df = pd.DataFrame(
        embedding,
        columns=["UMAP1", "UMAP2"],
        index=work_df.index,
    )

    if color_col is not None and color_col in work_df.columns:
        embedding_df[color_col] = work_df[color_col].values

    if hover_cols:
        for col in hover_cols:
            if col in work_df.columns:
                embedding_df[col] = work_df[col].values

    return embedding_df, feature_cols



def plot_umap_plotly(
    embedding_df: pd.DataFrame,
    color_col: str = None,
    hover_cols: list = None,
    title: str = "UMAP Projection",
    width: int = 950,
    height: int = 750,
    point_size: int = 8,
    opacity: float = 0.8,
):
    """Build an interactive Plotly view of a UMAP embedding."""
    hover_cols = hover_cols or []

    if color_col is not None and color_col in embedding_df.columns:
        fig = px.scatter(
            embedding_df,
            x="UMAP1",
            y="UMAP2",
            color=embedding_df[color_col].astype(str),
            hover_data=hover_cols,
            title=title,
        )
    else:
        fig = px.scatter(
            embedding_df,
            x="UMAP1",
            y="UMAP2",
            hover_data=hover_cols,
            title=title,
        )

    fig.update_traces(
        marker=dict(size=point_size, opacity=opacity),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        width=width,
        height=height,
        template="simple_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=70, b=40),
        legend_title_text=color_col if color_col else None,
    )

    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False, scaleanchor=None)

    return fig


def render_umap_page(df, continuous_cols=None, group_col=None, participants_col=None):
    """Render UMAP controls, visualization, and downloads."""
    st.header("UMAP projection")
    with st.expander("How to use UMAP options", expanded=False):
        st.markdown("""
            **Color by**
            - Chooses which column is used to color the UMAP points.
            - Best for categorical variables such as sex, group, diagnosis, cluster, or site.
            - If you choose no color column, all points are shown with the same color.

            **Hover columns**
            - These columns are shown when you move the mouse over a point.
            - Useful for participant ID, group, age, diagnosis, or other metadata.
            - Hover columns do not affect the UMAP calculation; they only help interpretation.

            **Metric**
            - This defines how similarity or distance is computed in the original feature space before UMAP creates the 2D embedding.
            - `euclidean`: good default for standardized continuous variables.
            - `manhattan`: can be useful when absolute differences are more meaningful than straight-line distance.
            - `cosine`: useful when the direction or profile pattern matters more than magnitude, often helpful in high-dimensional data.
            - `correlation`: useful when you want samples with similar variable patterns to be close, even if their absolute levels differ.
            - `chebyshev`: emphasizes the largest single-variable difference between two observations.

            **Important**
            - The UMAP axes themselves usually do not have direct clinical meaning.
            - UMAP mainly helps you inspect clustering, separation, gradients, and outliers visually.
            """)

    if df is None:
        st.warning("Upload data first.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    default_cols = continuous_cols if continuous_cols else numeric_cols

    c1, c2, c3 = st.columns([2.2, 1.1, 1.1], gap="small")

    with c1:
        feature_cols = st.multiselect(
            "Variables for UMAP",
            options=df.columns.tolist(),
            default=default_cols[: min(15, len(default_cols))],
            key="umap_feature_cols",
        )

    with c2:
        color_options = [None] + df.columns.tolist()
        default_color_idx = color_options.index(group_col) if group_col in color_options else 0
        color_col = st.selectbox(
            "Color by",
            options=color_options,
            index=default_color_idx,
            key="umap_color_col",
            help="Column used only for point coloring; categorical labels work best.",
        )

    with c3:
        scale_data = st.checkbox(
            "Standardize features",
            value=True,
            key="umap_scale_data",
            help="Standardize numeric features before UMAP.",
        )

    p1, p2, p3, p4 = st.columns(4, gap="small")

    with p1:
        n_neighbors = st.slider(
            "n_neighbors",
            min_value=2,
            max_value=100,
            value=15,
            step=1,
            key="umap_n_neighbors",
        )

    with p2:
        min_dist = st.slider(
            "min_dist",
            min_value=0.0,
            max_value=0.99,
            value=0.10,
            step=0.01,
            key="umap_min_dist",
        )

    with p3:
        metric = st.selectbox(
            "Metric",
            options=["euclidean", "manhattan", "cosine", "correlation", "chebyshev"],
            index=0,
            key="umap_metric",
            help = "Distance metric used " \
            "by UMAP in the original feature space. " \
            "Euclidean: is a common default; "\
            "Chebyshev: emphasizes the largest single-variable difference between two observations."\
\
            "Manhattan: can be useful when absolute differences are more meaningful than straight-line distance." \

            "cosine or correlation: can work better for high-dimensional pattern-based data."
        )

    with p4:
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
            key="umap_random_state",
        )

    hover_default = []
    if participants_col and participants_col in df.columns:
        hover_default.append(participants_col)
    if color_col and color_col in df.columns and color_col not in hover_default:
        hover_default.append(color_col)

    hover_cols = st.multiselect(
        "Hover columns",
        options=df.columns.tolist(),
        default=hover_default,
        key="umap_hover_cols",
    )

    if len(feature_cols) < 2:
        st.info("Select at least two numeric variables.")
        return

    run_umap = st.button("Run UMAP", key="run_umap_btn", width="content")

    if run_umap:
        try:
            embedding_df, used_features = compute_umap_embedding(
                df=df,
                feature_cols=feature_cols,
                color_col=color_col,
                hover_cols=hover_cols,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                scale_data=scale_data,
                random_state=int(random_state),
            )

            title = (
                f"UMAP projection "
                f"(n_neighbors={n_neighbors}, min_dist={min_dist:.2f}, metric={metric})"
            )

            fig = plot_umap_plotly(
                embedding_df=embedding_df,
                color_col=color_col,
                hover_cols=hover_cols,
                title=title,
                width=950,
                height=750,
                point_size=8,
                opacity=0.82,
            )

            st.plotly_chart(fig, width="stretch")

            st.caption(
                f"Rows embedded: {embedding_df.shape[0]} | "
                f"Features used: {len(used_features)}"
            )

            with st.expander("Preview embedding table", expanded=False):
                st.dataframe(embedding_df.head(50), width="stretch")

            _render_umap_downloads(fig, embedding_df, filename_stem="umap_projection")

        except Exception as e:
            st.error(f"Error running UMAP: {e}")
