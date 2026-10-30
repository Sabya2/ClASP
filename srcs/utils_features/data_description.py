"""Dataset summaries and downloadable publication tables."""

import re
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np

from scipy.stats import f_oneway, kruskal, chi2_contingency
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment


def get_basic_info(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact overview of the dataset structure."""
    info_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_values": df.isna().sum().values,
        "missing_percent": (df.isna().mean() * 100).round(2).values,
        "unique_values": df.nunique(dropna=True).values,
    })

    return (
        info_df
        .sort_values(["missing_values", "unique_values"], ascending=[False, False])
        .reset_index(drop=True)
    )


def check_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns whose names contain whitespace or special characters."""
    issues = []

    for col in df.columns:
        col_name = str(col)
        problems = []

        if col_name != col_name.strip():
            problems.append("Leading/trailing whitespace")

        # Letters, numbers, and underscores are the supported characters.
        if re.search(r"[^a-zA-Z0-9_]", col_name):
            problems.append("Special character or whitespace")

        if problems:
            issues.append({
                "Column_Name": repr(col_name),
                "Issues": ", ".join(problems),
            })

    return pd.DataFrame(issues, columns=["Column_Name", "Issues"])


def format_p_value(p: float) -> str:
    """Format a p-value for display."""
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"

def cohens_d(x, y):
    """Calculate Cohen's d using the pooled sample standard deviation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_sd


def format_effect_size_d(d):
    """Format Cohen's d for display."""
    if pd.isna(d):
        return "NA"
    return f"{d:.2f}"

def build_continuous_summary_table(
    df: pd.DataFrame,
    cont_cols: list,
    group_col: str = None,
    split_by_group: bool = False,
    round_digits: int = 2,
    summary_format: str = "mean_sd",
) -> pd.DataFrame:
    """
    Create a summary table for continuous variables.

    Default:
        Only full cohort summary is shown.

    If split_by_group=True:
        Full cohort + each group + p-value are shown.
    """

    def summarize_series(series: pd.Series):
        """
        Convert a column to numeric, remove missing values,
        and return a formatted summary string plus clean values.
        """
        values = pd.to_numeric(series, errors="coerce").dropna()
        n = len(values)

        if n == 0:
            return "NA", None

        if summary_format == "mean_sd":
            mean_val = values.mean()
            sd_val = values.std(ddof=1) if n > 1 else 0.0
            summary = f"{mean_val:.{round_digits}f} ± {sd_val:.{round_digits}f} (n={n})"

        elif summary_format == "median_iqr":
            median_val = np.median(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            summary = f"{median_val:.{round_digits}f} ± {iqr:.{round_digits}f} (n={n})"

        else:
            raise ValueError("summary_format must be 'mean_sd' or 'median_iqr'")

        return summary, values

    rows = []

    # Find group names only if group-based split is requested
    if split_by_group and group_col:
        groups = list(pd.Series(df[group_col]).dropna().unique())
    else:
        groups = []

    for col in cont_cols:
        row = {"variable": col}

        # Full cohort summary
        full_summary, _ = summarize_series(df[col])
        row["Full cohort"] = full_summary

        # Optional grouped summary
        if split_by_group and group_col:
            valid_arrays = []
            group_value_map = {}

            for group in groups:
                group_values = df.loc[df[group_col] == group, col]
                group_summary, clean_values = summarize_series(group_values)
                row[str(group)] = group_summary

                if clean_values is not None and len(clean_values) > 0:
                    valid_arrays.append(clean_values.values)
                    group_value_map[str(group)] = clean_values.values
                else:
                    group_value_map[str(group)] = np.array([])

            # Statistical test across groups
            if len(valid_arrays) >= 2:
                try:
                    if summary_format == "mean_sd":
                        _, pval = f_oneway(*valid_arrays)
                    else:
                        _, pval = kruskal(*valid_arrays)
                except Exception:
                    pval = np.nan
            else:
                pval = np.nan

            row["p_value"] = format_p_value(pval)

            # Cohen's d only for exactly 2 groups
            if len(groups) == 2:
                g1, g2 = [str(g) for g in groups]
                d_val = cohens_d(group_value_map[g1], group_value_map[g2])
                row["cohens_d"] = format_effect_size_d(d_val)

        rows.append(row)

    # Decide final column order
    if split_by_group and group_col:
        column_order = ["variable", "Full cohort"] + [str(g) for g in groups] + ["p_value"]
        if len(groups) == 2:
            column_order += ["cohens_d"]
    else:
        column_order = ["variable", "Full cohort"]

    if rows:
        return pd.DataFrame(rows)[column_order]

    return pd.DataFrame(columns=column_order)


# =========================================================
# Categorical summary table
# =========================================================



def build_categorical_summary_table(
    df: pd.DataFrame,
    cat_cols: list,
    group_col: str = None,
    split_by_group: bool = False,
    round_digits: int = 1,
) -> pd.DataFrame:
    """
    Create a summary table for categorical variables.

    Default:
        Only full cohort summary is shown.

    If split_by_group=True:
        Full cohort + each group + p-value are shown.
    """
    rows = []

    if split_by_group and group_col is not None and group_col in df.columns:
        groups = list(pd.Series(df[group_col]).dropna().unique())
    else:
        groups = []

    for col in cat_cols:
        categories = pd.Series(df[col].dropna().unique()).tolist()

        pval = np.nan
        if split_by_group and group_col is not None and group_col in df.columns:
            test_df = df[[group_col, col]].dropna(subset=[group_col, col]).copy()

            group_n = test_df[group_col].nunique(dropna=True)
            cat_n = test_df[col].nunique(dropna=True)

            if group_n >= 2 and cat_n >= 2:
                try:
                    contingency_table = pd.crosstab(test_df[col], test_df[group_col])
                    _, pval, _, _ = chi2_contingency(contingency_table)
                except Exception:
                    pval = np.nan

        full_total = df[col].notna().sum()

        for idx, category in enumerate(categories):
            row = {
                "variable": col,
                "category": category,
                "subcategory": f"{group_col}",
            }

            full_count = (df[col] == category).sum()
            full_pct = (100 * full_count / full_total) if full_total > 0 else 0
            row["Full cohort"] = f"{full_count} ({full_pct:.{round_digits}f}%)"

            if split_by_group and group_col is not None and group_col in df.columns:
                for group in groups:
                    group_df = df[df[group_col] == group]
                    total = group_df[col].notna().sum()
                    count = (group_df[col] == category).sum()
                    pct = (100 * count / total) if total > 0 else 0
                    row[str(group)] = f"{count} ({pct:.{round_digits}f}%)"

                row["p_value"] = format_p_value(pval) if idx == 0 else ""

            rows.append(row)

    if split_by_group and group_col is not None and group_col in df.columns:
        column_order = ["variable", "category", "subcategory", "Full cohort"] + [str(g) for g in groups] # + ["p_value"]
    else:
        column_order = ["variable", "category", "subcategory", "Full cohort"]

    if rows:
        return pd.DataFrame(rows)[column_order]

    return pd.DataFrame(columns=column_order)



# =========================================================
# Download helpers
# =========================================================

@st.cache_data
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert dataframe to CSV bytes for download.
    """
    return df.to_csv(index=False).encode("utf-8")


def autofit_columns(ws, min_width=12, max_width=45, padding=4):
    """
    Adjust Excel column widths based on cell content.
    """
    for col_idx, col_cells in enumerate(ws.columns, start=1):
        max_len = 0

        for cell in col_cells:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))

        adjusted_width = min(max(max_len + padding, min_width), max_width)
        ws.column_dimensions[get_column_letter(col_idx)].width = adjusted_width


@st.cache_data
def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """
    Convert dataframe to formatted Excel bytes for download.
    """
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.book[sheet_name]

        # Make columns wider for readability
        autofit_columns(ws, min_width=12, max_width=45, padding=4)

        # Center-align all cells and wrap long text
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True
                )

    return buffer.getvalue()


# =========================================================
# Display helper
# =========================================================

def render_publication_table(title: str, df: pd.DataFrame, file_prefix: str, sheet_name: str):
    """
    Render a clean table with CSV and Excel download buttons.
    """
    st.markdown(f"### {title}")

    col1, col2, col3 = st.columns([7, 1.5, 1.5])

    with col2:
        st.download_button(
            label="CSV",
            data=df_to_csv_bytes(df),
            file_name=f"{file_prefix}.csv",
            mime="text/csv",
            key=f"{file_prefix}_csv",
            width="stretch",
        )

    with col3:
        st.download_button(
            label="Excel",
            data=df_to_excel_bytes(df, sheet_name),
            file_name=f"{file_prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{file_prefix}_xlsx",
            width="stretch",
        )

    # Static table is cleaner than st.dataframe for publication-style display
    st.table(df)


# =========================================================
# Page renderer
# =========================================================


def render_data_description_page(
    df: pd.DataFrame,
    group_col: str,
    continuous_cols: list,
    categorical_cols: list,
):
    """
    Render the full Data Description page in Streamlit.
    """
    st.header("Data description")

    if df is None:
        st.warning("Please upload a CSV or Excel file.")
        return

    # -----------------------------------------------------
    # Dataset filtering section
    # -----------------------------------------------------
    df_filtered = df.copy()

    filter_col1, filter_col2 = st.columns([1, 2], gap="small")

    with filter_col1:
        apply_null_filter = st.checkbox(
            "Data filter - Remove rows with nulls based on selected columns",
            value=False,
            key="data_description_apply_null_filter",
        )

    with filter_col2:
        if apply_null_filter:
            filter_columns = st.multiselect(
                "Select column(s)",
                options=df.columns.tolist(),
                default=[],
                key="data_description_filter_columns",
            )

            if filter_columns:
                before_n = len(df_filtered)
                df_filtered = df_filtered.dropna(subset=filter_columns).copy()
                after_n = len(df_filtered)
                removed_n = before_n - after_n

                st.caption(
                    f"Removed {removed_n} rows with null values in the selected columns. Remaining rows: {after_n}."
                )
            else:
                st.info("Select at least one column to apply the null-row filter.")

    with st.expander("Data preview and structure", expanded=False):
        st.dataframe(get_basic_info(df_filtered), width="stretch")

    problematic_columns = check_column_names(df_filtered)
    if not problematic_columns.empty:
        st.error(
            "❗ Invalid column names detected. Use only letters, numbers, and "
            "underscores (`_`) in column names."
        )
        with st.expander("View invalid column names", expanded=False):
            issue_details = "\n".join(
                f"- `{row.Column_Name}` — {row.Issues}"
                for row in problematic_columns.itertuples(index=False)
            )
            st.error(f"**Affected columns**\n\n{issue_details}")

    # -----------------------------------------------------
    # Summary settings
    # -----------------------------------------------------
    summary_choice = st.radio(
        "Select summary type",
        options=["Mean ± SD", "Median + IQR"],
        horizontal=True,
        key="summary_choice_data_description",
    )

    summary_format = "mean_sd" if summary_choice == "Mean ± SD" else "median_iqr"

    # -----------------------------------------------------
    # Grouping options
    # -----------------------------------------------------
    split_by_group = st.checkbox(
        "Compare demographic table by group",
        value=False,
        key="demographic_split_by_group",
    )

    selected_group_col = None
    possible_group_cols = categorical_cols if categorical_cols else []

    if split_by_group:
        if not possible_group_cols:
            st.info("No categorical column available for grouping.")
            return

        default_idx = possible_group_cols.index(group_col) if group_col in possible_group_cols else 0

        selected_group_col = st.selectbox(
            "Select group column for comparison",
            options=possible_group_cols,
            index=default_idx,
            key="demographic_group_col",
            help="Select the categorical column used to compare differences between groups.",
        )

        if selected_group_col is None:
            st.info("Please select a group column to generate group-based summaries.")
            return

    # -----------------------------------------------------
    # Continuous summary
    # -----------------------------------------------------
    if continuous_cols and (not split_by_group or selected_group_col):
        cont_tbl = build_continuous_summary_table(
            df=df_filtered,
            cont_cols=continuous_cols,
            group_col=selected_group_col,
            split_by_group=split_by_group,
            round_digits=2,
            summary_format=summary_format,
        )

        render_publication_table(
            title="Continuous summary",
            df=cont_tbl,
            file_prefix="continuous_summary",
            sheet_name="Continuous Summary",
        )
    else:
        st.info("Select at least one continuous variable, and a group column if splitting.")

    # -----------------------------------------------------
    # Categorical summary
    # -----------------------------------------------------
    if categorical_cols and (not split_by_group or selected_group_col):
        # Optional: exclude the selected grouping variable from being summarized against itself
        categorical_vars_for_table = [
            c for c in categorical_cols
            if not (split_by_group and selected_group_col and c == selected_group_col)
        ]

        if categorical_vars_for_table:
            cat_tbl = build_categorical_summary_table(
                df=df_filtered,
                cat_cols=categorical_vars_for_table,
                group_col=selected_group_col,
                split_by_group=split_by_group,
                round_digits=1,
            )

            render_publication_table(
                title="Categorical summary",
                df=cat_tbl,
                file_prefix="categorical_summary",
                sheet_name="Categorical Summary",
            )
        else:
            st.info("No categorical variables available after excluding the selected group column.")
    else:
        st.info("Select at least one categorical variable, and a group column if splitting.")
