"""Shared data-loading and session-state helpers."""

import re
import hashlib
import copy
from collections import defaultdict
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from srcs.utils_visualisation.plot_style import apply_graph_font

def set_paper_plot_style():
    """Apply the application's default Matplotlib style."""
    plt.style.use("seaborn-v0_8")
    apply_graph_font()




DEFAULT_SESSION_STATE = {
    "df": None,
    "df_name": None,
    "group_col": None,
    "continuous_cols": [],
    "categorical_cols": [],
    "participants_col": None,
    "sheet_name": None,
    "upload_digest": None,
    "temporal_cols": [],
    "temporal_families": {},
    "temporal_summary": None,
    "temporal_columns_signature": None,
}


NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def normalize_temporal_template(template):
    """Normalize separators so equivalent numbered columns group together."""
    template = str(template).strip().lower()
    template = re.sub(r"[\s_-]+", "_", template)
    template = re.sub(r"_*\{time\}_*", "{time}", template)
    template = re.sub(r"_*([\[\]()])_*", r"\1", template)
    return template.strip("_")


def temporal_template_to_family_name(template):
    """Turn a numbered-column template into a readable family name."""
    family = re.sub(
        r"(?i)(?:^|[\s_-]+)T[p]?[\s_-]*\{time\}",
        "",
        template,
    )
    units = (
        r"seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|"
        r"months?|years?"
    )
    family = re.sub(
        rf"(?i)[\s_-]*\{{time\}}\s*(?:{units})",
        "",
        family,
    )
    family = re.sub(
        rf"(?i)[\s_-]*(?:{units})[\s_-]*\{{time\}}",
        "",
        family,
    )
    family = family.replace("{time}", "")
    family = re.sub(r"_{2,}", "_", family)
    family = re.sub(r"\s{2,}", " ", family)
    family = re.sub(r"[-_]\s*$", "", family)
    family = re.sub(r"^\s*[-_]", "", family)
    family = re.sub(r"[_-]+\s*(?=\[)", " ", family)
    return family.strip(" _-")


def generate_number_templates(column):
    """Return a candidate template for every number in a column name."""
    column = str(column).strip()
    candidates = []
    for number_position, match in enumerate(NUMBER_PATTERN.finditer(column)):
        time_text = match.group()
        time_value = float(time_text)
        if time_value.is_integer():
            time_value = int(time_value)
        template = f"{column[:match.start()]}{{time}}{column[match.end():]}"
        candidates.append({
            "column": column,
            "template": template,
            "normalized_template": normalize_temporal_template(template),
            "time": time_value,
            "time_text": time_text,
            "number_position": number_position,
        })
    return candidates


def identify_repeated_measure_families(df, minimum_timepoints=2):
    """Find column families that differ at one numeric measurement position."""
    if df is None:
        return [], {}, pd.DataFrame()

    template_candidates = defaultdict(list)
    for column in df.columns:
        for candidate in generate_number_templates(column):
            template_candidates[candidate["normalized_template"]].append(candidate)

    families = {}
    for candidates in template_candidates.values():
        unique_candidates = {
            (
                candidate["column"],
                candidate["time"],
                candidate["number_position"],
            ): candidate
            for candidate in candidates
        }
        candidates = list(unique_candidates.values())
        distinct_columns = {item["column"] for item in candidates}
        distinct_times = {item["time"] for item in candidates}
        if (
            len(distinct_columns) < minimum_timepoints
            or len(distinct_times) < minimum_timepoints
        ):
            continue

        candidates.sort(key=lambda item: (item["time"], item["column"]))
        template = candidates[0]["template"]
        family_name = temporal_template_to_family_name(template)
        family_key = (family_name, normalize_temporal_template(template))
        families[family_key] = {
            "family": family_name,
            "template": template,
            "timepoints": [item["time"] for item in candidates],
            "columns": [item["column"] for item in candidates],
            "items": candidates,
        }

    repeated_columns = list(dict.fromkeys(
        column
        for family_data in families.values()
        for column in family_data["columns"]
    ))
    summary_rows = [
        {
            "family": family_data["family"],
            "template": family_data["template"],
            "timepoint": item["time"],
            "timepoint_text": item["time_text"],
            "column": item["column"],
            "number_of_timepoints": len(set(family_data["timepoints"])),
        }
        for family_data in families.values()
        for item in family_data["items"]
    ]
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["family", "template", "timepoint"]
        ).reset_index(drop=True)
    return repeated_columns, families, summary


def get_repeated_measure_families(df=None, minimum_timepoints=2, force=False):
    """Return detected families and keep them available in Streamlit state."""
    if df is None:
        df = st.session_state.get("df")

    signature = (
        tuple(map(str, df.columns)) if df is not None else None,
        minimum_timepoints,
    )
    if force or st.session_state.get("temporal_columns_signature") != signature:
        columns, families, summary = identify_repeated_measure_families(
            df,
            minimum_timepoints=minimum_timepoints,
        )
        st.session_state["temporal_cols"] = columns
        st.session_state["temporal_families"] = families
        st.session_state["temporal_summary"] = summary
        st.session_state["temporal_columns_signature"] = signature

    return (
        st.session_state.get("temporal_cols", []),
        st.session_state.get("temporal_families", {}),
        st.session_state.get("temporal_summary", pd.DataFrame()),
    )


def init_session_state():
    """Initialize shared Streamlit state without replacing existing values."""
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def read_csv_file(file_obj):
    """Read an uploaded CSV file."""
    return pd.read_csv(file_obj)


def upload_content_digest(content):
    """Return a stable identity for uploaded bytes."""
    return hashlib.sha256(content).hexdigest()


def upload_requires_reload(current_digest, new_digest, current_sheet=None, new_sheet=None):
    """Return whether file content or the selected workbook sheet changed."""
    return current_digest != new_digest or current_sheet != new_sheet


def _clear_loaded_dataset():
    """Remove a dataset and all state derived from it after a failed replacement."""
    for key in list(st.session_state):
        if key in DEFAULT_SESSION_STATE or key.endswith("_result") or key.endswith("_results"):
            value = DEFAULT_SESSION_STATE.get(key)
            st.session_state[key] = copy.deepcopy(value)
    st.session_state["df"] = None
    st.session_state["df_name"] = None
    st.session_state["sheet_name"] = None
    st.session_state["upload_digest"] = None


def get_excel_sheet_names(file_obj):
    """Return workbook sheet names and reusable file bytes."""
    content = file_obj.read()
    file_obj.seek(0)
    xls = pd.ExcelFile(BytesIO(content))
    return xls.sheet_names, content


def read_excel_sheet(file_bytes, sheet_name=0):
    """Read one sheet from uploaded Excel bytes."""
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return pd.read_excel(xls, sheet_name=sheet_name)


def infer_variable_types(df, group_col=None, unique_threshold=12):
    """Infer continuous and categorical columns from dtypes and cardinality."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    object_like_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    low_cardinality_numeric = [
        c for c in numeric_cols
        if c != group_col and df[c].nunique(dropna=True) <= unique_threshold
    ]

    categorical_cols = list(dict.fromkeys(object_like_cols + low_cardinality_numeric))
    continuous_cols = [c for c in numeric_cols if c != group_col and c not in categorical_cols]
    return continuous_cols, categorical_cols


def choose_default_group_col(df):
    """Choose a likely grouping column, preferring common clinical names."""
    if df is None or df.empty:
        return None

    priority_names = [
        "AgeGroup", "group_label", "group", "Group", "GROUP", "Cohort",
        "status", "chohort_exp_age_group", "sex", "Group_biologicalAge",
    ]
    for column in priority_names:
        if column in df.columns:
            return column

    candidate_cat = [
        column for column in df.columns
        if str(df[column].dtype) in {"object", "category", "bool"}
        or df[column].nunique(dropna=True) <= 12
    ]
    return candidate_cat[0] if candidate_cat else None


def get_state():
    """Return the analysis state used by all pages."""
    return {
        "df": st.session_state.get("df"),
        "group_col": st.session_state.get("group_col"),
        "continuous_cols": st.session_state.get("continuous_cols", []),
        "categorical_cols": st.session_state.get("categorical_cols", []),
        "participants_col": st.session_state.get("participants_col"),
        "temporal_cols": st.session_state.get("temporal_cols", []),
        "temporal_families": st.session_state.get("temporal_families", {}),
        "temporal_summary": st.session_state.get("temporal_summary"),
    }


def _reset_variable_state(df):
    """Reset inferred variable selections for a newly loaded dataset."""
    preserved = {"stats_file_uploader", "excel_sheet_selectbox"}
    for key in list(st.session_state):
        if key not in preserved and (
            key.endswith(("_multiselect", "_selectbox", "_result", "_results"))
            or key.startswith(("corr_", "group_comp_", "uniols_", "multiols_", "resp_", "temporal_"))
        ):
            del st.session_state[key]
    default_group = choose_default_group_col(df)
    cont_cols, cat_cols = infer_variable_types(df, group_col=default_group)

    st.session_state["group_col"] = default_group
    st.session_state["continuous_cols"] = cont_cols[: min(len(cont_cols), 100)]
    st.session_state["categorical_cols"] = [c for c in cat_cols if c != default_group][: min(len(cat_cols), 8)]
    get_repeated_measure_families(df, minimum_timepoints=2, force=True)


def load_dataframe_ui():
    """Render the upload controls and return the current dataframe."""
    st.sidebar.header("1. Load data")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="stats_file_uploader",
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_type = file_name.lower().split(".")[-1]
        file_bytes = uploaded_file.getvalue()
        content_digest = upload_content_digest(file_bytes)

        if file_type == "csv":
            needs_reload = upload_requires_reload(
                st.session_state.get("upload_digest"), content_digest
            )
            if needs_reload:
                try:
                    df = read_csv_file(BytesIO(file_bytes))
                    st.session_state["df"] = df
                    st.session_state["df_name"] = file_name
                    st.session_state["sheet_name"] = None
                    st.session_state["upload_digest"] = content_digest
                    _reset_variable_state(df)
                    st.sidebar.success("CSV loaded successfully.")
                except Exception as e:
                    _clear_loaded_dataset()
                    st.sidebar.error(f"Could not load CSV: {e}")

        elif file_type in ["xlsx", "xls"]:
            try:
                xls = pd.ExcelFile(BytesIO(file_bytes))
                sheet_names = xls.sheet_names
                selected_sheet = st.sidebar.selectbox(
                    "Excel sheet",
                    options=sheet_names,
                    index=(
                        sheet_names.index(st.session_state["sheet_name"])
                        if st.session_state.get("sheet_name") in sheet_names
                        else 0
                    ),
                    key="excel_sheet_selectbox",
                )

                needs_reload = upload_requires_reload(
                    st.session_state.get("upload_digest"), content_digest,
                    st.session_state.get("sheet_name"), selected_sheet,
                )

                if needs_reload:
                    df = read_excel_sheet(file_bytes, selected_sheet)
                    st.session_state["df"] = df
                    st.session_state["df_name"] = file_name
                    st.session_state["sheet_name"] = selected_sheet
                    st.session_state["upload_digest"] = content_digest
                    _reset_variable_state(df)
                    st.sidebar.success("Excel sheet loaded successfully.")
            except Exception as e:
                _clear_loaded_dataset()
                st.sidebar.error(f"Could not load Excel file: {e}")

    return st.session_state.get("df")

def is_potential_id_col(series):
    """Return whether a column is sufficiently complete and unique to be an ID."""
    if series.empty or series.isna().mean() > 0.1:
        return False
    return series.nunique() / len(series) >= 0.5


def render_sidebar_variable_selector(df):
    """Render shared variable selectors and store their current values."""
    st.sidebar.header("2. Variable selection")

    if df is None:
        st.sidebar.info("Upload a file first.")
        return

    all_cols = df.columns.tolist()
    if not all_cols:
        st.sidebar.warning("No columns found.")
        return

    id_candidates = [column for column in all_cols if is_potential_id_col(df[column])]

    current_participant = st.session_state.get("participants_col")
    if current_participant not in id_candidates and current_participant is not None:
        current_participant = None

    id_options = id_candidates + ["None"]
    idx = id_options.index(current_participant or "None")

    selected_participant = st.sidebar.selectbox(
        "Participants column Identifier (optional)",
        options=id_options,
        index=idx,
        key="participants_col_selectbox",
    )

    st.session_state["participants_col"] = (
        None if selected_participant == "None" else selected_participant
    )

    previous_continuous = st.session_state.get("continuous_cols", [])
    valid_continuous_default = [c for c in previous_continuous if c in all_cols]

    st.session_state["continuous_cols"] = st.sidebar.multiselect(
        "Continuous columns",
        options=all_cols,
        default=valid_continuous_default,
        key="continuous_cols_multiselect",
    )

    saved_cont = st.session_state["continuous_cols"]

    categorical_options = [c for c in all_cols if c not in saved_cont]

    previous_categorical = st.session_state.get("categorical_cols", [])
    valid_categorical_default = [c for c in previous_categorical if c in categorical_options]

    st.session_state["categorical_cols"] = st.sidebar.multiselect(
        "Categorical columns",
        options=categorical_options,
        default=valid_categorical_default,
        key="categorical_cols_multiselect",
    )

    st.sidebar.markdown("### Session selections")
    st.sidebar.caption(f"Participants: {st.session_state['participants_col']}")
    st.sidebar.caption(f"Continuous: {len(st.session_state['continuous_cols'])}")
    st.sidebar.caption(f"Categorical: {len(st.session_state['categorical_cols'])}")
