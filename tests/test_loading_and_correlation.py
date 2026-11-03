from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd

from srcs.utils_features.common import (
    read_csv_file, read_excel_sheet, upload_content_digest, upload_requires_reload,
)
from srcs.utils_visualisation.correlation_plots import plot_correlation_heatmap_with_significance_plotly
import srcs.utils_features.common as loading


def test_same_name_csv_and_excel_content_changes_are_detectable():
    csv1, csv2 = b"x\n1\n", b"x\n2\n"
    assert upload_content_digest(csv1) != upload_content_digest(csv2)
    digest = upload_content_digest(csv1)
    assert not upload_requires_reload(digest, digest)
    assert upload_requires_reload(digest, upload_content_digest(csv2))
    assert read_csv_file(BytesIO(csv2)).iloc[0, 0] == 2

    buffers = []
    for value in (1, 2):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({"x": [value]}).to_excel(writer, sheet_name="same", index=False)
        buffers.append(buffer.getvalue())
    assert upload_content_digest(buffers[0]) != upload_content_digest(buffers[1])
    assert read_excel_sheet(buffers[1], "same").iloc[0, 0] == 2
    assert not upload_requires_reload(
        upload_content_digest(buffers[1]), upload_content_digest(buffers[1]), "same", "same"
    )
    assert upload_requires_reload(
        upload_content_digest(buffers[1]), upload_content_digest(buffers[1]), "first", "same"
    )


def test_failed_replacement_clears_old_dataset_and_results(monkeypatch):
    state = {
        "df": pd.DataFrame({"old": [1]}), "df_name": "data.csv",
        "sheet_name": None, "upload_digest": "old", "analysis_results": {"stale": True},
    }
    monkeypatch.setattr(loading, "st", SimpleNamespace(session_state=state))
    loading._clear_loaded_dataset()
    assert state["df"] is None
    assert state["df_name"] is None
    assert state["upload_digest"] is None
    assert state["analysis_results"] is None


def test_correlation_threshold_masks_display_only_and_scale_is_fixed():
    df = pd.DataFrame({"x": range(10), "strong": np.arange(10) * 2, "weak": [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]})
    figures = {
        threshold: plot_correlation_heatmap_with_significance_plotly(df, min_abs_corr=threshold)
        for threshold in (0, 0.3, 0.8)
    }
    counts = [np.isfinite(np.asarray(figures[t].data[0].z, dtype=float)).sum() for t in (0, 0.3, 0.8)]
    assert counts[0] >= counts[1] >= counts[2]
    assert figures[0].data[0].zmin == -1
    assert figures[0].data[0].zmax == 1
    assert "|r| ≥ 0.80" in figures[0.8].layout.title.text
