"""Testable descriptive summaries for temporal plots."""

import numpy as np
import pandas as pd


def summarize_median_trajectory(
    data,
    group_cols,
    value_col="Value",
    participant_col=None,
    confidence=0.95,
    n_boot=2000,
    random_state=20261102,
):
    """Summarize median trajectories with a percentile participant bootstrap CI.

    Missing values are removed. When a participant column is supplied, duplicate
    observations for a participant within a plotted cell are first reduced to a
    participant median, then whole participant values are resampled. A singleton
    or constant cell has a zero-width interval.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    columns = list(group_cols) + ([participant_col] if participant_col else []) + [value_col]
    clean = data.loc[:, list(dict.fromkeys(columns))].copy()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
    if participant_col:
        clean = (
            clean.groupby(list(group_cols) + [participant_col], dropna=False, as_index=False)[value_col]
            .median()
        )

    rng = np.random.default_rng(random_state)
    alpha = (1 - confidence) / 2
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    for key, group in clean.groupby(grouper, dropna=False, sort=False):
        values = group[value_col].to_numpy(dtype=float)
        n = values.size
        if n == 0:
            continue
        median = float(np.median(values))
        if n == 1 or np.all(values == values[0]):
            low = high = median
        else:
            draws = rng.choice(values, size=(n_boot, n), replace=True)
            boot_medians = np.median(draws, axis=1)
            low, high = np.quantile(boot_medians, [alpha, 1 - alpha])
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row.update({
            "median": median,
            "ci_low": float(low),
            "ci_high": float(high),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "std": float(np.std(values, ddof=1)) if n > 1 else np.nan,
            "n": int(n),
            "interval_method": f"{confidence:.0%} percentile bootstrap CI for median ({n_boot} resamples)",
        })
        rows.append(row)
    return pd.DataFrame(rows)
