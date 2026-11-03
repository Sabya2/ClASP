"""Testable descriptive summaries for temporal plots."""

import re
import hashlib

import numpy as np
import pandas as pd


def build_temporal_explanation(normalise, interval_kind, participant_col=None):
    """Return context-sensitive explanations for temporal plot settings."""
    if normalise not in {"Yes", "No"}:
        raise ValueError("normalise must be 'Yes' or 'No'")

    if normalise == "Yes":
        participant_text = (
            f" using participant identifier `{participant_col}`"
            if participant_col else ""
        )
        baseline = (
            "Each participant's first available numeric value in the selected family, "
            f"ordered by detected timepoint, is used as baseline{participant_text}. "
            "Values are shown as value / baseline × 100, so 100 represents that "
            "participant's baseline. A zero or missing baseline cannot be normalized; "
            "those normalized observations are omitted."
        )
    else:
        baseline = (
            "Baseline normalization is off. Values remain in their imported measurement "
            "units, and each plotted point is the median of the available values at that "
            "timepoint."
        )

    interval_explanations = {
        "95% bootstrap CI": (
            "The interval is a pointwise 95% percentile bootstrap confidence interval "
            "for the median, based on 2,000 participant-level resamples. It describes "
            "sampling uncertainty around the median; it is not an interval for an "
            "individual participant. Singleton and constant cells have zero-width intervals."
        ),
        "IQR": (
            "The bars span Q1 to Q3, the middle 50% of observed participant values. "
            "The IQR describes data spread and is not a confidence interval or standard error."
        ),
        "SD (descriptive spread)": (
            "The bars show one sample standard deviation on either side of the displayed "
            "median. SD describes observed spread; it is not a standard error or confidence interval."
        ),
    }
    if interval_kind not in interval_explanations:
        raise ValueError("Unsupported temporal interval kind")

    return {
        "baseline": baseline,
        "interval": interval_explanations[interval_kind],
        "limits": (
            "Missing measurements are omitted separately at each timepoint, so sample size "
            "can change over time. These trajectories are descriptive and do not provide a "
            "repeated-measures or longitudinal hypothesis test."
        ),
    }


def safe_temporal_filename(value, max_length=100):
    """Return a filesystem-friendly label for temporal plot downloads."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    cleaned = cleaned or "temporal_plot"
    if len(cleaned) > max_length:
        suffix = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:max_length - 9]}_{suffix}"
    return cleaned


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
