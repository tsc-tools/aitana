"""
Fault injection and detector scoring utilities for anomaly detection evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Offset magnitude presets as multipliers of each column's diff std.
# "easy": large offset — easy for a detector to catch.
# "hard": small offset — harder for a detector to catch.
_OFFSET_PRESETS = {
    "easy": (4.0, 6.0),
    "hard": (1.0, 2.0),
}


def inject_faults(
    df,
    columns=None,
    rate=0.06,
    offset_range="hard",
    mode="independent",
    coupled_groups=None,
    rng=None,
):
    """
    Inject synthetic faults into a DataFrame for anomaly detection evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series DataFrame. Index should be a DatetimeIndex.
    columns : list of str, optional
        Columns to target in independent mode. Defaults to all columns.
        Ignored in coupled mode (only columns in coupled_groups are affected).
    rate : float
        Per-timestep fault probability (independent) or per-timestep
        probability that a coupled group fires (coupled).
    offset_range : str or tuple of float
        Offset magnitude as a multiplier of each column's diff std.
        ``"hard"``: (1, 2) — small, harder to detect.
        ``"easy"``: (4, 6) — large, easier to detect.
        Tuple ``(k1, k2)``: custom multiplier range.
    mode : str
        ``"independent"``: per-column Bernoulli injection.
        ``"coupled"``: same fault timestamps injected across each group in
        ``coupled_groups``, mimicking a multi-stream precursor.
    coupled_groups : list of list of str, optional
        Groups of columns for coupled injection. Required when
        ``mode="coupled"``.
    rng : int, np.random.Generator, or None
        Random seed or generator for reproducibility.

    Returns
    -------
    corrupted_df : pd.DataFrame
        Copy of ``df`` with faults injected.
    mask_df : pd.DataFrame
        Boolean DataFrame of the same shape as ``df``; True where a fault
        was injected.
    """
    if isinstance(rng, (int, type(None))):
        rng = np.random.default_rng(rng)

    if isinstance(offset_range, str):
        k1, k2 = _OFFSET_PRESETS[offset_range]
    else:
        k1, k2 = offset_range

    corrupted = df.copy()
    mask_data = {col: np.zeros(len(df), dtype=bool) for col in df.columns}

    if mode == "independent":
        if columns is None:
            columns = list(df.columns)
        for col in columns:
            fault_locs = rng.random(len(df)) < rate
            n_faults = int(fault_locs.sum())
            if n_faults == 0:
                continue
            diff_std = float(df[col].diff().std())
            magnitudes = rng.uniform(k1, k2, size=n_faults) * diff_std
            signs = rng.choice([-1.0, 1.0], size=n_faults)
            col_arr = df[col].to_numpy(dtype=float).copy()
            col_arr[fault_locs] += magnitudes * signs
            corrupted[col] = col_arr
            mask_data[col] = fault_locs

    elif mode == "coupled":
        if coupled_groups is None:
            raise ValueError("coupled_groups must be provided when mode='coupled'")
        for group in coupled_groups:
            fault_locs = rng.random(len(df)) < rate
            n_faults = int(fault_locs.sum())
            if n_faults == 0:
                continue
            for col in group:
                if col not in df.columns:
                    continue
                diff_std = float(df[col].diff().std())
                magnitudes = rng.uniform(k1, k2, size=n_faults) * diff_std
                signs = rng.choice([-1.0, 1.0], size=n_faults)
                col_arr = df[col].to_numpy(dtype=float).copy()
                col_arr[fault_locs] += magnitudes * signs
                corrupted[col] = col_arr
                mask_data[col] = mask_data[col] | fault_locs

    else:
        raise ValueError(f"Unknown mode {mode!r}. Choose 'independent' or 'coupled'.")

    mask = pd.DataFrame(mask_data, index=df.index)
    return corrupted, mask


def score_detector(anomaly_index, mask, threshold):
    """
    Score a detector against the fault-injection ground-truth mask.

    Computes the full ROC curve and AUC using pointwise classification
    (not eruption-window apportionment).

    Parameters
    ----------
    anomaly_index : array-like of float
        Per-timestep anomaly scores in [0, 1].
    mask : pd.DataFrame or array-like of bool
        Ground-truth fault mask. If a DataFrame, any faulted column at a
        timestep counts as a positive label (logical OR across columns).
    threshold : float
        Operating threshold for computing binary alarm bits (TP/FP/TN/FN).

    Returns
    -------
    dict with keys:
        ``fpr``, ``tpr``, ``thresholds`` — full ROC curve arrays.
        ``auc`` — area under the ROC curve.
        ``tp``, ``fp``, ``tn``, ``fn`` — confusion matrix counts at ``threshold``.
    """
    if isinstance(mask, pd.DataFrame):
        y_true = mask.any(axis=1).to_numpy()
    else:
        y_true = np.asarray(mask, dtype=bool)

    scores = np.asarray(anomaly_index, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    y_pred = scores >= threshold
    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    tn = int(np.sum(~y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
