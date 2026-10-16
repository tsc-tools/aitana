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


def _inject_offsets(
    df, sample_locs, persistent, columns, offset_range, mode, coupled_groups, rng
):
    """
    Add random offsets at sampled locations, per column or per coupled group.

    Shared engine of ``inject_faults`` and ``inject_change_points``.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series DataFrame.
    sample_locs : callable
        ``sample_locs(rng)`` returns the integer positions to offset. Called
        once per column (independent) or once per group (coupled).
    persistent : bool
        If True, each offset persists from its location onward and successive
        offsets accumulate (a step); otherwise it affects that timestep only.
    columns, offset_range, mode, coupled_groups, rng
        As in ``inject_faults``.

    Returns
    -------
    corrupted_df : pd.DataFrame
        Copy of ``df`` with the offsets added.
    mask_df : pd.DataFrame
        Boolean DataFrame of the same shape as ``df``; True at every sampled
        location.
    """
    if isinstance(rng, (int, type(None))):
        rng = np.random.default_rng(rng)

    if isinstance(offset_range, str):
        k1, k2 = _OFFSET_PRESETS[offset_range]
    else:
        k1, k2 = offset_range

    if mode == "independent":
        if columns is None:
            columns = list(df.columns)
        groups = [[col] for col in columns]
    elif mode == "coupled":
        if coupled_groups is None:
            raise ValueError("coupled_groups must be provided when mode='coupled'")
        groups = [
            [col for col in group if col in df.columns] for group in coupled_groups
        ]
    else:
        raise ValueError(f"Unknown mode {mode!r}. Choose 'independent' or 'coupled'.")

    corrupted = df.copy()
    mask_data = {col: np.zeros(len(df), dtype=bool) for col in df.columns}
    for group in groups:
        locs = sample_locs(rng)
        if len(locs) == 0:
            continue
        for col in group:
            diff_std = float(df[col].diff().std())
            magnitudes = rng.uniform(k1, k2, size=len(locs)) * diff_std
            signs = rng.choice([-1.0, 1.0], size=len(locs))
            offset = np.zeros(len(df))
            offset[locs] = magnitudes * signs
            if persistent:
                offset = np.cumsum(offset)
            # Add to corrupted, not df, so a column in several groups keeps
            # every group's offsets.
            corrupted[col] = corrupted[col].to_numpy(dtype=float) + offset
            mask_data[col][locs] = True

    mask = pd.DataFrame(mask_data, index=df.index)
    return corrupted, mask


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
    return _inject_offsets(
        df,
        lambda rng: np.flatnonzero(rng.random(len(df)) < rate),
        persistent=False,
        columns=columns,
        offset_range=offset_range,
        mode=mode,
        coupled_groups=coupled_groups,
        rng=rng,
    )


def inject_change_points(
    df,
    columns=None,
    n_changes=3,
    offset_range="hard",
    mode="independent",
    coupled_groups=None,
    rng=None,
):
    """
    Inject persistent regime shifts (change points) into a DataFrame.

    Unlike ``inject_faults``, which perturbs isolated timesteps (sensor
    glitches), each injected change point offsets the signal from its onset
    onward until the next change point (or end of series) — a synthetic
    analogue of a genuine regime change, for evaluating change-point/state
    detectors (e.g. ClaSP) rather than pointwise anomaly detectors.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series DataFrame. Index should be a DatetimeIndex.
    columns : list of str, optional
        Columns to target in independent mode. Defaults to all columns.
        Ignored in coupled mode (only columns in coupled_groups are affected).
    n_changes : int
        Number of change points to inject per column (independent) or per
        group (coupled).
    offset_range : str or tuple of float
        Per-change-point offset magnitude as a multiplier of each column's
        diff std, accumulated at each successive change point. Same presets
        as ``inject_faults``.
    mode : str
        ``"independent"``: per-column change points at independent onsets.
        ``"coupled"``: same onsets injected across each group in
        ``coupled_groups``.
    coupled_groups : list of list of str, optional
        Groups of columns for coupled injection. Required when
        ``mode="coupled"``.
    rng : int, np.random.Generator, or None
        Random seed or generator for reproducibility.

    Returns
    -------
    corrupted_df : pd.DataFrame
        Copy of ``df`` with persistent regime shifts injected.
    change_points : pd.DataFrame
        Boolean DataFrame of the same shape as ``df``; True at the onset
        timestep of a change point (not across the whole shifted segment).
    """
    def _sample_onsets(rng):
        n = min(n_changes, len(df) - 1)
        if n <= 0:
            return np.array([], dtype=int)
        return np.sort(rng.choice(np.arange(1, len(df)), size=n, replace=False))

    return _inject_offsets(
        df,
        _sample_onsets,
        persistent=True,
        columns=columns,
        offset_range=offset_range,
        mode=mode,
        coupled_groups=coupled_groups,
        rng=rng,
    )


def simulate_regime_series(
    state_sequence,
    segment_length,
    n_features=1,
    state_noise_std=None,
    state_means=None,
    rng=None,
):
    """
    Generate a synthetic multivariate time series stitched from discrete regimes.

    Each entry in ``state_sequence`` selects a regime label (e.g.
    ``["A", "B", "A", "C"]``); repeated labels reuse the same mean, so
    recurring regimes are known ground truth for state-detection algorithms
    (e.g. CLaP), and segment boundaries are the known ground-truth change
    points for change-point detectors (e.g. ClaSP).

    Parameters
    ----------
    state_sequence : list of str
        Sequence of regime labels, e.g. ``["A", "B", "A", "C"]``.
    segment_length : int or list of int
        Number of timesteps per regime. A single int applies uniformly to
        every segment; a list must have the same length as
        ``state_sequence``.
    n_features : int
        Number of feature columns to generate.
    state_noise_std : dict, optional
        Mapping from regime label to the standard deviation of that state's
        additive Gaussian noise. If not given, every state defaults to 1.0.
    state_means : dict, optional
        Mapping from regime label to a length-``n_features`` array of means.
        If not given, one is drawn per unique label from ``Normal(0, 5)``.
    rng : int, np.random.Generator, or None
        Random seed or generator for reproducibility.

    Returns
    -------
    series : pd.DataFrame
        Synthetic ``(n_timesteps, n_features)`` series with a RangeIndex.
    change_points : np.ndarray of int
        Index of the first timestep of each new segment (excludes 0).
    labels : pd.Series
        Per-timestep ground-truth regime label, aligned to ``series``.
    """
    if isinstance(rng, (int, type(None))):
        rng = np.random.default_rng(rng)

    if isinstance(segment_length, int):
        segment_lengths = [segment_length] * len(state_sequence)
    else:
        segment_lengths = list(segment_length)
        if len(segment_lengths) != len(state_sequence):
            raise ValueError("segment_length list must match len(state_sequence)")

    unique_states = sorted(set(state_sequence))
    if state_means is None:
        state_means = {
            state: rng.normal(0.0, 5.0, n_features) for state in unique_states
        }
    if state_noise_std is None:
        state_noise_std = {state: 1.0 for state in unique_states}

    segments = []
    labels = []
    change_points = []
    cursor = 0
    for i, (state, length) in enumerate(zip(state_sequence, segment_lengths)):
        mean = np.asarray(state_means[state], dtype=float)
        std = state_noise_std[state]
        segments.append(mean + rng.normal(0.0, std, size=(length, n_features)))
        labels.extend([state] * length)
        if i > 0:
            change_points.append(cursor)
        cursor += length

    series = pd.DataFrame(
        np.vstack(segments), columns=[f"f{i}" for i in range(n_features)]
    )
    labels = pd.Series(labels, name="state")
    return series, np.array(change_points, dtype=int), labels


def quiet_series(n, period=144, noise=0.05, rng=None):
    """
    Generate a quiet background: a repeating template plus small noise.

    The template is a sine of length ``period`` plus a half-amplitude third
    harmonic, so normal behaviour recurs and every normal subsequence has
    close matches in the history. Similarity-based detectors (e.g. a
    z-normalized matrix profile) need this; on pure noise every subsequence
    is roughly equally far from all others.

    Parameters
    ----------
    n : int
        Number of samples.
    period : int
        Samples per template cycle; the default is one day of 10-min samples.
    noise : float
        Standard deviation of the additive Gaussian noise.
    rng : int, np.random.Generator, or None
        Random seed or generator for reproducibility.

    Returns
    -------
    pd.Series
        Synthetic series on a RangeIndex.
    """
    if isinstance(rng, (int, type(None))):
        rng = np.random.default_rng(rng)

    phase = 2 * np.pi * np.arange(n) / period
    return pd.Series(
        np.sin(phase) + 0.5 * np.sin(3 * phase) + noise * rng.normal(size=n)
    )


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
