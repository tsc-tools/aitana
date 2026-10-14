"""
Tests for the matrix-profile anomaly index.

The synthetic background repeats a daily template plus small noise. A
z-normalized matrix profile can only call a subsequence unusual if normal
subsequences have close matches in the history; on pure noise every
subsequence is roughly equally far from all others.
"""

import numpy as np
import pandas as pd
import pytest
from aitana.anomaly.evaluation import (
    inject_faults,
    quiet_series,
    score_detector,
    simulate_regime_series,
)
from aitana.anomaly.matrix_profile import MatrixProfile

M = 36  # 6 h of 10-min samples
DAY = 144
SCORE_START = 28 * DAY + M
N = SCORE_START + DAY
THRESHOLD = 0.999


def windows_covering(mask, m=M):
    """
    Mark every subsequence (labelled by its last sample) that contains a True.

    Parameters
    ----------
    mask : np.ndarray of bool
        Per-sample ground truth.
    m : int
        Subsequence length.

    Returns
    -------
    np.ndarray of bool
    """
    return np.convolve(mask.astype(int), np.ones(m, dtype=int))[: len(mask)] > 0


@pytest.fixture(scope="module")
def detector():
    return MatrixProfile(window=M)


def test_contract(detector):
    x = quiet_series(N, rng=0)
    x.index = pd.date_range("2022-01-01", periods=N, freq="10min")
    x.iloc[SCORE_START + 50 : SCORE_START + 55] = np.nan
    score_start = x.index[SCORE_START]

    aindex = detector.compute_anomaly_index(x, score_start)

    assert aindex.index.equals(x.index)
    assert aindex[:score_start].iloc[:-1].isna().all()
    scored = aindex[score_start:]
    assert scored.notna().any()
    assert scored.min() >= 0 and scored.max() <= 1
    # Every subsequence touching the gap is NaN, not a false alarm.
    gap = np.zeros(N, dtype=bool)
    gap[SCORE_START + 50 : SCORE_START + 55] = True
    assert aindex[windows_covering(gap)].isna().all()


def test_change_point(detector):
    """A level shift is flagged while windows span it, not on the new plateau."""
    step, change_points, _ = simulate_regime_series(
        ["A", "B"],
        segment_length=[SCORE_START + 48, N - SCORE_START - 48],
        state_means={"A": [0.0], "B": [1.0]},
        state_noise_std={"A": 0.0, "B": 0.0},
    )
    x = quiet_series(N, rng=0) + step["f0"]
    cp = change_points[0]

    aindex = detector.compute_anomaly_index(x, SCORE_START).to_numpy()

    end = np.arange(N)
    spanning = (end >= cp) & (end <= cp + M - 2)
    plateau = end >= cp + M - 1
    scored = end >= SCORE_START
    result = score_detector(aindex[scored], spanning[scored], THRESHOLD)
    assert result["auc"] >= 0.95
    assert aindex[spanning].max() >= THRESHOLD
    # Amplitude is left to the existing level-threshold alert.
    assert aindex[plateau].max() < THRESHOLD


def test_glitches(detector):
    x = quiet_series(N, rng=0).to_frame("rsam")
    corrupted, mask = inject_faults(
        x.iloc[SCORE_START:], rate=0.01, offset_range=(15, 25), rng=0
    )
    x.iloc[SCORE_START:] = corrupted
    faults = np.zeros(N, dtype=bool)
    faults[SCORE_START:] = mask["rsam"].to_numpy()
    assert faults.any()

    aindex = detector.compute_anomaly_index(x["rsam"], SCORE_START).to_numpy()

    scored = np.arange(N) >= SCORE_START
    result = score_detector(aindex[scored], windows_covering(faults)[scored], THRESHOLD)
    assert result["auc"] >= 0.95
    assert result["fp"] == 0


def test_repeated_burst(detector):
    """
    The first occurrence of a new shape is flagged even though it recurs.

    In a plain self-join both occurrences would match each other and score
    low; the left matrix profile only compares against earlier data.
    """
    x = quiet_series(N, rng=0)
    burst = 30 * np.exp(-0.5 * ((np.arange(18) - 9) / 3) ** 2)
    first, second = SCORE_START + 30, SCORE_START + 90
    x.iloc[first : first + 18] += burst
    x.iloc[second : second + 18] += burst

    aindex = detector.compute_anomaly_index(x, SCORE_START)

    # subsequences that contain a whole burst
    assert aindex.iloc[first + 17 : first + M].max() >= THRESHOLD
    assert aindex.iloc[second + 17 : second + M].max() < THRESHOLD
