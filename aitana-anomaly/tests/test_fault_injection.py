"""
Tests for Phase 0: fault injection harness and detector scoring.

Verify:
- Independent mode: ground-truth mask bookkeeping is correct, non-faulted
  entries are unchanged, faulted entries differ.
- Coupled mode: faults land on the same timestamps across all columns in
  a group; columns outside any group are untouched.
- inject_change_points: mask marks only the onset timestep, but the shift
  itself persists after onset; coupled onsets are shared across a group.
- simulate_regime_series: segment shapes and change-point locations match
  the requested state sequence, and repeated regime labels share a mean.
- score_detector: a trivial threshold-on-raw-value detector achieves high
  AUC against easy faults, confirming the harness end-to-end.
- normalize_score: ECDF percentile ranks are in [0, 1] and monotone.
"""

import numpy as np
import pandas as pd
import pytest

from aitana.anomaly.anomaly_base import AnomalyDetectionBaseClass
from aitana.anomaly.evaluation import (
    inject_change_points,
    inject_faults,
    score_detector,
    simulate_regime_series,
)


@pytest.fixture
def toy_df():
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, n),
            "b": rng.normal(5.0, 1.0, n),
            "c": rng.normal(-3.0, 2.0, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )


# ---------------------------------------------------------------------------
# Independent mode
# ---------------------------------------------------------------------------


def test_independent_mask_shape(toy_df):
    corrupted, mask = inject_faults(toy_df, rate=0.1, mode="independent", rng=1)
    assert mask.shape == toy_df.shape
    assert mask.dtypes.eq(bool).all().all()


def test_independent_unfaulted_entries_unchanged(toy_df):
    corrupted, mask = inject_faults(toy_df, rate=0.1, mode="independent", rng=2)
    # Entries where mask is False must be identical to the original.
    unchanged = (corrupted == toy_df) | mask
    assert unchanged.all().all()


def test_independent_faulted_entries_differ(toy_df):
    corrupted, mask = inject_faults(toy_df, rate=0.1, mode="independent", rng=3)
    assert mask.any().any(), "expected at least one fault to be injected"
    # Every True entry in the mask must differ from the original value.
    for col in toy_df.columns:
        col_mask = mask[col]
        if col_mask.any():
            assert (corrupted.loc[col_mask, col] != toy_df.loc[col_mask, col]).all()


def test_independent_column_subset(toy_df):
    # Only inject into columns "a" and "b"; "c" should be untouched.
    corrupted, mask = inject_faults(
        toy_df, columns=["a", "b"], rate=0.2, mode="independent", rng=4
    )
    assert not mask["c"].any()
    pd.testing.assert_series_equal(corrupted["c"], toy_df["c"])


def test_independent_easy_vs_hard_offset(toy_df):
    # Easy faults should produce larger deviations than hard faults.
    _, mask_easy = inject_faults(toy_df, rate=0.2, offset_range="easy", rng=5)
    _, mask_hard = inject_faults(toy_df, rate=0.2, offset_range="hard", rng=5)
    corrupted_easy, _ = inject_faults(toy_df, rate=0.2, offset_range="easy", rng=5)
    corrupted_hard, _ = inject_faults(toy_df, rate=0.2, offset_range="hard", rng=5)
    easy_dev = (corrupted_easy - toy_df).abs().mean().mean()
    hard_dev = (corrupted_hard - toy_df).abs().mean().mean()
    assert easy_dev > hard_dev


def test_independent_custom_offset_range(toy_df):
    corrupted, mask = inject_faults(
        toy_df, rate=0.1, offset_range=(10.0, 12.0), mode="independent", rng=6
    )
    assert mask.any().any()


# ---------------------------------------------------------------------------
# Coupled mode
# ---------------------------------------------------------------------------


def test_coupled_same_timestamps(toy_df):
    groups = [["a", "b"]]
    corrupted, mask = inject_faults(
        toy_df, mode="coupled", coupled_groups=groups, rate=0.1, rng=7
    )
    # "a" and "b" must be faulted at exactly the same timesteps.
    pd.testing.assert_series_equal(
        mask["a"].reset_index(drop=True),
        mask["b"].reset_index(drop=True),
        check_names=False,
    )


def test_coupled_non_group_column_untouched(toy_df):
    groups = [["a", "b"]]
    corrupted, mask = inject_faults(
        toy_df, mode="coupled", coupled_groups=groups, rate=0.1, rng=8
    )
    assert not mask["c"].any()
    pd.testing.assert_series_equal(corrupted["c"], toy_df["c"])


def test_coupled_multiple_groups(toy_df):
    groups = [["a", "b"], ["c"]]
    corrupted, mask = inject_faults(
        toy_df, mode="coupled", coupled_groups=groups, rate=0.1, rng=9
    )
    # Each group's internal columns share fault timestamps, but the two
    # groups are independent draws so their fault sets need not match.
    pd.testing.assert_series_equal(
        mask["a"].reset_index(drop=True),
        mask["b"].reset_index(drop=True),
        check_names=False,
    )


def test_coupled_overlapping_groups_keep_all_faults(toy_df):
    # "a" is in both groups, so it must carry both groups' faults, not just
    # the last group's.
    groups = [["a", "b"], ["a", "c"]]
    corrupted, mask = inject_faults(
        toy_df, mode="coupled", coupled_groups=groups, rate=0.1, rng=12
    )
    pd.testing.assert_series_equal(
        corrupted["a"] != toy_df["a"], mask["a"], check_names=False
    )


def test_coupled_requires_coupled_groups(toy_df):
    with pytest.raises(ValueError, match="coupled_groups"):
        inject_faults(toy_df, mode="coupled", coupled_groups=None, rng=10)


def test_unknown_mode_raises(toy_df):
    with pytest.raises(ValueError, match="Unknown mode"):
        inject_faults(toy_df, mode="bogus", rng=11)


# ---------------------------------------------------------------------------
# inject_change_points
# ---------------------------------------------------------------------------


def test_change_points_mask_is_sparse(toy_df):
    # Unlike inject_faults, the mask should mark only onsets, not the whole
    # shifted segment that follows.
    corrupted, change_points = inject_change_points(
        toy_df, n_changes=3, mode="independent", rng=1
    )
    assert change_points.shape == toy_df.shape
    for col in toy_df.columns:
        assert change_points[col].sum() == 3


def test_change_points_shift_persists_after_onset(toy_df):
    corrupted, change_points = inject_change_points(
        toy_df, columns=["a"], n_changes=1, offset_range="easy", mode="independent", rng=2
    )
    onset = change_points.index[change_points["a"]][0]
    onset_pos = toy_df.index.get_loc(onset)
    # The shift should still be present several steps after the onset, not
    # just at the onset timestep itself.
    later_pos = min(onset_pos + 5, len(toy_df) - 1)
    later = toy_df.index[later_pos]
    assert corrupted.loc[later, "a"] != toy_df.loc[later, "a"]


def test_change_points_before_onset_unchanged(toy_df):
    corrupted, change_points = inject_change_points(
        toy_df, columns=["a"], n_changes=1, mode="independent", rng=3
    )
    onset = change_points.index[change_points["a"]][0]
    onset_pos = toy_df.index.get_loc(onset)
    if onset_pos > 0:
        before = toy_df.index[onset_pos - 1]
        assert corrupted.loc[before, "a"] == toy_df.loc[before, "a"]


def test_change_points_coupled_same_onsets(toy_df):
    groups = [["a", "b"]]
    corrupted, change_points = inject_change_points(
        toy_df, mode="coupled", coupled_groups=groups, n_changes=2, rng=4
    )
    pd.testing.assert_series_equal(
        change_points["a"].reset_index(drop=True),
        change_points["b"].reset_index(drop=True),
        check_names=False,
    )
    assert not change_points["c"].any()


def test_change_points_coupled_overlapping_groups_keep_all_shifts(toy_df):
    # "a" is in both groups, so its level must step at every onset the mask
    # records, not just the last group's.
    groups = [["a", "b"], ["a", "c"]]
    corrupted, change_points = inject_change_points(
        toy_df, mode="coupled", coupled_groups=groups, n_changes=2, rng=7
    )
    level = (corrupted["a"] - toy_df["a"]).to_numpy()
    steps = np.flatnonzero(np.abs(np.diff(level)) > 1e-9) + 1
    np.testing.assert_array_equal(steps, np.flatnonzero(change_points["a"]))


def test_change_points_coupled_requires_coupled_groups(toy_df):
    with pytest.raises(ValueError, match="coupled_groups"):
        inject_change_points(toy_df, mode="coupled", coupled_groups=None, rng=5)


def test_change_points_unknown_mode_raises(toy_df):
    with pytest.raises(ValueError, match="Unknown mode"):
        inject_change_points(toy_df, mode="bogus", rng=6)


# ---------------------------------------------------------------------------
# simulate_regime_series
# ---------------------------------------------------------------------------


def test_simulate_regime_series_shape():
    series, change_points, labels = simulate_regime_series(
        ["A", "B", "A", "C"], segment_length=50, n_features=2, rng=0
    )
    assert series.shape == (200, 2)
    assert len(labels) == 200
    assert len(change_points) == 3


def test_simulate_regime_series_change_point_locations():
    series, change_points, labels = simulate_regime_series(
        ["A", "B", "A", "C"], segment_length=[10, 20, 30, 40], rng=1
    )
    np.testing.assert_array_equal(change_points, [10, 30, 60])


def test_simulate_regime_series_repeated_state_same_mean():
    # Both "A" segments must be drawn from the same mean so a state
    # detector could recover a shared label for them.
    series, change_points, labels = simulate_regime_series(
        ["A", "B", "A"],
        segment_length=500,
        n_features=1,
        state_noise_std={"A": 0.1, "B": 0.1},
        rng=2,
    )
    mean_a1 = series.loc[labels == "A"].iloc[:500].mean().item()
    mean_a2 = series.loc[labels == "A"].iloc[500:].mean().item()
    assert abs(mean_a1 - mean_a2) < 0.1


def test_simulate_regime_series_segment_length_mismatch_raises():
    with pytest.raises(ValueError):
        simulate_regime_series(["A", "B"], segment_length=[10], rng=3)


# ---------------------------------------------------------------------------
# score_detector: end-to-end harness check
# ---------------------------------------------------------------------------


def test_score_detector_easy_faults_high_auc():
    """
    A trivial detector (absolute deviation from the mean) should achieve
    high AUC against easy faults, confirming the harness bookkeeping is
    correct end-to-end.
    """
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    n = 1000
    df = pd.DataFrame(
        {"x": rng.normal(0.0, 1.0, n)},
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )

    corrupted, mask = inject_faults(
        df, rate=0.05, offset_range="easy", mode="independent", rng=rng_seed
    )

    # Trivial detector: normalised absolute deviation from the clean mean.
    raw = (corrupted["x"] - df["x"].mean()).abs()
    anomaly_index = (raw / raw.max()).to_numpy()

    result = score_detector(anomaly_index, mask, threshold=0.5)

    assert 0.0 <= result["auc"] <= 1.0
    assert result["auc"] > 0.8, f"Expected AUC > 0.8, got {result['auc']:.3f}"
    assert result["tp"] + result["fn"] == int(mask.any(axis=1).sum())
    assert result["fp"] + result["tn"] == int((~mask.any(axis=1)).sum())


def test_score_detector_array_mask():
    """score_detector accepts a plain boolean array as the mask."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.random(n) < 0.1
    scores = np.where(y_true, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n))
    result = score_detector(scores, y_true, threshold=0.5)
    assert result["auc"] > 0.9


def test_score_detector_confusion_matrix_sums(toy_df):
    corrupted, mask = inject_faults(
        toy_df, rate=0.05, offset_range="easy", mode="independent", rng=0
    )
    raw = (corrupted - toy_df).abs().mean(axis=1)
    anomaly_index = (raw / raw.max()).to_numpy()
    result = score_detector(anomaly_index, mask, threshold=0.5)
    n = len(toy_df)
    assert result["tp"] + result["fp"] + result["tn"] + result["fn"] == n


# ---------------------------------------------------------------------------
# normalize_score
# ---------------------------------------------------------------------------


def test_normalize_score_range():
    ref = np.linspace(0, 10, 100)
    raw = np.array([-1.0, 0.0, 5.0, 10.0, 11.0])
    scores = AnomalyDetectionBaseClass.normalize_score(raw, ref)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_normalize_score_monotone():
    ref = np.random.default_rng(0).normal(0, 1, 500)
    raw = np.linspace(-3, 3, 50)
    scores = AnomalyDetectionBaseClass.normalize_score(raw, ref)
    assert (np.diff(scores) >= 0).all()


def test_normalize_score_reference_point_above_all():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # A score above all reference values should return 1.0.
    scores = AnomalyDetectionBaseClass.normalize_score([6.0], ref)
    assert scores[0] == 1.0


def test_normalize_score_reference_point_below_all():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # A score below all reference values should return 0.0.
    scores = AnomalyDetectionBaseClass.normalize_score([0.0], ref)
    assert scores[0] == 0.0
