"""
Checks of SNet's preprocessing (common grid, anti-aliasing, gaps) and of the
gap masking in its scattering coefficients, on synthetic 100 Hz traces, and of
the embedding and the novelty index of coefficients from two synthetic spectral
regimes.
"""

import numpy as np
import pandas as pd
import xarray as xr
from obspy import Stream, Trace, UTCDateTime

from aitana.anomaly.snet import SNet, SNetNovelty, embed

START = UTCDateTime(2022, 3, 13)


def stream(duration, offset=0.0, signal=None, rng=0):
    """
    Three 100 Hz traces of noise, plus an optional signal, from START + offset.

    Parameters
    ----------
    duration : float
        Length in seconds; one closing sample is added, as GeoNet day files have.
    offset : float
        Start of the traces after START in seconds.
    signal : callable, optional
        Function of time in seconds added to every trace.
    rng : int
        Seed of the noise.

    Returns
    -------
    obspy.Stream
        Components Z, N, E.
    """
    rng = np.random.default_rng(rng)
    t = np.arange(int(duration * 100) + 1) / 100.0
    st = Stream()
    for cha in "ZNE":
        data = rng.normal(size=t.size)
        if signal is not None:
            data = signal(t)
        st += Trace(
            data,
            header=dict(network="NZ", station="FWVZ", location="10", channel=f"HH{cha}",
                        sampling_rate=100.0, starttime=START + offset),
        )
    return st


def test_preprocess_grid():
    """Traces starting a few ms late are resampled onto the requested grid."""
    st = SNet().preprocess(stream(3600, offset=0.003), START, START + 3600)
    assert [tr.stats.channel for tr in st] == ["HHZ", "HHN", "HHE"]
    for tr in st:
        assert tr.stats.starttime == START
        assert tr.stats.sampling_rate == 25.0
        assert tr.stats.npts == 3600 * 25
        # the first grid sample precedes the trace
        assert np.isnan(tr.data[0])
        assert np.isfinite(tr.data[1:]).all()


def test_preprocess_antialias():
    """A 14.5 Hz signal must not alias onto the 10.5 Hz scattering filter."""

    def amplitude(freq):
        st = stream(600, signal=lambda t: np.sin(2 * np.pi * freq * t))
        data = SNet().preprocess(st, START, START + 600)[0].data[100:-100]
        return np.sqrt(2) * data.std()

    assert amplitude(5.0) > 0.95
    assert amplitude(14.5) < 10 ** (-25 / 20)


def test_gaps_are_masked():
    """Windows missing more than 10% of a component are NaN, others are not."""
    st = stream(3600)
    st[0].data[300 * 100 : 306 * 100] = np.nan  # 6 s in Z
    st[1].data[2100 * 100 : 2400 * 100] = np.nan  # 35-40 min in N
    snet = SNet()
    feature = snet.compute(snet.preprocess(st, START, START + 3600))

    starts = [START.datetime + np.timedelta64(m, "m") for m in range(0, 50, 10)]
    np.testing.assert_array_equal(feature.datetime.values, np.array(starts, dtype="datetime64[ns]"))
    # windows starting at 20 and 30 min hold the 5-min gap: 25% missing
    expected = [False, False, True, True, False]
    for name in ("order1", "order2"):
        gappy = feature[name].isnull()
        dims = [d for d in gappy.dims if d != "datetime"]
        np.testing.assert_array_equal(gappy.all(dims).values, expected)
        np.testing.assert_array_equal(gappy.any(dims).values, expected)


def coefficients(regimes, rng=0):
    """
    Scattering coefficients of windows from two spectral regimes.

    Parameters
    ----------
    regimes : array-like of int
        Regime of each window: 0 has its energy around 1 Hz, 1 around 5 Hz.
    rng : int
        Seed of the noise.

    Returns
    -------
    xr.Dataset
        ``order1`` and ``order2`` coefficients shaped like ``SNet.compute``
        output, 10 min apart from START.
    """
    rng = np.random.default_rng(rng)
    n = len(regimes)
    freq = np.geomspace(10.5, 0.195, 24)
    peak = np.where(np.asarray(regimes) == 0, 1.0, 5.0)[:, None]
    spectrum = -np.log(freq / peak) ** 2
    coords = {
        "datetime": pd.date_range(START.datetime, periods=n, freq="10min"),
        "channel": np.arange(3),
        "frequency": freq,
    }
    order1 = np.exp(spectrum[:, None, :] + 0.3 * rng.normal(size=(n, 3, 24)))
    coords2 = dict(coords, **{"order2 frequency": np.geomspace(6.25, 0.195, 6)})
    order2 = np.exp(spectrum[:, None, :, None] + 0.3 * rng.normal(size=(n, 3, 24, 6)))
    return xr.Dataset({
        "order1": xr.DataArray(order1, dims=list(coords), coords=coords),
        "order2": xr.DataArray(order2, dims=list(coords2), coords=coords2),
    })


def test_embedding():
    """
    A rare regime forms a group of its own, as it is part of the fit; windows are
    labelled 10 min after each window's label, gaps are NaN.
    """
    regimes = np.repeat([0, 1], [360, 40])
    snet = coefficients(regimes)
    # a gap, which the fit must leave out
    snet["order1"][3] = np.nan
    snet["order2"][3] = np.nan
    embedding = embed(snet)
    assert embedding.dims == ("datetime", "component")
    np.testing.assert_array_equal(
        embedding.datetime.values, snet.indexes["datetime"] + pd.Timedelta("10min")
    )
    assert embedding[3].isnull().all()

    points = np.delete(embedding.values, 3, axis=0)
    assert np.isfinite(points).all()
    regimes = np.delete(regimes, 3)
    distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    assert (regimes[distance.argmin(axis=1)] == regimes)[regimes == 1].mean() >= 0.95


def test_novelty_contract():
    """Scores are labelled 10 min after each window's label, in [0, 1], NaN for gaps."""
    detector = SNetNovelty().fit(coefficients(np.zeros(300, int)))
    new = coefficients(np.zeros(50, int), rng=1)
    new["order1"][5] = np.nan
    new["order2"][5] = np.nan
    index = detector.compute_anomaly_index(new)
    np.testing.assert_array_equal(
        index.index, new.indexes["datetime"] + pd.Timedelta("10min")
    )
    assert np.isnan(index.iloc[5])
    assert index.drop(index.index[5]).between(0, 1).all()
    assert detector.distance_.index.equals(index.index)
    assert np.isnan(detector.distance_.iloc[5])


def test_novelty_flags_new_regime():
    """Windows of a regime missing from the reference score 1, familiar ones rarely high."""
    detector = SNetNovelty().fit(coefficients(np.zeros(300, int)))
    index = detector.compute_anomaly_index(coefficients(np.repeat([0, 1], 50), rng=1))
    assert (index.iloc[50:] == 1.0).all()
    assert (index.iloc[:50] >= 0.99).mean() < 0.1


def test_novelty_exclusion():
    """
    Overlapping reference windows are near-duplicates; ignoring them keeps the
    reference scores comparable to those of new windows.
    """
    reference = coefficients(np.zeros(300, int))
    # every second window nearly repeats the one 10 min before it
    for name in ("order1", "order2"):
        reference[name][1::2] = reference[name][::2].values * 1.001
    new = coefficients(np.zeros(100, int), rng=1)

    index = SNetNovelty().fit(reference).compute_anomaly_index(new)
    assert (index >= 0.99).mean() < 0.1
    # without the exclusion, every new window looks novel
    index = SNetNovelty(exclusion="0min").fit(reference).compute_anomaly_index(new)
    assert (index == 1.0).all()
    # reference windows are not scored against themselves
    index = SNetNovelty().fit(reference).compute_anomaly_index(reference)
    assert 0.4 < index.median() < 0.6
