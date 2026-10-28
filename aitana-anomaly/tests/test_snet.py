"""
Checks of SNet's preprocessing (common grid, anti-aliasing, gaps) and of the
gap masking in its scattering coefficients, on synthetic 100 Hz traces.
"""

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from aitana.anomaly.snet import SNet

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
