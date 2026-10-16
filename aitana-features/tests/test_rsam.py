import inspect
import os

import numpy as np
import pytest
from aitana.testing import synthetic_trace
from aitana.features.rsam import RSAM
from obspy import Trace, UTCDateTime


def original_rsam(trace):
    """
    RSAM implementation by Steve Sherburn
    """
    f1 = 1
    f2 = 4
    data = np.zeros(145)
    t = trace.stats.starttime
    nans = np.isnan(trace.data)
    indices = np.arange(trace.stats.npts)
    trace.data[nans] = np.interp(indices[nans], indices[~nans], trace.data[~nans])
    index = 0
    # loop through data in 600sec (10 min) blocks
    while t < trace.stats.endtime:
        tr_10m = trace.slice(t, t + 600)

        duration = tr_10m.stats.npts * tr_10m.stats.delta
        if duration >= 500:
            if duration < 600:
                tr_10m = trace.slice(trace.stats.endtime - 600, trace.stats.endtime)
            tr_10m.detrend(type="constant")
            tr_10m.filter(
                "bandpass", freqmin=f1, freqmax=f2, corners=4, zerophase=False
            )
            absolute = np.absolute(tr_10m.data)  # absolute value
            tr_10m.data = absolute  # assign back to trace
            mean = tr_10m.data.mean()
            mean = mean / 1e-9  # convert to nanometres so dealing with whole numbers
            data[index] = mean
            index += 1
        t += 600
    data = np.resize(data, index)
    return data


@pytest.fixture(scope="session")
def rsam_setup():
    amplitudes = [0.1, 1.0]
    offsets = [1000, 2000]
    frequencies = [0.1, 3.0]
    phases = [0.0, np.pi * 0.25]
    starttime = UTCDateTime(1978, 7, 18, 8, 8, 24)
    tr1 = synthetic_trace(
        sinusoid=False,
        addchirp=False,
        starttime=starttime,
        offsets=offsets,
        amplitudes=amplitudes,
        phases=phases,
        frequencies=frequencies,
    )
    tr1.data *= 1e-9
    tr2 = synthetic_trace(
        sinusoid=False,
        addchirp=False,
        starttime=starttime,
        offsets=offsets,
        amplitudes=amplitudes,
        phases=phases,
        frequencies=frequencies,
        gaps=True,
    )
    tr2.data *= 1e-9
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
        "data",
    )

    return tr1, tr2, data_dir


def test_with_gaps(rsam_setup):
    tr1, tr2, data_dir = rsam_setup
    r = RSAM()
    xdf = r.compute(tr1)
    max_err = np.abs((xdf["rsam"].values - original_rsam(tr1.copy()))).max()
    assert max_err < 0.01

    xdf1 = r.compute(tr2)
    max_err = np.abs((xdf1["rsam"].values - original_rsam(tr2.copy()))).max()
    assert max_err < 0.01
    assert UTCDateTime(str(xdf1.datetime.values[0])) == UTCDateTime(
        xdf1.attrs["starttime"]
    )
    assert (
        UTCDateTime(str(xdf1.datetime.values[-1]))
        == UTCDateTime(xdf1.attrs["starttime"]) + 5.0 * 600.0
    )


@pytest.fixture(scope="session")
def decreasing_outofband_signal():
    interval = 600.0
    inband_freq, outofband_freq = 3.5, 8.0
    outofband_levels = [
        1.0,
        0.5,
        0.1,
    ]  # target cumulative out-of-band amplitude per window
    outofband_deltas = [outofband_levels[0]] + list(np.diff(outofband_levels))
    n_windows = len(outofband_levels)

    return synthetic_trace(
        nsec=interval * n_windows,
        frequencies=[inband_freq] + [outofband_freq] * n_windows,
        amplitudes=[1.0] + outofband_deltas,
        phases=[0.0] * (n_windows + 1),
        offsets=[0.0] + [i * interval for i in range(n_windows)],
        starttime=UTCDateTime(2020, 1, 1),
        sinusoid=False,
        addchirp=False,
        noise=False,
    )


def test_rsam_energy_prop_increases_as_outofband_content_decreases(
    decreasing_outofband_signal,
):
    r = RSAM()
    xdf = r.compute(decreasing_outofband_signal)
    energy_prop = xdf["rsam_energy_prob"].values

    assert len(energy_prop) == 3
    assert not np.isnan(energy_prop).any()
    assert np.all(np.diff(energy_prop) > 0)


@pytest.fixture
def noise_trace():
    rng = np.random.default_rng(0)
    return Trace(
        rng.normal(size=6 * 60000),
        header={"sampling_rate": 100.0, "starttime": UTCDateTime(2022, 1, 1)},
    )


def test_max_gap_fraction_blanks_gappy_windows(noise_trace):
    # 5% of window 1 and 30% of window 3 are missing; only window 3 exceeds 10%.
    tr = noise_trace.copy()
    tr.data[60000:63000] = np.nan
    tr.data[180000:198000] = np.nan
    default = RSAM().compute(tr)
    blanked = RSAM(max_gap_fraction=0.1).compute(tr)
    for name in ("rsam", "rsam_energy_prob"):
        values = blanked[name].values
        np.testing.assert_array_equal(
            np.isnan(values), [False, False, False, True, False, False]
        )
        keep = ~np.isnan(values)
        np.testing.assert_array_equal(values[keep], default[name].values[keep])


def test_trailing_padded_window_dropped(noise_trace):
    # A trace that includes the closing sample must not yield a seventh window
    # made of one sample and zero padding.
    tr = noise_trace.copy()
    tr.data = np.append(tr.data, 0.0)
    xdf = RSAM().compute(tr)
    assert len(xdf.datetime) == 6
    assert UTCDateTime(str(xdf.datetime.values[-1])) == tr.stats.starttime + 5 * 600.0


def test_trailing_window_mostly_data_kept(noise_trace):
    # As in the reference implementation, a last window with at least 500 of
    # its 600 s covered by data is kept.
    tr = noise_trace.slice(endtime=noise_trace.stats.starttime + 5 * 600 + 550)
    assert len(RSAM().compute(tr).datetime) == 6
