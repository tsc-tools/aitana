import inspect
import os

import numpy as np
import pytest
from aitana.access.util import test_signal
from aitana.features.rsam import RSAM
from obspy import UTCDateTime


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
    tr1 = test_signal(
        sinusoid=False,
        addchirp=False,
        starttime=starttime,
        offsets=offsets,
        amplitudes=amplitudes,
        phases=phases,
        frequencies=frequencies,
    )
    tr1.data *= 1e-9
    tr2 = test_signal(
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


def test_with_gaps(rsam_setup, setup_sds):
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

    return test_signal(
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
