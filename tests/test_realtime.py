"""Features computed in 1 hour windows, as in real time, and stitched together by
aitana-storage."""

import warnings

import numpy as np
import pytest
from obspy import UTCDateTime
from scipy.signal import find_peaks

from aitana.features.spectral_features import SpectralFeatures
from aitana.features.ssam import SSAM
from aitana.storage import Storage
from aitana.testing import synthetic_trace


@pytest.mark.slow
def test_realtime_ssam(tmp_path_factory):
    """
    Test with 1 hour windows as in the real-time computation
    """
    savedir = tmp_path_factory.mktemp("realtime")
    linfreqs = np.arange(0, 25.1, 0.1)
    testpeaks = np.array([0.1, 3.0, 5.0, 6.0, 10.0])
    s = SSAM(
        interval=60,
        frequencies=linfreqs,
        resample_int=(None, "10min"),
        timestamp="start",
    )
    starttime = UTCDateTime(2020, 4, 21, 0, 0, 0)
    nsec = 86500  # Slightly longer than 24 h to avoid rounding issues
    step = 3600
    tr = synthetic_trace(nsec=nsec, starttime=starttime, gaps=True)
    sg = Storage(
        "test",
        savedir,
        starttime=starttime.datetime,
        endtime=(starttime + nsec).datetime,
    )
    store = sg.get_substore("WIZ", "00", "HHZ")
    startwin = starttime
    while True:
        endwin = startwin + step
        if endwin > tr.stats.endtime:
            break
        ntr = tr.slice(startwin, endwin)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store.save(s.compute(ntr))
        startwin = endwin

    ssam = store("ssam")
    assert ssam.shape == (251, 144)
    peaks = find_peaks(ssam.data[:, 0], height=1e-1)[0]
    np.testing.assert_array_almost_equal(linfreqs[peaks], testpeaks, 2)


@pytest.mark.slow
def test_realtime_spectral_features(tmp_path):
    """
    Test with 1 hour windows as in the real-time computation
    """
    sf = SpectralFeatures(filtertype="highpass", filterfreq=(0.5, None))
    starttime = UTCDateTime(2020, 4, 21, 0, 0, 0)
    nsec = 86500  # Slightly longer than 24 h to avoid rounding issues
    step = 3600
    tr = synthetic_trace(nsec=nsec, starttime=starttime, gaps=True)

    startwin = starttime
    sg = Storage("test", tmp_path)
    sg.starttime = starttime.datetime
    sg.endtime = (starttime + nsec).datetime
    st = sg.get_substore()
    while True:
        endwin = startwin + step
        if endwin > tr.stats.endtime:
            break
        ntr = tr.slice(startwin, endwin)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.save(sf.compute(ntr))
        startwin = endwin
    for feature in ["bandwidth", "central_freq", "predom_freq"]:
        assert st(feature).data.shape == (144,)
