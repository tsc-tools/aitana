import numpy as np
from obspy import UTCDateTime
import pytest

from aitana.features.dsar import DSAR
from aitana.testing import synthetic_trace

@pytest.fixture(scope="session")
def decreasing_hf_signal():
    interval = 600.0
    lf_freq, hf_freq = 6.0, 12.0
    hf_levels = [1.0, 0.5, 0.1]  # target cumulative hf amplitude per window
    hf_deltas = [hf_levels[0]] + list(np.diff(hf_levels))
    n_hf = len(hf_levels)

    return synthetic_trace(
        nsec=interval * n_hf,
        frequencies=[lf_freq] + [hf_freq] * n_hf,
        amplitudes=[1.0] + hf_deltas,
        phases=[0.0] * (n_hf + 1),
        offsets=[0.0] + [i * interval for i in range(n_hf)],
        starttime=UTCDateTime(2020, 1, 1),
        sinusoid=False,
        addchirp=False,
        noise=False,
    )

def test_dsar_increases_as_hf_content_decreases(decreasing_hf_signal):
    dsar = DSAR(filtertype='bandpass', lowerfreqband=(4.5, 8),
                higherfreqband=(8, 16))
    xdf = dsar.compute(decreasing_hf_signal)
    dsar_vals = xdf['dsar'].values

    assert len(dsar_vals) == 3
    assert not np.isnan(dsar_vals).any()
    assert np.all(np.diff(dsar_vals) > 0)
