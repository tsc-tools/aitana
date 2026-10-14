import math as M

import numpy as np
from obspy import Trace, UTCDateTime
from scipy.signal import chirp, sweep_poly


def synthetic_trace(
    nsec=3600,
    sampling_rate=100.0,
    frequencies=[0.1, 3.0, 10.0],
    amplitudes=[0.1, 1.0, 0.7],
    phases=[0.0, np.pi * 0.25, np.pi],
    offsets=[0.0, 0.0, 0.0],
    starttime=UTCDateTime(1970, 1, 1),
    gaps=False,
    noise=True,
    noise_std=0.5,
    sinusoid=True,
    addchirp=True,
    network="NZ",
    station="BLUB",
    location="",
    channel="HHZ",
):
    """
    Produce a test signal for which we know where the peaks
    are in the spectrogram.

    :param nsec: Length of the trace in seconds.
    :type nsec: int
    :param sampling_rate: Sampling rate of the signal in Hz.
    :type sampling_rate: float
    :param starttime: Starttime of the trace.
    :type starttime: :class:`~obspy.UTCDateTime`
    :param gaps: If 'True' add gaps to the test signal.
    :type gaps: boolean
    :param noise_std: Standard deviation of the noise.
    :type noise_std: float
    :param sinusoid: Add a signal with a sinusoidal frequency change.
    :type sinusoid: bool
    :param station: Station name of the returned trace.
    :type station: str
    :param location: Location of the returned trace.
    :type location: str
    :return: Time series of test data
    :rtype: :class:`~obspy.Trace`
    """
    t = np.arange(0, nsec, 1 / sampling_rate)
    signals = []
    # Some constant signals with different phases
    for f, A, ph, dt in zip(frequencies, amplitudes, phases, offsets):
        _s = np.zeros(t.size)
        dt_idx = int(dt * sampling_rate)
        _s[dt_idx:] = A * np.sin(2.0 * np.pi * f * t[dt_idx:] + ph)
        signals.append(_s)
    if addchirp:
        # Frequency-swept signals
        # Chirp
        s4 = 0.8 * chirp(t, 5, t[-1], 15, method="quadratic")
        signals.append(s4)

    if sinusoid:
        vals = [6]
        for k in range(0, 7):
            vals.append(
                2
                * (-1) ** k
                * np.power(2 * np.pi / nsec, 2 * k + 1)
                / M.factorial(2 * k + 1)
            )
            vals.append(0)
        p = np.poly1d(np.array(vals[::-1]))
        s5 = sweep_poly(t, p)
        signals.append(s5)

    # add some noise
    if noise:
        rs = np.random.default_rng(42)
        noise = rs.normal(loc=0.0, scale=noise_std, size=t.size)
        for s in signals:
            noise += s
        signal = noise
    else:
        signal = signals[0]
        for s in signals[1:]:
            signal += s
    stats = {
        "network": network,
        "station": station,
        "location": location,
        "channel": channel,
        "npts": len(signal),
        "sampling_rate": sampling_rate,
        "mseed": {"dataquality": "D"},
    }
    stats["starttime"] = starttime
    stats["endtime"] = stats["starttime"] + nsec
    if gaps:
        p1 = (0.27, 0.33)
        p2 = (0.69, 0.83)
        p3 = (0.95, 1)
        for pmin, pmax in [p1, p2, p3]:
            idx0 = int(pmin * nsec * sampling_rate)
            idx1 = int(pmax * nsec * sampling_rate)
            signal[idx0:idx1] = np.nan
    return Trace(data=signal, header=stats)
