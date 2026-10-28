## testing.synthetic.synthetic_trace()


Produce a test signal for which we know where the peaks


Usage

``` python
testing.synthetic.synthetic_trace(
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
    channel="HHZ"
)
```


are in the spectrogram.

## Parameters

- **nsec** : `int` -- Length of the trace in seconds.
- **sampling_rate** : `float` -- Sampling rate of the signal in Hz.
- **starttime** : `~obspy.UTCDateTime` -- Starttime of the trace.
- **gaps** : `boolean` -- If 'True' add gaps to the test signal.
- **noise_std** : `float` -- Standard deviation of the noise.
- **sinusoid** : `bool` -- Add a signal with a sinusoidal frequency change.
- **station** : `str` -- Station name of the returned trace.
- **location** : `str` -- Location of the returned trace.

## Returns

`~obspy.Trace`  
Time series of test data
