## features.ssam.SSAM


SSAM is basically the same as computing a spectrogram. Data are split into


Usage

``` python
features.ssam.SSAM(
    interval=10,
    per_lap=0,
    scale_by_freq=True,
    smooth=None,
    frequencies=np.linspace(0, 25, 251),
    fbfilters=16,
    nhob=8,
    timestamp="center",
    resample_int=None,
    configfile=None
)
```


interval length segments and the spectrum of each section is computed. Before computing the spectrum, the mean is removed and a hanning window is applied to each segment. The percentage of overlap of each segment can be specified with per_lap and Welch-type smoothing can be applied.

## Parameters

- **interval** : `int` -- Segment length in seconds.
- **per_lap** : `float or int` -- If the value is less than 1 it is treated as the percentage of segment overlap; else it is the step size in sample points
- **scale_by_freq** : `boolean` -- Divide by the sampling frequency so that density function has units of dB/Hz and can be integrated by the frequency values.
- **smooth** : `int` -- The number of windows over which to average the spectrogram.
- **frequencies** : `numpy.ndarray` -- Frequencies at which to return the spectrogram. This uses linear interpolation to compute the spectrogram at the given frequencies from the original spectrogram.
- **timestamp** : `str` -- Can be either 'center' or 'start'. If 'center', the timestamp of each spectrum is the center of the windows from which the spectrum was computed. If 'start' it is the timestamp of the first sample of the first window.
- **resample_int** : `(str, str)` -- Interval to upsample the dataset using linear interpolation over the first interval and then downsample using the mean over the second interval.
- **configfile** : `str or dict` -- Configuration as a .yml file or a dictionary.

``` python
>>> from aitana.features.ssam import SSAM
>>> from aitana.testing import synthetic_trace
>>> import numpy as np
>>> tr = synthetic_trace(gaps=True)
>>> s = SSAM(interval=60, per_lap=.9, smooth=10,
...          frequencies=np.arange(0, 25.1, .1))
```


## Methods

| Name | Description |
|----|----|
| [compute()](#compute) | Compute SSAM for the given trace. |
| [filterbank()](#filterbank) | Computes the integral in an array of bandpass filters that |
| [sonogram()](#sonogram) | Computes the sonogram of the given data which can be windowed or not. |

------------------------------------------------------------------------


#### compute()


Compute SSAM for the given trace.


Usage

``` python
compute(trace)
```


## Parameters

- **trace** : `obspy.Trace` -- The seismic waveform data.

## Returns

`tuple`  
spectrogram, frequencies, times


------------------------------------------------------------------------


#### filterbank()


Computes the integral in an array of bandpass filters that


Usage

``` python
filterbank(nfilters, nfft, fs)
```


separates the input signal into multiple components. Each component consists of a single frequency sub-band of the original signal.

## Parameters

- **nfilters** : `int`
- **nfft** : `int` -- Number of samples
- **fs** : `float` -- Sampling frquency


------------------------------------------------------------------------


#### sonogram()


Computes the sonogram of the given data which can be windowed or not.


Usage

``` python
sonogram(data, fs, fc1, nofb=8)
```


The sonogram is determined by the power in half octave bands of the given data.

If data are windowed the analytic signal and the envelope of each window is returned.

## Parameters

- **data** : `~numpy.ndarray` -- Data to make envelope of.
- **fs** -- Sampling frequency in Hz.
- **fc1** -- Center frequency of lowest half octave band.
- **nofb** -- Number of half octave bands.
- **no_win** -- Number of data windows.

## Returns

  
Central frequencies, Half octave bands.
