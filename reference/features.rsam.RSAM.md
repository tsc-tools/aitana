## features.rsam.RSAM


Compute RSAM as the mean of the absolute value of a signal filtered


Usage

``` python
features.rsam.RSAM(
    interval=600.0,
    per_lap=0,
    timestamp="start",
    filterfreq=(2, 5),
    configfile=None,
    max_gap_fraction=None
)
```


by default between 2 and 5 Hz and the proportion of signal energy in the RSAM bandwidth, relative to a 0.5-10 Hz bandwidth.

## Parameters

- **interval** : `int, optional` -- Length of the time interval in seconds over which to compute RSAM.
- **per_lap** : `float or int` -- If the value is less than 1 it is treated as the percentage of segment overlap; else it is the step size in sample points
- **timestamp** : `str` -- Can be either 'center' or 'start'. If 'center', the timestamp of each value is the center of the windows from which the value was computed. If 'start' it is the timestamp of the first sample of the first window.
- **filterfreq** : `tuple, optional` -- The low and high cutoff frequency for bandpass filter.
- **configfile** : `str or dict` -- Configuration as a .yml file or a dictionary.
- **max_gap_fraction** : `float, optional` -- If given, windows in which a larger fraction of samples is missing (NaN, or padding past the end of the trace) are set to NaN instead of being computed across the interpolated gap.


## Methods

| Name | Description |
|----|----|
| [compute()](#compute) | :param trace: The seismic waveform data |

------------------------------------------------------------------------


#### compute()


## Parameters

- **trace** -- The seismic waveform data


Usage

``` python
compute(trace)
```


## Parameters

- **trace** : `obspy.Trace`
