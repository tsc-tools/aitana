## features.util.trace_window_times()


Subdivide an `obspy.core.trace.Trace` into windows of `window_len`


Usage

``` python
features.util.trace_window_times(
    trace,
    window_len,
    min_len=0.8,
)
```


length and iterate over window start & end times.

## Parameters

- **trace** : `obspy.core.trace.Trace` -- The data to be windowed.
- **window_len** : `float` -- The window length in seconds.
- **min_len** : `float, optional :yield: The window start & end times` -- Windows with length \< `min_len * window_len` are skipped, defaults to 0.8

## Returns

`obspy.core.utcdatetime.UTCDateTime``, ``obspy.core.utcdatetime.UTCDateTime``,`
