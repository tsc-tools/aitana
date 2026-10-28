## features.util.trace_window_data()


Subdivide an `obspy.core.trace.Trace` into windows of `interval`


Usage

``` python
features.util.trace_window_data(
    trace,
    window_len,
    min_len=0.8,
)
```


length and iterate over windows.

## Parameters

- **trace** : `obspy.core.trace.Trace` -- The data to be windowed.
- **window_len** : `float` -- The window length in seconds.
- **min_len** : `float, optional :yield: A trace data windowNone` -- Windows with length \< `min_length * window_len` are skipped, defaults to 0.8

## Returns

`obspy.core.trace.Trace`
