## features.util.round_time()


Find closest multiple of interval to time.


Usage

``` python
features.util.round_time(
    time,
    interval,
)
```


## Parameters

- **time** : `obspy.UTCDateTime` -- Time to round
- **interval** : `int` -- Interval length in seconds

## Returns

`obspy.UtCDateTime`  
The new time that is a multiple of interval
