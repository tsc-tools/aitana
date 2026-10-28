## storage.utils.fill_time_gaps()


Fill gaps in time series with NaN values by reindexing to a complete datetime range.


Usage

``` python
storage.utils.fill_time_gaps(
    xds,
    timedim="datetime",
)
```


## Parameters


`xds: xr.Dataset`  
Input dataset with potential time gaps

`freq: str`  
Frequency string (e.g., 'H', 'D', '15min'). If None, will try to infer.

`timedim: str = ``"datetime"`  
Name of the time dimension, by default 'datetime'


## Returns


`xr.Dataset`  
Dataset with gaps filled with NaN
