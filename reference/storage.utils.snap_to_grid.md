## storage.utils.snap_to_grid()


Round time stamps to the nearest point of a regular grid.


Usage

``` python
storage.utils.snap_to_grid(
    times,
    interval,
    origin=None,
    check=False,
)
```


## Parameters


`times: array-like of datetime64`  
Time stamps to snap.

`interval: pd.Timedelta`  
Grid spacing.

`origin: datetime64 - like = None`  
A point of the grid. By default the first time stamp, rounded to whole seconds (whole milliseconds for spacings below a second), so that the grid keeps the phase of the data, e.g. time stamps at window centres.

`check: bool = ``False`  
Raise a ValueError if a time stamp is further from its grid point than jitter explains: 1% of the interval, at most a second.


## Returns


`np.ndarray`  
datetime64\[ns\] time stamps `origin + k * interval`; halfway cases round up.
