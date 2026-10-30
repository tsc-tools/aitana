## storage.utils.get_dt()


Infer the sampling interval of a time axis.


Usage

``` python
storage.utils.get_dt(times)
```


The median spacing, rounded to whole seconds (whole milliseconds below a second), so that time stamps slightly off the grid don't change the interval: times stored as float hours decode up to a microsecond off, and traces can start a few milliseconds late.


## Parameters


`times: array-like of datetime64`  
At least two sorted time stamps.


## Returns


`pd.Timedelta`  
The sampling interval.
