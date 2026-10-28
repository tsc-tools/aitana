## storage.utils.floor_datetime()


Floor a datetime to the latest multiple of a given interval.


Usage

``` python
storage.utils.floor_datetime(
    dt,
    interval,
)
```


Assumes `dt` represents a UTC time (naive or tz-aware is fine) and aligns against the Unix epoch 1970-01-01T00:00:00Z. The interval is in seconds (int/float) or a timedelta. Returns a datetime with the same "naive vs aware" form as `dt`.


## Examples

``` python
>>> from datetime import datetime
>>> floor_datetime(datetime.fromisoformat('2025-11-27T10:12:43'), 600)
```

datetime.datetime(2025, 11, 27, 10, 10)
