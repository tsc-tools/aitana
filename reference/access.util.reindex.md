## access.util.reindex()


Reindex and forward fill to generate a


Usage

``` python
access.util.reindex(
    df,
    dates,
    fill_method=None,
    ffill_interval=14,
)
```


timeseries that can be used to set the evidence for a BN.

## Parameters

- **df** : `pandas.DataFrame` -- Dataframe to reindex
- **dates** : `pandas.DataTimeIndex` -- new date index
- **ffill_interval** : `int` -- Best-by interval for data
