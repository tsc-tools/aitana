## access.util.gradient()


Compute gradient for time series by first smoothing


Usage

``` python
access.util.gradient(
    df,
    period="14D",
)
```


the time series and then computing the first-order difference.


## Parameters

    :param df: Dataframe
    :type df: ~pandas.DataFrame
    :param period: Period over which to compute a rolling mean.
    :type period: str
