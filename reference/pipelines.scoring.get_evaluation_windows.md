## pipelines.scoring.get_evaluation_windows()


Get the evaluation windows for the forecasted rates.


Usage

``` python
pipelines.scoring.get_evaluation_windows(
    starttime,
    endtime,
    pew,
)
```


## Arguments:

    starttime: pd.Timestamp
        The start time of the forecast.
    endtime: pd.Timestamp
        The end time of the forecast.
    pew: int
        The pre-eruption window size.


## Returns

    positive_windows: list
        The pre-eruption windows.
    negative_windows: list
        Non pre-eruption windows.
