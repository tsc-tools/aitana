## pipelines.visualise.scoring_plot()


Plot the forecast probabilities and the evaluation windows.


Usage

``` python
pipelines.visualise.scoring_plot(
    forecast, threshold, scoring_function, debug=False, ax=None
)
```


## Arguments:

    forecast: pandas.Series
        The forecast probabilities.
    threshold: float
        The threshold value.
    scoring_function: Callable
        Function that returns (stats, time_windows) given (threshold, trace).
    debug: bool, optional
        Whether to print debug information.
    ax: matplotlib Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
