## pipelines.scoring.compute_rates()


Compute the forecasted rates for positive and negative windows.


Usage

``` python
pipelines.scoring.compute_rates(
    model,
    pew,
    debug=False,
)
```


## Arguments:

    model: pandas.DataFrame
        The model probabilities.
    pew: int
        The pre-eruption window size.
    debug: bool, optional
        Whether to print debug information.


## Returns

    dict: The forecasted rates for positive and negative windows as well as the overall rate.
