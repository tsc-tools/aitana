## pipelines.scoring.evaluate_threshold()


Evaluate the number of true positives, false positives, true negatives and false negatives for a given threshold.


Usage

``` python
pipelines.scoring.evaluate_threshold(
    thresh, model, debug=False, pew=90, return_windows=False
)
```


## Arguments:

    thresh: float
        The threshold value.
    model: pandas.DataFrame
        The forecasted probabilities.
    debug: bool, optional
        Whether to print debug information.
    pew: int, optional
        The pre-eruption window size.
    return_windows: bool, optional
        Whether to return the positive and negative windows.


## Returns

    dict: The evaluation results.
    list: A list of dictionaries for the tp, fp, tn, fn windows.
