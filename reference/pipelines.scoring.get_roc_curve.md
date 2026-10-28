## pipelines.scoring.get_roc_curve()


Compute the ROC curve for a given model and thresholds.


Usage

``` python
pipelines.scoring.get_roc_curve(
    model,
    thresholds,
    func,
    debug=False,
)
```


## Arguments:

    model: pandas.DataFrame
        The model probabilities.
    thresholds: list
        The thresholds to evaluate.
    func: function
        The evaluation function.
    debug: bool, optional
        Whether to print debug information.
