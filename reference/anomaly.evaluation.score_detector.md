## anomaly.evaluation.score_detector()


Score a detector against the fault-injection ground-truth mask.


Usage

``` python
anomaly.evaluation.score_detector(
    anomaly_index,
    mask,
    threshold,
)
```


Computes the full ROC curve and AUC using pointwise classification (not eruption-window apportionment).


## Parameters


`anomaly_index: array-like of float`  
Per-timestep anomaly scores in \[0, 1\].

`mask: pd.DataFrame or array-like of bool`  
Ground-truth fault mask. If a DataFrame, any faulted column at a timestep counts as a positive label (logical OR across columns).

`threshold: float`  
Operating threshold for computing binary alarm bits (TP/FP/TN/FN).


## Returns


`dict with keys:`  
`fpr`, `tpr`, `thresholds` -- full ROC curve arrays. `auc` -- area under the ROC curve. `tp`, `fp`, `tn`, `fn` -- confusion matrix counts at `threshold`.
