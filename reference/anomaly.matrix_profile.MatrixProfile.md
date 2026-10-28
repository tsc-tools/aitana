## anomaly.matrix_profile.MatrixProfile


Score how unlike its own past each subsequence of a time series is.


Usage

``` python
anomaly.matrix_profile.MatrixProfile(window=36)
```


The raw score of a subsequence is its z-normalized distance to the most similar *earlier* subsequence (the left matrix profile), i.e. the score a self-join would have given it at the moment it arrived. Using only the past means a new shape that repeats (e.g. oscillating tremor) is still flagged on its first occurrences instead of matching its own repeats.

Z-normalization discards level and scale: a step change is flagged while a window contains the transition, but not once the series has settled on the new level. The detector therefore complements, rather than replaces, an amplitude threshold.

The history before `score_start` doubles as the reference distribution used to map raw scores to `[0, 1]`. Because the history is a trailing window, a new shape that persists for longer than the history becomes "normal".


## Parameters


`window: int = ``36`  
Subsequence length in samples.


## Attributes


`profile_: pd.Series`  
Raw score from the last [compute_anomaly_index](anomaly.matrix_profile.MatrixProfile.md#aitana.anomaly.matrix_profile.MatrixProfile.compute_anomaly_index) call, on the same index and NaN in the same places as the anomaly index: the left matrix profile, i.e. the z-normalized Euclidean distance (0 to `2 * sqrt(window)`) of each scored subsequence to its nearest earlier one.


## Methods

| Name | Description |
|----|----|
| [compute_anomaly_index()](#compute_anomaly_index) | Compute the anomaly index for the scored part of a series. |

------------------------------------------------------------------------


#### compute_anomaly_index()


Compute the anomaly index for the scored part of a series.


Usage

``` python
compute_anomaly_index(series, score_start)
```


##### Parameters


`series: pd.Series`  
Regularly sampled values with a sorted index (history followed by the period to score). NaNs are allowed.

`score_start: index label`  
First index label to score. Everything before it is history.


##### Returns


`pd.Series`  
Anomaly index on `series.index`, labelled by the last sample of each subsequence. Values are in `[0, 1]` from `score_start` onward and NaN before it; subsequences containing NaN are NaN.
