## anomaly.evaluation.simulate_regime_series()


Generate a synthetic multivariate time series stitched from discrete regimes.


Usage

``` python
anomaly.evaluation.simulate_regime_series(
    state_sequence,
    segment_length,
    n_features=1,
    state_noise_std=None,
    state_means=None,
    rng=None
)
```


Each entry in `state_sequence` selects a regime label (e.g. `["A", "B", "A", "C"]`); repeated labels reuse the same mean, so recurring regimes are known ground truth for state-detection algorithms (e.g. CLaP), and segment boundaries are the known ground-truth change points for change-point detectors (e.g. ClaSP).


## Parameters


`state_sequence: list of str`  
Sequence of regime labels, e.g. `["A", "B", "A", "C"]`.

`segment_length: int or list of int`  
Number of timesteps per regime. A single int applies uniformly to every segment; a list must have the same length as `state_sequence`.

`n_features: int = ``1`  
Number of feature columns to generate.

`state_noise_std: dict = None`  
Mapping from regime label to the standard deviation of that state's additive Gaussian noise. If not given, every state defaults to 1.0.

`state_means: dict = None`  
Mapping from regime label to a length-`n_features` array of means. If not given, one is drawn per unique label from `Normal(0, 5)`.

`rng: int, np.random.Generator, or None = None`  
Random seed or generator for reproducibility.


## Returns


`series: pd.DataFrame`  
Synthetic `(n_timesteps, n_features)` series with a RangeIndex.

`change_points: np.ndarray of int`  
Index of the first timestep of each new segment (excludes 0).

`labels: pd.Series`  
Per-timestep ground-truth regime label, aligned to `series`.
