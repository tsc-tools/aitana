## anomaly.evaluation.inject_change_points()


Inject persistent regime shifts (change points) into a DataFrame.


Usage

``` python
anomaly.evaluation.inject_change_points(
    df,
    columns=None,
    n_changes=3,
    offset_range="hard",
    mode="independent",
    coupled_groups=None,
    rng=None
)
```


Unlike [inject_faults](anomaly.evaluation.inject_faults.md#aitana.anomaly.evaluation.inject_faults), which perturbs isolated timesteps (sensor glitches), each injected change point offsets the signal from its onset onward until the next change point (or end of series) -- a synthetic analogue of a genuine regime change, for evaluating change-point/state detectors (e.g. ClaSP) rather than pointwise anomaly detectors.


## Parameters


`df: pd.DataFrame`  
Input time series DataFrame. Index should be a DatetimeIndex.

`columns: list of str = None`  
Columns to target in independent mode. Defaults to all columns. Ignored in coupled mode (only columns in coupled_groups are affected).

`n_changes: int = ``3`  
Number of change points to inject per column (independent) or per group (coupled).

`offset_range: str or tuple of float = ``"hard"`  
Per-change-point offset magnitude as a multiplier of each column's diff std, accumulated at each successive change point. Same presets as [inject_faults](anomaly.evaluation.inject_faults.md#aitana.anomaly.evaluation.inject_faults).

`mode: str = ``"independent"`  
`"independent"`: per-column change points at independent onsets. `"coupled"`: same onsets injected across each group in `coupled_groups`.

`coupled_groups: list of list of str = None`  
Groups of columns for coupled injection. Required when `mode="coupled"`.

`rng: int, np.random.Generator, or None = None`  
Random seed or generator for reproducibility.


## Returns


`corrupted_df: pd.DataFrame`  
Copy of `df` with persistent regime shifts injected.

`change_points: pd.DataFrame`  
Boolean DataFrame of the same shape as `df`; True at the onset timestep of a change point (not across the whole shifted segment).
