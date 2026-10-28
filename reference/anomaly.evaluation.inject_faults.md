## anomaly.evaluation.inject_faults()


Inject synthetic faults into a DataFrame for anomaly detection evaluation.


Usage

``` python
anomaly.evaluation.inject_faults(
    df,
    columns=None,
    rate=0.06,
    offset_range="hard",
    mode="independent",
    coupled_groups=None,
    rng=None
)
```


## Parameters


`df: pd.DataFrame`  
Input time series DataFrame. Index should be a DatetimeIndex.

`columns: list of str = None`  
Columns to target in independent mode. Defaults to all columns. Ignored in coupled mode (only columns in coupled_groups are affected).

`rate: float = ``0.06`  
Per-timestep fault probability (independent) or per-timestep probability that a coupled group fires (coupled).

`offset_range: str or tuple of float = ``"hard"`  
Offset magnitude as a multiplier of each column's diff std. `"hard"`: (1, 2) -- small, harder to detect. `"easy"`: (4, 6) -- large, easier to detect. Tuple `(k1, k2)`: custom multiplier range.

`mode: str = ``"independent"`  
`"independent"`: per-column Bernoulli injection. `"coupled"`: same fault timestamps injected across each group in `coupled_groups`, mimicking a multi-stream precursor.

`coupled_groups: list of list of str = None`  
Groups of columns for coupled injection. Required when `mode="coupled"`.

`rng: int, np.random.Generator, or None = None`  
Random seed or generator for reproducibility.


## Returns


`corrupted_df: pd.DataFrame`  
Copy of `df` with faults injected.

`mask_df: pd.DataFrame`  
Boolean DataFrame of the same shape as `df`; True where a fault was injected.
