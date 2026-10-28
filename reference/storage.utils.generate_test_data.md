## storage.utils.generate_test_data()


Generate a 1D or 2D feature for testing.


Usage

``` python
storage.utils.generate_test_data(
    dim=1,
    ndays=30,
    nfreqs=10,
    tstart=datetime.now(),
    freq="10min",
    intervals=None,
    feature_names=None,
    seed=42,
    freq_names=None,
    add_nans=True
)
```


## Parameters


`dim: int = ``1`  
Dimension of the data (1 or 2).

`ndays: int = ``30`  
Number of days to generate data for.

`nfreqs: int = ``10`  
Number of frequencies (only for dim=2).

`tstart: datetime = datetime.now()`  
Start time of the data.

`freq: str = ``"10min"`  
Frequency of the data (e.g., '10min').

`intervals: int = None`  
Number of intervals to generate. If None, calculated from ndays and freq.

`feature_names: list = None`  
Names of the features to generate.

`seed: int = ``42`  
Random seed for reproducibility.

`freq_names: list = None`  
Names of the frequency dimensions (only for dim=2).

`add_nans: bool = ``True`  
Whether to add NaN values to the data.


## Returns


`xr.Dataset`  
Generated test dataset.
