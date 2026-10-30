## anomaly.anomaly_base.MinMaxScale


Fill gaps with the median and scale each feature column to \[0, 1\].


Usage

``` python
anomaly.anomaly_base.MinMaxScale(
    store,
    features,
)
```


A `transform` for [aitana.storage.storage.AitanaDataset](storage.storage.AitanaDataset.md#aitana.storage.storage.AitanaDataset). Minimum, maximum and median of each feature come from its whole archive, computed once with [min_max_scale_netcdf](anomaly.anomaly_base.min_max_scale_netcdf.md#aitana.anomaly.anomaly_base.min_max_scale_netcdf) and cached as `scaling_params_<feature>.json` in the store.


## Parameters


`store: aitana.storage.Path`  
Store holding the features, in NetCDF format.

`features: list of str`  
Feature names, in the order given to the dataset.


## Methods

| Name | Description |
|----|----|
| [__call__()](#__call__) | Scale a block of time steps. |

------------------------------------------------------------------------


#### \_\_call\_\_()


Scale a block of time steps.


Usage

``` python
__call__(block)
```


##### Parameters


`block: np.ndarray`  
Array of shape `[n_timesteps, n_features]`, with columns in the dataset's order.


##### Returns


`np.ndarray`  
The scaled block; NaN are replaced by the median before scaling.
