## storage.utils.merge_arrays()


Merge two xarray datasets with the same datetime index.


Usage

``` python
storage.utils.merge_arrays(
    xds_old, xds_new, timedim="datetime", resolution=None
)
```


## Parameters


`xds_old: xr.DataArray`  
Old array.

`xds_new: xr.DataArray`  
New array.

`resolution: float = None`  
Time resolution in hours.


## Returns


`xr.DataArray`  
Merged array.
