## storage.xarray2zarr.xarray2zarr()


Write xarray dataset to zarr files.


Usage

``` python
storage.xarray2zarr.xarray2zarr(
    xds,
    path,
    group="original",
    chunk_size=1000,
    timedim="datetime",
    interval=None,
    archive_starttime=datetime(2000, 1, 1)
)
```


## Parameters


`xds: xr.Dataset`  
Dataset to write.

`path: str`  
Path to write the dataset.

`mode: str`  
Write mode, by default 'a'.

`group: str = ``"original"`  
Group name, by default 'original'

`chunk_size: int = ``1000`  
Chunk size as the number of days.

`timedim: str = ``"datetime"`  
Name of the time dimension, by default 'datetime'

`interval: str = None`  
Sampling interval. If None, it is inferred from the data, and time stamps further off the store's grid than jitter raise a ValueError. If given, data go to the nearest grid point.

`archive_starttime: datetime = datetime(2000, 1, 1)`  
Start of a new store; its grid runs through the data's time stamps.

`fill_gaps: bool`  
Whether to fill time gaps with NaN before writing, by default False


## Returns


`None`
