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

`fill_gaps: bool`  
Whether to fill time gaps with NaN before writing, by default False


## Returns


`None`
