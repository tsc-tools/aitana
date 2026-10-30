## storage.xarray2netcdf.xarray2netcdf()


Store an xarray dataset as an HDF5 file.


Usage

``` python
storage.xarray2netcdf.xarray2netcdf(
    xArray,
    fdir,
    group="original",
    timedim="datetime",
    archive_starttime=datetime(2000, 1, 1),
    resolution=None,
    mode="a"
)
```


## Parameters


`xArray: xarray.Dataset`  
Data to store.

`fdir: str`  
Directory to store data under.

`group: str = ``"original"`  
Hdf5 group name.

`timedim: str = ``"datetime"`  
Name of time dimension.

`archive_starttime: datetime = datetime(2000, 1, 1)`  
Start time of archive. A new archive's grid runs through the data's first time stamp and starts at its first point at or after this time; if the data start earlier, the archive starts with them.

`resolution: float = None`  
Time resolution of the data in hours. If None, the resolution is determined from the data, and time stamps further off the archive's grid than jitter raise a ValueError. If given, data go to the nearest grid point.
