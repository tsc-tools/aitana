## access.seismic.SeismicData


Usage

``` python
access.seismic.SeismicData(
    clients, chunk_size=None, cache_dir=None, cache_min_age=2 * 86400.0
)
```


## Methods

| Name | Description |
|----|----|
| [__init__()](#__init__) | Get seismic waveforms from the first of several clients that has |
| [get_raw()](#get_raw) | Get raw data per UTC day from the cache or from the first client |
| [get_waveforms()](#get_waveforms) | Yield waveform data for requested time span. |
| [to_sds()](#to_sds) | Path of a day file in the cache, writing `st` to it if given. |

------------------------------------------------------------------------


#### \_\_init\_\_()


Get seismic waveforms from the first of several clients that has


Usage

``` python
__init__(clients, chunk_size=None, cache_dir=None, cache_min_age=2 * 86400.0)
```


data, optionally through a local cache of raw day files.


##### Parameters


`clients: list of WaveformBaseclass`  
Waveform clients, tried in order.

`chunk_size: float = None`  
Length in seconds of the chunks [get_waveforms](access.seismic.SeismicData.md#aitana.access.seismic.SeismicData.get_waveforms) yields.

`cache_dir: str = None`  
Parent directory of the cache, by default `~/aitana_cache`. Raw day files go into its `raw` subdirectory in SDS layout. Files directly in `cache_dir` are processed data written by earlier versions and are never read.

`cache_min_age: float = 2 * 86400.0`  
Seconds after the end of a UTC day before it is cached.


------------------------------------------------------------------------


#### get_raw()


Get raw data per UTC day from the cache or from the first client


Usage

``` python
get_raw(net, site, loc, comp, start, end)
```


that has data. Days that ended more than `cache_min_age` ago are fetched whole and written to the cache.


##### Parameters


`net: str`  
Network, station, location and channel codes.

`site: str`  
Network, station, location and channel codes.

`loc: str`  
Network, station, location and channel codes.

`comp: str`  
Network, station, location and channel codes.

`start: UTCDateTime`  
Start and end of the requested time span.

`end: UTCDateTime`  
Start and end of the requested time span.


##### Returns


`st: Stream`  
Raw data between `start` and `end`.

`client: WaveformBaseclass or None`  
First client that delivered data, None if all of it came from the cache.


------------------------------------------------------------------------


#### get_waveforms()


Yield waveform data for requested time span.


Usage

``` python
get_waveforms(net, site, loc, comp, start, end, cache=False)
```


##### Parameters


`net: str`  
Network, station, location and channel codes. `comp` may include a simple wildcard of the form `??` or `?` or patterns like `HH?` to request multiple components (e.g. `HH?` -\> `HHZ, HHN, HHE`). Only a single wildcard character is currently supported and will expand to the common three-component set `Z, N, E` (or `Z, 1, 2` if those exist in the data - both tried).

`site: str`  
Network, station, location and channel codes. `comp` may include a simple wildcard of the form `??` or `?` or patterns like `HH?` to request multiple components (e.g. `HH?` -\> `HHZ, HHN, HHE`). Only a single wildcard character is currently supported and will expand to the common three-component set `Z, N, E` (or `Z, 1, 2` if those exist in the data - both tried).

`loc: str`  
Network, station, location and channel codes. `comp` may include a simple wildcard of the form `??` or `?` or patterns like `HH?` to request multiple components (e.g. `HH?` -\> `HHZ, HHN, HHE`). Only a single wildcard character is currently supported and will expand to the common three-component set `Z, N, E` (or `Z, 1, 2` if those exist in the data - both tried).

`comp: str`  
Network, station, location and channel codes. `comp` may include a simple wildcard of the form `??` or `?` or patterns like `HH?` to request multiple components (e.g. `HH?` -\> `HHZ, HHN, HHE`). Only a single wildcard character is currently supported and will expand to the common three-component set `Z, N, E` (or `Z, 1, 2` if those exist in the data - both tried).

`start: UTCDateTime`  
Start and end times.

`end: UTCDateTime`  
Start and end times.

`cache: bool = ``False`  
If True, raw data of UTC days that ended more than `cache_min_age` ago are read from, or written to, the local SDS cache.


##### Yields


`Trace or Stream`  
For a single component request, yields an ObsPy `Trace` (backwards compatible). For a wildcard multi-component request, yields an ObsPy `Stream` with one Trace per component for each chunk.


------------------------------------------------------------------------


#### to_sds()


Path of a day file in the cache, writing `st` to it if given.


Usage

``` python
to_sds(st, net, site, loc, comp, day)
```


##### Parameters


`st: Stream or None`  
Raw data of one UTC day. If None, nothing is written.

`net: str`  
Network, station, location and channel codes.

`site: str`  
Network, station, location and channel codes.

`loc: str`  
Network, station, location and channel codes.

`comp: str`  
Network, station, location and channel codes.

`day: UTCDateTime`  
Midnight at the start of the day.


##### Returns


`str`  
Path of the day file.
