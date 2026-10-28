## access.seismic.S3Waveforms


Usage

``` python
access.seismic.S3Waveforms(
    debug=False,
    fill_value=np.nan,
)
```


## Methods

| Name | Description |
|----|----|
| [__init__()](#__init__) | Get seismic waveforms from GeoNet's open data archive on AWS S3. |
| [get_raw()](#get_raw) | Download waveforms. |

------------------------------------------------------------------------


#### \_\_init\_\_()


Get seismic waveforms from GeoNet's open data archive on AWS S3.


Usage

``` python
__init__(debug=False, fill_value=np.nan)
```


##### Parameters


`debug: bool = ``False`  
If True, debug messages will be printed to the console.

`fill_value: int, float, 'interpolate', np.nan, or None = np.nan`  
Value to use for filling gaps in the data.


------------------------------------------------------------------------


#### get_raw()


Download waveforms.


Usage

``` python
get_raw(net, site, loc, comp, start, end)
```


##### Parameters


`net: str`  
Network code.

`site: str`  
Station code.

`loc: str`  
Location code.

`comp: str`  
Channel code.

`start: UTCDateTime`  
Start time of the data to be downloaded.

`end: UTCDateTime`  
End time of the data to be downloaded.


##### Returns


`Stream`  
The day files' waveform data in counts, trimmed to the requested time span.
