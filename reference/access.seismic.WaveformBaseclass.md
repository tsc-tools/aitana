## access.seismic.WaveformBaseclass


Base class for waveform clients.


Usage

``` python
access.seismic.WaveformBaseclass()
```


Clients implement [get_raw](access.seismic.SeismicData.md#aitana.access.seismic.SeismicData.get_raw). Sensitivity removal, gap filling and trimming happen in [process](access.seismic.MockSDSWaveforms.md#aitana.access.seismic.MockSDSWaveforms.process), so that [SeismicData](access.seismic.SeismicData.md#aitana.access.seismic.SeismicData) can apply them to raw data read from its cache as well.


## Methods

| Name | Description |
|----|----|
| [get_raw()](#get_raw) | Get waveform data as delivered by the source. |
| [get_waveforms()](#get_waveforms) | Get waveform data with the instrument sensitivity removed. |
| [process()](#process) | Remove the instrument sensitivity, fill gaps and trim to the |

------------------------------------------------------------------------


#### get_raw()


Get waveform data as delivered by the source.


Usage

``` python
get_raw(net, site, loc, comp, start, end)
```


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


`Stream`  
Unmerged waveform data in counts.


------------------------------------------------------------------------


#### get_waveforms()


Get waveform data with the instrument sensitivity removed.


Usage

``` python
get_waveforms(net, site, loc, comp, start, end)
```


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


`Trace`  
Waveform data with the instrument sensitivity removed.


------------------------------------------------------------------------


#### process()


Remove the instrument sensitivity, fill gaps and trim to the


Usage

``` python
process(st, net, site, loc, comp, start, end)
```


requested time span.


##### Parameters


`st: Stream`  
Raw waveform data as returned by [get_raw](access.seismic.SeismicData.md#aitana.access.seismic.SeismicData.get_raw).

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


`Trace`  
Waveform data with the instrument sensitivity removed.
