## access.seismic.MockSDSWaveforms


Mock SDSWaveforms class for testing by creating


Usage

``` python
access.seismic.MockSDSWaveforms(sds_dir)
```


synthetic data for the requested streams.


## Methods

| Name | Description |
|----|----|
| [__init__()](#__init__) | Get seismic waveforms from a local SDS archive. |
| [generate_dataset()](#generate_dataset) | Generate a test dataset for the integration tests. |
| [get_raw()](#get_raw) | Generate synthetic daily files that start at midnight on the |
| [process()](#process) | Return the first trace unchanged, since synthetic data have no |
| [save2sds()](#save2sds) | Save a trace to a SDS directory structure. |

------------------------------------------------------------------------


#### \_\_init\_\_()


Get seismic waveforms from a local SDS archive.


Usage

``` python
__init__(sds_dir)
```


------------------------------------------------------------------------


#### generate_dataset()


Generate a test dataset for the integration tests.


Usage

``` python
generate_dataset(rootdir, net, site, loc, comp, start, end)
```


##### Parameters


`rootdir: str`  
Parent directory for the test dataset.

`start: str`  
Start time for the test dataset in ISO 8601 format.

`end: str`  
End time for the test dataset in ISO 8601 format.


------------------------------------------------------------------------


#### get_raw()


Generate synthetic daily files that start at midnight on the


Usage

``` python
get_raw(net, site, loc, comp, startdate, enddate)
```


startdate and end at midnight on the enddate. The return the requested timespan from these files.


------------------------------------------------------------------------


#### process()


Return the first trace unchanged, since synthetic data have no


Usage

``` python
process(st, net, site, loc, comp, startdate, enddate)
```


station metadata.


##### Parameters


`st: Stream`  
Synthetic waveform data as returned by [get_raw](access.seismic.SeismicData.md#aitana.access.seismic.SeismicData.get_raw).

`net: str`  
Network, station, location and channel codes.

`site: str`  
Network, station, location and channel codes.

`loc: str`  
Network, station, location and channel codes.

`comp: str`  
Network, station, location and channel codes.

`startdate: UTCDateTime`  
Start and end of the requested time span.

`enddate: UTCDateTime`  
Start and end of the requested time span.


##### Returns


`Trace`  
First trace of `st`.


------------------------------------------------------------------------


#### save2sds()


Save a trace to a SDS directory structure.


Usage

``` python
save2sds(trace, rootdir)
```


##### Parameters


`trace: ``obspy.Trace`  
Seismic/acoustic trace to be written to disk.

`rootdir: str`  
Root directory for the SDS directory structure.
