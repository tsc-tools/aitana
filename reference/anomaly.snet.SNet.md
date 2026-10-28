## anomaly.snet.SNet


Usage

``` python
anomaly.snet.SNet(max_gap_fraction=0.1)
```


## Methods

| Name | Description |
|----|----|
| [preprocess()](#preprocess) | Resample traces onto a common grid at `sampling_rate_hertz`. |

------------------------------------------------------------------------


#### preprocess()


Resample traces onto a common grid at `sampling_rate_hertz`.


Usage

``` python
preprocess(st, starttime, endtime)
```


Gaps (NaN) are filled with the last valid sample so that each trace can be detrended and low-pass filtered against aliasing, then the trace is interpolated onto the grid. Grid samples in a gap or outside the trace are NaN, for [compute](features.rsam.RSAM.md#aitana.features.rsam.RSAM.compute) to mask.


##### Parameters


`st: obspy.Stream`  
Traces of the components, e.g. Z, N, E.

`starttime: obspy.UTCDateTime`  
First sample of the grid.

`endtime: obspy.UTCDateTime`  
End of the grid (exclusive).


##### Returns


`obspy.Stream`  
One trace per input trace, each with `(endtime - starttime) * sampling_rate_hertz` samples from `starttime`.
