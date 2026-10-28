## features.dsar.DSAR


DSAR is the ratio of


Usage

``` python
features.dsar.DSAR(
    interval=600.0,
    filtertype="bandpass",
    lowerfreqband=(4.5, 8),
    higherfreqband=(8, 16),
    configfile=None,
    reindex=True
)
```


low frequency to high frequency amplitudes, as suggested in: https://doi.org/10.1130/G46107.1

The time-domain displacement signal is bandpass filtered in two frequency bands, 4.5 to 8 Hz and 8 to 16 Hz, and DSAR is defined as the median ratio of the absolute signals. The quantity is interpreted in the publication as a change in seismic attenuation in the vicinity of the seismic station.


## Methods

| Name | Description |
|----|----|
| [compute()](#compute) | Compute the displacement seismic amplitude ratio (DSAR). |

------------------------------------------------------------------------


#### compute()


Compute the displacement seismic amplitude ratio (DSAR).


Usage

``` python
compute(trace)
```


TODO:

Publication is quite vague and doesn't have enough detail to understand the method. Unclear whether median is every 10 minutes or every day. This affects the bootstrap calc- ulation too, but this might not be necessary..

## Parameters

- **stream** : `str` -- The SLCN code for the seismic stream
- **startdate** : `obspy.UTCDateTime` -- Datetime of the first sample.
- **enddate** : `obspy.UTCDateTime` -- Datetime of the last sample.
- **interval** : `int, optional` -- Length of the time interval in seconds over which to compute DSAR (default 10 mins)
