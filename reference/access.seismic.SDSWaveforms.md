## access.seismic.SDSWaveforms


Usage

``` python
access.seismic.SDSWaveforms(
    sds_dir,
    fdsn_urls,
    fill_value=np.nan,
)
```


## Methods

| Name | Description |
|----|----|
| [__init__()](#__init__) | Get seismic waveforms from a local SDS archive. |

------------------------------------------------------------------------


#### \_\_init\_\_()


Get seismic waveforms from a local SDS archive.


Usage

``` python
__init__(sds_dir, fdsn_urls, fill_value=np.nan)
```


## Parameters

- **fill_value** : `int, float, 'interpolate', np.nan, or None` -- Default value used to fill gaps and pad traces to cover the whole time period defined by `startdate` and `enddate`. If 'interpolate' is chosen, the traces will not be padded.
