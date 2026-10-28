## access.api.AitanaAPI


Usage

``` python
access.api.AitanaAPI(
    rootdir,
    backend="netcdf",
    fdsn_urls=("https://service.geonet.org.nz", "https://service-nrt.geonet.org.nz")
)
```


## Methods

| Name | Description |
|----|----|
| [waveform()](#waveform) | Get instrument-corrected waveforms. |

------------------------------------------------------------------------


#### waveform()


Get instrument-corrected waveforms.


Usage

``` python
waveform(
    network,
    station,
    location,
    channel,
    starttime,
    duration,
    maxResLimit=1.0,
    resolution="full"
)
```
