## access.seismic.PostProcess


Class for checking waveforms and station metadata for valid values,


Usage

``` python
access.seismic.PostProcess(
    st=Stream(),
    inv=Inventory(),
    inv_dt=Inventory(),
    startdate=None,
    enddate=None,
    loc="*",
    comp="*Z",
    fill_value=np.nan
)
```


filling waveform data gaps and removing instrument sensitivity


## Methods

| Name | Description |
|----|----|
| [check_case_incomplete_station_metadata()](#check_case_incomplete_station_metadata) | The case where a a channel period start/stops during the |
| [check_case_multiple_channel_periods()](#check_case_multiple_channel_periods) | Checks if there are multiple channel operation periods in a given |
| [check_case_no_data()](#check_case_no_data) | Checks if either no stream, or no station metadata exist. |
| [output_case_multiple_channel_periods()](#output_case_multiple_channel_periods) | Stitches two response periods together to make a single merged |

------------------------------------------------------------------------


#### check_case_incomplete_station_metadata()


The case where a a channel period start/stops during the


Usage

``` python
check_case_incomplete_station_metadata()
```


selected time window. Some parts of the trace in the stream might not correspond to a valid period according to station metadata.


------------------------------------------------------------------------


#### check_case_multiple_channel_periods()


Checks if there are multiple channel operation periods in a given


Usage

``` python
check_case_multiple_channel_periods()
```


time period. If so, different response removal is required.


------------------------------------------------------------------------


#### check_case_no_data()


Checks if either no stream, or no station metadata exist.


Usage

``` python
check_case_no_data()
```


------------------------------------------------------------------------


#### output_case_multiple_channel_periods()


Stitches two response periods together to make a single merged


Usage

``` python
output_case_multiple_channel_periods()
```


trace.
