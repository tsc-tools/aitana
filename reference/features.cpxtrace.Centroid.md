## features.cpxtrace.Centroid


Computes the centroid time of the given data which can be windowed or


Usage

``` python
features.cpxtrace.Centroid(
    interval=600.0,
    configfile=None,
)
```


not. The centroid time is determined as the time in the processed window where 50 per cent of the area below the envelope is reached.

The time derivative of the centroid time is returned if input data are windowed only.

## Parameters

- **fil_coef** : `` :class:`~numpy.ndarray `` -- Filter coefficients for computing time derivative.

## Returns

  
**DataFrame** - Time , Centroid time input data


## Methods

| Name | Description |
|----|----|
| [compute()](#compute) | :param trace: The seismic waveform data |

------------------------------------------------------------------------


#### compute()


## Parameters

- **trace** -- The seismic waveform data


Usage

``` python
compute(trace)
```


## Parameters

- **trace** : `obspy.Trace`
