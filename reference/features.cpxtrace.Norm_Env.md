## features.cpxtrace.Norm_Env


Computes the normalized envelope of the given data which can be


Usage

``` python
features.cpxtrace.Norm_Env(
    window=10, interval=600.0, fil_coef=(None, None, None), configfile=None
)
```


windowed or not. In order to obtain a normalized measure of the signal envelope the instantaneous bandwidth of the smoothed envelope is normalized by the Nyquist frequency and is integrated afterwards.

The time derivative of the normalized envelope is returned if input data are windowed only.

## Parameters

- **fil_coef** : `` :class:`~numpy.ndarray `` -- Filter coefficients for computing time derivative.

## Returns

  
**DataFrame** - Time , Normalized envelope of input data


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
