## features.cpxtrace.Inst_Band


Computes the instantaneous bandwidth of the given data which can be


Usage

``` python
features.cpxtrace.Inst_Band(
    interval=600.0, fil_coef=(None, None, None), configfile=None
)
```


windowed or not. The instantaneous bandwidth is determined by the time derivative of the envelope normalized by the envelope of the input data.

## Parameters

- **fil_coef** : `` :class:`~numpy.ndarray `` -- Filter coefficients for computing time derivative.

## Returns

  
**DataFrame** - Time,Instantaneous bandwidth of input data


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
