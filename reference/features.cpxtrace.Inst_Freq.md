## features.cpxtrace.Inst_Freq


Computes the instantaneous frequency of the given data which can be


Usage

``` python
features.cpxtrace.Inst_Freq(
    interval=600.0, fil_coef=(None, None, None), configfile=None
)
```


windowed or not. The instantaneous frequency is determined by the time derivative of the analytic signal of the input data.

## Parameters

- **fil_coef** : `` :class:`~numpy.ndarray `` -- Filter coefficients for computing time derivative.

## Returns

  
**DataFrame** - Time, Instantaneous frequency of input data .


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
