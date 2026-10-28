## features.spectral_features.SpectralFeatures


Usage

``` python
features.spectral_features.SpectralFeatures(
    interval=600.0,
    filtertype="highpass",
    filterfreq=(0.5, None),
    ncep=8,
    configfile=None
)
```


## Methods

| Name | Description |
|----|----|
| [bandwidth()](#bandwidth) | The Vanmarcke (1970) bandwidth parameter, q where |
| [centralFrequency()](#centralFrequency) | The 'central' frequency of the acceleration power |
| [compute()](#compute) | :param trace: The seismic waveform data |
| [predominantFrequency()](#predominantFrequency) | The peak frequency of the acceleration power spectral |

------------------------------------------------------------------------


#### bandwidth()


The Vanmarcke (1970) bandwidth parameter, q where


Usage

``` python
bandwidth(psd, freq)
```


0 \<= q \<= 1. Calculated using the spectral moments of the acceleration power spectral density function. 0 is a Dirac delta PSDF (sine wave), 1 is a flat PSDF (white signal).


------------------------------------------------------------------------


#### centralFrequency()


The 'central' frequency of the acceleration power


Usage

``` python
centralFrequency(psd, freq)
```


spectral density function (Hz). Defined as the square root of the ratio between the second and zero-th spectral moments. The spectral moments are calculated with log-spaced frequencies.


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

## Returns

  
DataFrame containing - central frequency (acceleration) in Hz - bandwidth parameter (acceleration) - predominant frequency (acceleration) in Hz - eight cepstral coefficients


------------------------------------------------------------------------


#### predominantFrequency()


The peak frequency of the acceleration power spectral


Usage

``` python
predominantFrequency(psd, freq)
```


density function, in Hz.
