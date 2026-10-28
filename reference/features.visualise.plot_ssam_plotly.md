## features.visualise.plot_ssam_plotly()


Plot spectrogram data using matplotlib.


Usage

``` python
features.visualise.plot_ssam_plotly(
    xdf,
    cmap="Ice_r",
    new_fig=True,
    **kwargs,
)
```


## Parameters

- **data** : `xarray.Dataset` -- An xarray dataset as returned by [aitana.features.ssam.SSAM](features.ssam.SSAM.md#aitana.features.ssam.SSAM)
- **log** : `bool` -- Logarithmic frequency axis if True, linear frequency axis otherwise.
- **dbscale** : `bool` -- If True 10 \* log10 of color values is taken, if False the sqrt is taken.
- **cmap** : `list` -- Specify a custom colormap instance from `plotly.colors.sequential`.
- **clip** : `[float, float]` -- adjust colormap to clip at lower and/or upper end. The given percentages of the amplitude range (linear or logarithmic depending on option `dbscale`) are clipped.
- **new_fig** : `boolean`
- **max_tpoints** : `int` -- Maximum number of time intervals. If the datasets exceeds that number it will be resampled to max_tpoints.

``` python
>>> import numpy as np
>>> from aitana.features.ssam import SSAM
>>> from aitana.testing import synthetic_trace
>>> from aitana.features.visualise import plot_ssam_plotly
>>> ft = SSAM(interval=60, per_lap=.9, smooth=10,                   frequencies=np.arange(0., 25.1, .1),                   timestamp='start')
>>> _ = ft.compute(synthetic_trace())
>>> fig = plot_ssam_plotly(ft.feature["ssam"])
```
