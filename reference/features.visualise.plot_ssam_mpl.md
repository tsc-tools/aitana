## features.visualise.plot_ssam_mpl()


Plot spectrogram data using matplotlib.


Usage

``` python
features.visualise.plot_ssam_mpl(
    xdf, axes=None, cmap=cm.viridis, figsize=(12, 6), **kwargs
)
```


## Parameters

- **xdf** : `xarray.Dataset` -- An xarray dataset as returned by [aitana.features.ssam.SSAM](features.ssam.SSAM.md#aitana.features.ssam.SSAM)
- **log** : `bool` -- Logarithmic frequency axis if True, linear frequency axis otherwise.
- **axes** : `matplotlib.axes.Axes` -- Plot into given axes.
- **dbscale** : `bool` -- If True 10 \* log10 of color values is taken, if False the sqrt is taken.
- **cmap** : `matplotlib.colors.Colormap` -- Specify a custom colormap instance.
- **clip** : `[float, float]` -- adjust colormap to clip at lower and/or upper end. The given percentages of the amplitude range (linear or logarithmic depending on option `dbscale`) are clipped.
- **figsize** : `[float, float]` -- Horizontal and vertical dimension of `matplotlib.figure.Figure`

``` python
>>> import numpy as np
>>> from aitana.features.ssam import SSAM
>>> from aitana.testing import synthetic_trace
>>> from aitana.features.visualise import plot_ssam_mpl
>>> ft = SSAM(interval=60, per_lap=.9, smooth=10,                   frequencies=np.arange(0., 25.1, .1),                   timestamp='start')
>>> _ = ft.compute(synthetic_trace())
>>> fig = plot_ssam_mpl(ft.feature["ssam"])
```
