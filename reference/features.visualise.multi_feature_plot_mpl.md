## features.visualise.multi_feature_plot_mpl()


Plot two features in one figure. 2D features have to be plotted first.


Usage

``` python
features.visualise.multi_feature_plot_mpl(
    feature1, feature2, figsize=(12, 5), **kwargs
)
```


## Parameters


`feature1: xarray.DataArray`  
First feature to plot. Can be 1D or 2D.

`feature2: xarray.DataArray`  
Second feature to plot. Has to be 1D.

`figsize: tuple = (12, 5)`  
Figure size, by default (12, 5)


## Returns


`matplotlib.figure.Figure`  
Figure with two y-axis.
