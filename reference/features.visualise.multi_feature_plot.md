## features.visualise.multi_feature_plot()


Return a combined time-series plot of two features.


Usage

``` python
features.visualise.multi_feature_plot(
    feature1,
    feature2,
    equal_yrange=False,
    rangeslider=False,
    log=False,
    kws_ssam=None
)
```


## Parameters

- **feature1** : `str` -- First feature to plot with y-scale on the left.
- **feature2** : `str` -- Second feature to plot with y-scale on the right.
- **equal_yrange** : `bool` -- If True align yscales.
- **rangeslider** : `bool` -- If True show secondary panel for zooming.
- **kws_ssam** : `dict` -- Keywords to be passed to [aitana.features.visualise.plot_ssam_plotly](features.visualise.plot_ssam_plotly.md#aitana.features.visualise.plot_ssam_plotly)

## Returns

`plotly.graph_objs.Figure`  
plotly figure that can be displayed in a Jupyter notebook or in a dash app.
