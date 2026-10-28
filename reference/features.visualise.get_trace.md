## features.visualise.get_trace()


Return a plotly graphics object from a feature.


Usage

``` python
features.visualise.get_trace(
    data,
    colour,
    kws_ssam,
)
```


## Parameters

- **data** : `xarray.DataArray` -- Feature data to plot.
- **colour** : `str` -- Colour for line plots.

## Returns

`plotly.graph_objs.Scatter`` or ``plotly.graph_objs.Figure`  
A 1-D line plot or a 2D heatmap.
