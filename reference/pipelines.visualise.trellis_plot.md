## pipelines.visualise.trellis_plot()


Create a trellis plot for the given models and data.


Usage

``` python
pipelines.visualise.trellis_plot(
    models,
    data,
    plot_uncertainty=False,
    groups=["b", "c", "d", "e"],
    q_min=0.15,
    q_max=0.85
)
```


## Parameters


`models: dict`  
A dictionary containing the models to plot. Each key should be a model name, and each value should be a dictionary with keys 'model', 'color', and optionally 'colorscale' for ensemble models. The 'model' key should contain a xarray DataArray.

`data: pd.DataFrame`  
A pandas DataFrame containing the data to plot. It should have a datetime index and a 'group' column indicating the group (b, c, d, e) for each time point.

`plot_uncertainty: bool = ``False`  
Whether to plot uncertainty for the models. Default is False.

`q_min: float = ``0.15`  
The minimum quantile to use for plotting uncertainty. Default is 0.15.

`q_max: float = ``0.85`  
The maximum quantile to use for plotting uncertainty. Default is 0.85.


## Returns


`fig: matplotlib.figure.Figure`  
A Matplotlib figure object containing the trellis plot.
