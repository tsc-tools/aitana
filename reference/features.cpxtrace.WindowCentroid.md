## features.cpxtrace.WindowCentroid


Computes the centroid of a given time window,


Usage

``` python
features.cpxtrace.WindowCentroid(
    interval=600.0,
    configfile=None,
)
```


defined between 0 and 1. Deltas at t_start, t_end and t_midpoint give values of 0, 1 and 0.5 respectively. A flat signal has a centroid = 0.57.

## Parameters

- **interval** : `int, optional` -- Length of the time interval to compute feature.
