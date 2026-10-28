## volcanoes.ruapehu.lake_height_scaling()


Calculate the scaling factor between the lake height sensors 08 and 03.


Usage

``` python
volcanoes.ruapehu.lake_height_scaling(recompute=False)
```


## Parameters

    recompute : bool
        If True, recompute the scaling factor. If False, load the
        scaling factor from the cache.


## Returns

    tuple
        A tuple containing the intercept and slope of the linear regression
        between the two sensors.
