## features.util.demean()


Return x minus the mean(x).


Usage

``` python
features.util.demean(
    x,
    axis=None,
)
```


## Parameters

- **x** : `~numpy.ndarray` -- array or sequence containing the data can have any dimensionality
- **axis** : `int` -- The axis along which to take the mean. See numpy.mean for a description of this argument.

## Returns

`~numpy.ndarray`  
input minus the arithmetic mean along the specified axis.
