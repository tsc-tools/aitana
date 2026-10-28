## features.util.apply_hanning()


Apply a hanning window to the given 1D or 2D array along


Usage

``` python
features.util.apply_hanning(
    x,
    return_window=None,
)
```


the given axis.

## Parameters

- **x** : `numpy.ndarray` -- 1-D or 2-D array or sequence containing the data
- **return_window** : `bool` -- If true, also return the 1D values of the window that was applied.

## Returns

  
**wtr\[, wvals\]** input multiplied with the hanning window function, hanning window function
