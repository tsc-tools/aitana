## features.util.stride_windows()


Get all windows of x with length n as a single array,


Usage

``` python
features.util.stride_windows(
    x,
    n,
    noverlap,
)
```


using strides to avoid data duplication.

## Parameters

- **x** : `~numpy.ndarray` -- 1D array or sequence containing the data.
- **n** : `int` -- The number of data points in each window.
- **noverlap** : `int` -- The overlap between adjacent windows. Default is 0 (no overlap)

## Returns

`~numpy.ndarray`  
Array with the windowed data in columns.
