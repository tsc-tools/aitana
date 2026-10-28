## features.util.window_array()


Take a 1-D numpy array and devide it into (overlapping)


Usage

``` python
features.util.window_array(
    x,
    nwin,
    noverlap,
    remove_mean=True,
    taper=True,
    padval=0,
    return_window=True
)
```


windows.

## Parameters

- **x** : `~numpy.ndarray` -- 1D array containing the data.
- **n** : `int` -- The number of data points in each window. If this is not a power of 2, the length of each window will be the nearest power of 2.
- **noverlap** : `int` -- The number of overlapping points between adjacent windows.
- **remove_mean** : `bool` -- Remove the individual mean from each window.
- **taper** : `bool` -- Apply a hanning taper to each window.
- **return_window** : `bool` -- If taper is 'True', return the values of the hanning taper.
- **padval** : `float, class:``numpy.nan``, or None` -- If this is not 'None', padval will be used to pad the input array such that its length is a multiple of the window length.

## Returns

`~numpy.ndarray`  
Array with the windowed data in columns.
